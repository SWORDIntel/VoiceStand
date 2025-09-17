use voicestand_core::{Result, VoiceStandError};
use wide::{f32x8, f32x4, CmpGt, CmpLt};
use std::simd::f32x16;
use bytemuck;
use tracing::{info, debug, warn};

/// SIMD-accelerated audio processing for Intel Meteor Lake
pub struct SIMDAudioProcessor {
    simd_capability: SIMDCapability,
    alignment_buffer: Vec<f32>,
    temp_buffers: SIMDTempBuffers,
}

/// SIMD capabilities detected on the system
#[derive(Debug, Clone, Copy)]
pub enum SIMDCapability {
    SSE2,      // Baseline x86_64
    AVX,       // 128-bit operations
    AVX2,      // 256-bit operations (available on Meteor Lake)
    AVX512,    // 512-bit operations (hidden on Meteor Lake P-cores)
}

/// Temporary buffers for SIMD operations (aligned)
struct SIMDTempBuffers {
    aligned_input: Vec<f32>,
    aligned_output: Vec<f32>,
    aligned_coefficients: Vec<f32>,
    fft_buffer: Vec<f32>,
}

impl SIMDAudioProcessor {
    /// Initialize SIMD audio processor with capability detection
    pub fn new() -> Result<Self> {
        let simd_capability = Self::detect_simd_capability()?;
        info!("Detected SIMD capability: {:?}", simd_capability);

        let temp_buffers = SIMDTempBuffers {
            aligned_input: Vec::new(),
            aligned_output: Vec::new(),
            aligned_coefficients: Vec::new(),
            fft_buffer: Vec::new(),
        };

        Ok(Self {
            simd_capability,
            alignment_buffer: Vec::new(),
            temp_buffers,
        })
    }

    /// Detect available SIMD instruction sets
    fn detect_simd_capability() -> Result<SIMDCapability> {
        // Use cpuid to detect SIMD capabilities
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") {
            Ok(SIMDCapability::AVX512)
        } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            Ok(SIMDCapability::AVX2)
        } else if is_x86_feature_detected!("avx") {
            Ok(SIMDCapability::AVX)
        } else {
            Ok(SIMDCapability::SSE2) // Baseline for x86_64
        }
    }

    /// SIMD-accelerated noise gate with vectorized threshold comparison
    pub fn simd_noise_gate(&mut self, samples: &mut [f32], threshold: f32, reduction_factor: f32) -> Result<SIMDProcessingStats> {
        let start_time = std::time::Instant::now();
        let original_len = samples.len();

        match self.simd_capability {
            SIMDCapability::AVX2 => {
                self.avx2_noise_gate(samples, threshold, reduction_factor)?;
            }
            SIMDCapability::AVX => {
                self.avx_noise_gate(samples, threshold, reduction_factor)?;
            }
            _ => {
                self.scalar_noise_gate(samples, threshold, reduction_factor)?;
            }
        }

        let processing_time = start_time.elapsed();

        Ok(SIMDProcessingStats {
            samples_processed: original_len,
            processing_time_us: processing_time.as_micros() as u32,
            simd_lanes_used: self.get_simd_lanes(),
            throughput_msamples_per_sec: (original_len as f64 / processing_time.as_secs_f64()) / 1_000_000.0,
            simd_efficiency: self.calculate_simd_efficiency(original_len, processing_time),
        })
    }

    /// AVX2-accelerated noise gate (8x parallel f32 operations)
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_noise_gate(&mut self, samples: &mut [f32], threshold: f32, reduction_factor: f32) -> Result<()> {
        let len = samples.len();
        let simd_len = len / 8 * 8; // Process 8 samples at a time

        // Create threshold and reduction factor vectors
        let threshold_vec = f32x8::splat(threshold);
        let reduction_vec = f32x8::splat(reduction_factor);

        // Process 8 samples at a time with AVX2
        for i in (0..simd_len).step_by(8) {
            // Load 8 samples
            let samples_vec = f32x8::from(&samples[i..i+8]);

            // Compute absolute values
            let abs_samples = samples_vec.abs();

            // Create mask: true where |sample| < threshold
            let mask = abs_samples.cmp_lt(threshold_vec);

            // Apply reduction factor where mask is true, keep original where false
            let reduced_samples = samples_vec * reduction_vec;
            let result = mask.blend(reduced_samples, samples_vec);

            // Store results back
            result.write_to_slice(&mut samples[i..i+8]);
        }

        // Handle remaining samples (scalar processing)
        for sample in &mut samples[simd_len..] {
            if sample.abs() < threshold {
                *sample *= reduction_factor;
            }
        }

        Ok(())
    }

    /// AVX-accelerated noise gate (4x parallel f32 operations)
    #[target_feature(enable = "avx")]
    unsafe fn avx_noise_gate(&mut self, samples: &mut [f32], threshold: f32, reduction_factor: f32) -> Result<()> {
        let len = samples.len();
        let simd_len = len / 4 * 4; // Process 4 samples at a time

        // Create threshold and reduction factor vectors
        let threshold_vec = f32x4::splat(threshold);
        let reduction_vec = f32x4::splat(reduction_factor);

        // Process 4 samples at a time with AVX
        for i in (0..simd_len).step_by(4) {
            // Load 4 samples
            let samples_vec = f32x4::from(&samples[i..i+4]);

            // Compute absolute values
            let abs_samples = samples_vec.abs();

            // Create mask: true where |sample| < threshold
            let mask = abs_samples.cmp_lt(threshold_vec);

            // Apply reduction factor where mask is true
            let reduced_samples = samples_vec * reduction_vec;
            let result = mask.blend(reduced_samples, samples_vec);

            // Store results back
            result.write_to_slice(&mut samples[i..i+4]);
        }

        // Handle remaining samples
        for sample in &mut samples[simd_len..] {
            if sample.abs() < threshold {
                *sample *= reduction_factor;
            }
        }

        Ok(())
    }

    /// Scalar fallback for noise gate
    fn scalar_noise_gate(&mut self, samples: &mut [f32], threshold: f32, reduction_factor: f32) -> Result<()> {
        for sample in samples {
            if sample.abs() < threshold {
                *sample *= reduction_factor;
            }
        }
        Ok(())
    }

    /// SIMD-accelerated audio normalization
    pub fn simd_normalize(&mut self, samples: &mut [f32]) -> Result<SIMDProcessingStats> {
        let start_time = std::time::Instant::now();
        let original_len = samples.len();

        if samples.is_empty() {
            return Ok(SIMDProcessingStats::default());
        }

        // Find peak amplitude using SIMD
        let peak = self.simd_find_peak(samples)?;

        if peak > 0.0 && peak != 1.0 {
            let scale = 0.95 / peak; // Leave headroom

            match self.simd_capability {
                SIMDCapability::AVX2 => {
                    self.avx2_scale(samples, scale)?;
                }
                SIMDCapability::AVX => {
                    self.avx_scale(samples, scale)?;
                }
                _ => {
                    self.scalar_scale(samples, scale)?;
                }
            }
        }

        let processing_time = start_time.elapsed();

        Ok(SIMDProcessingStats {
            samples_processed: original_len,
            processing_time_us: processing_time.as_micros() as u32,
            simd_lanes_used: self.get_simd_lanes(),
            throughput_msamples_per_sec: (original_len as f64 / processing_time.as_secs_f64()) / 1_000_000.0,
            simd_efficiency: self.calculate_simd_efficiency(original_len, processing_time),
        })
    }

    /// Find peak amplitude using SIMD
    fn simd_find_peak(&self, samples: &[f32]) -> Result<f32> {
        match self.simd_capability {
            SIMDCapability::AVX2 => unsafe { self.avx2_find_peak(samples) },
            SIMDCapability::AVX => unsafe { self.avx_find_peak(samples) },
            _ => Ok(self.scalar_find_peak(samples)),
        }
    }

    /// AVX2 peak finding (8-way parallel)
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_find_peak(&self, samples: &[f32]) -> Result<f32> {
        let len = samples.len();
        let simd_len = len / 8 * 8;

        let mut max_vec = f32x8::splat(0.0);

        // Process 8 samples at a time
        for i in (0..simd_len).step_by(8) {
            let samples_vec = f32x8::from(&samples[i..i+8]);
            let abs_samples = samples_vec.abs();
            max_vec = max_vec.max(abs_samples);
        }

        // Horizontal maximum of the vector
        let max_array = max_vec.to_array();
        let mut peak = max_array.iter().fold(0.0f32, |a, &b| a.max(b));

        // Handle remaining samples
        for &sample in &samples[simd_len..] {
            peak = peak.max(sample.abs());
        }

        Ok(peak)
    }

    /// AVX peak finding (4-way parallel)
    #[target_feature(enable = "avx")]
    unsafe fn avx_find_peak(&self, samples: &[f32]) -> Result<f32> {
        let len = samples.len();
        let simd_len = len / 4 * 4;

        let mut max_vec = f32x4::splat(0.0);

        // Process 4 samples at a time
        for i in (0..simd_len).step_by(4) {
            let samples_vec = f32x4::from(&samples[i..i+4]);
            let abs_samples = samples_vec.abs();
            max_vec = max_vec.max(abs_samples);
        }

        // Horizontal maximum
        let max_array = max_vec.to_array();
        let mut peak = max_array.iter().fold(0.0f32, |a, &b| a.max(b));

        // Handle remaining samples
        for &sample in &samples[simd_len..] {
            peak = peak.max(sample.abs());
        }

        Ok(peak)
    }

    /// Scalar peak finding
    fn scalar_find_peak(&self, samples: &[f32]) -> f32 {
        samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
    }

    /// AVX2 scaling (8-way parallel multiplication)
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_scale(&mut self, samples: &mut [f32], scale: f32) -> Result<()> {
        let len = samples.len();
        let simd_len = len / 8 * 8;

        let scale_vec = f32x8::splat(scale);

        for i in (0..simd_len).step_by(8) {
            let samples_vec = f32x8::from(&samples[i..i+8]);
            let scaled = samples_vec * scale_vec;
            scaled.write_to_slice(&mut samples[i..i+8]);
        }

        // Handle remaining samples
        for sample in &mut samples[simd_len..] {
            *sample *= scale;
        }

        Ok(())
    }

    /// AVX scaling (4-way parallel multiplication)
    #[target_feature(enable = "avx")]
    unsafe fn avx_scale(&mut self, samples: &mut [f32], scale: f32) -> Result<()> {
        let len = samples.len();
        let simd_len = len / 4 * 4;

        let scale_vec = f32x4::splat(scale);

        for i in (0..simd_len).step_by(4) {
            let samples_vec = f32x4::from(&samples[i..i+4]);
            let scaled = samples_vec * scale_vec;
            scaled.write_to_slice(&mut samples[i..i+4]);
        }

        // Handle remaining samples
        for sample in &mut samples[simd_len..] {
            *sample *= scale;
        }

        Ok(())
    }

    /// Scalar scaling
    fn scalar_scale(&mut self, samples: &mut [f32], scale: f32) -> Result<()> {
        for sample in samples {
            *sample *= scale;
        }
        Ok(())
    }

    /// SIMD-accelerated high-pass filter with IIR implementation
    pub fn simd_high_pass_filter(&mut self, samples: &mut [f32], cutoff_freq: f32, sample_rate: f32) -> Result<SIMDProcessingStats> {
        let start_time = std::time::Instant::now();
        let original_len = samples.len();

        if samples.len() < 2 {
            return Ok(SIMDProcessingStats::default());
        }

        // Calculate filter coefficients
        let dt = 1.0 / sample_rate;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_freq);
        let alpha = rc / (rc + dt);

        // IIR filter implementation (sequential due to data dependencies)
        let mut y_prev = samples[0];
        let mut x_prev = samples[0];

        for i in 1..samples.len() {
            let x = samples[i];
            let y = alpha * (y_prev + x - x_prev);
            samples[i] = y;

            y_prev = y;
            x_prev = x;
        }

        let processing_time = start_time.elapsed();

        Ok(SIMDProcessingStats {
            samples_processed: original_len,
            processing_time_us: processing_time.as_micros() as u32,
            simd_lanes_used: 1, // IIR filters are inherently sequential
            throughput_msamples_per_sec: (original_len as f64 / processing_time.as_secs_f64()) / 1_000_000.0,
            simd_efficiency: 0.0, // No SIMD benefit for IIR filters
        })
    }

    /// SIMD-accelerated FIR filter with convolution
    pub fn simd_fir_filter(&mut self, samples: &mut [f32], coefficients: &[f32]) -> Result<SIMDProcessingStats> {
        let start_time = std::time::Instant::now();
        let original_len = samples.len();

        if samples.len() < coefficients.len() {
            return Ok(SIMDProcessingStats::default());
        }

        // Prepare aligned buffers
        self.prepare_aligned_buffers(samples.len(), coefficients.len())?;

        match self.simd_capability {
            SIMDCapability::AVX2 => {
                self.avx2_fir_filter(samples, coefficients)?;
            }
            SIMDCapability::AVX => {
                self.avx_fir_filter(samples, coefficients)?;
            }
            _ => {
                self.scalar_fir_filter(samples, coefficients)?;
            }
        }

        let processing_time = start_time.elapsed();

        Ok(SIMDProcessingStats {
            samples_processed: original_len,
            processing_time_us: processing_time.as_micros() as u32,
            simd_lanes_used: self.get_simd_lanes(),
            throughput_msamples_per_sec: (original_len as f64 / processing_time.as_secs_f64()) / 1_000_000.0,
            simd_efficiency: self.calculate_simd_efficiency(original_len, processing_time),
        })
    }

    /// AVX2 FIR filter implementation
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn avx2_fir_filter(&mut self, samples: &mut [f32], coefficients: &[f32]) -> Result<()> {
        let output_len = samples.len() - coefficients.len() + 1;

        for i in 0..output_len {
            let mut sum_vec = f32x8::splat(0.0);
            let coeff_len = coefficients.len();
            let simd_coeff_len = coeff_len / 8 * 8;

            // Vectorized dot product for filter coefficients
            for j in (0..simd_coeff_len).step_by(8) {
                let samples_vec = f32x8::from(&samples[i + j..i + j + 8]);
                let coeff_vec = f32x8::from(&coefficients[j..j + 8]);
                sum_vec = samples_vec.mul_add(coeff_vec, sum_vec); // FMA: sum += samples * coeff
            }

            // Horizontal sum of the vector
            let sum_array = sum_vec.to_array();
            let mut sum = sum_array.iter().sum::<f32>();

            // Handle remaining coefficients
            for j in simd_coeff_len..coeff_len {
                sum += samples[i + j] * coefficients[j];
            }

            self.temp_buffers.aligned_output[i] = sum;
        }

        // Copy results back (preserve original length by padding with zeros)
        samples[..output_len].copy_from_slice(&self.temp_buffers.aligned_output[..output_len]);
        for i in output_len..samples.len() {
            samples[i] = 0.0;
        }

        Ok(())
    }

    /// AVX FIR filter implementation
    #[target_feature(enable = "avx")]
    unsafe fn avx_fir_filter(&mut self, samples: &mut [f32], coefficients: &[f32]) -> Result<()> {
        let output_len = samples.len() - coefficients.len() + 1;

        for i in 0..output_len {
            let mut sum_vec = f32x4::splat(0.0);
            let coeff_len = coefficients.len();
            let simd_coeff_len = coeff_len / 4 * 4;

            // Vectorized dot product
            for j in (0..simd_coeff_len).step_by(4) {
                let samples_vec = f32x4::from(&samples[i + j..i + j + 4]);
                let coeff_vec = f32x4::from(&coefficients[j..j + 4]);
                sum_vec = sum_vec + (samples_vec * coeff_vec);
            }

            // Horizontal sum
            let sum_array = sum_vec.to_array();
            let mut sum = sum_array.iter().sum::<f32>();

            // Handle remaining coefficients
            for j in simd_coeff_len..coeff_len {
                sum += samples[i + j] * coefficients[j];
            }

            self.temp_buffers.aligned_output[i] = sum;
        }

        // Copy results back
        samples[..output_len].copy_from_slice(&self.temp_buffers.aligned_output[..output_len]);
        for i in output_len..samples.len() {
            samples[i] = 0.0;
        }

        Ok(())
    }

    /// Scalar FIR filter fallback
    fn scalar_fir_filter(&mut self, samples: &mut [f32], coefficients: &[f32]) -> Result<()> {
        let output_len = samples.len() - coefficients.len() + 1;

        for i in 0..output_len {
            let mut sum = 0.0;
            for j in 0..coefficients.len() {
                sum += samples[i + j] * coefficients[j];
            }
            self.temp_buffers.aligned_output[i] = sum;
        }

        samples[..output_len].copy_from_slice(&self.temp_buffers.aligned_output[..output_len]);
        for i in output_len..samples.len() {
            samples[i] = 0.0;
        }

        Ok(())
    }

    /// Prepare aligned buffers for SIMD operations
    fn prepare_aligned_buffers(&mut self, sample_len: usize, coeff_len: usize) -> Result<()> {
        let output_len = sample_len.max(coeff_len);

        // Ensure buffers are large enough
        if self.temp_buffers.aligned_output.len() < output_len {
            self.temp_buffers.aligned_output.resize(output_len, 0.0);
        }

        Ok(())
    }

    /// Get SIMD lane count for current capability
    fn get_simd_lanes(&self) -> u32 {
        match self.simd_capability {
            SIMDCapability::AVX512 => 16, // 512-bit / 32-bit = 16 lanes
            SIMDCapability::AVX2 => 8,   // 256-bit / 32-bit = 8 lanes
            SIMDCapability::AVX => 4,    // 128-bit / 32-bit = 4 lanes (actually 256-bit but conservative)
            SIMDCapability::SSE2 => 4,   // 128-bit / 32-bit = 4 lanes
        }
    }

    /// Calculate SIMD efficiency based on theoretical vs actual performance
    fn calculate_simd_efficiency(&self, samples_processed: usize, processing_time: std::time::Duration) -> f32 {
        let lanes = self.get_simd_lanes() as f64;
        let actual_throughput = samples_processed as f64 / processing_time.as_secs_f64();

        // Estimate theoretical peak throughput (simplified)
        let theoretical_peak = match self.simd_capability {
            SIMDCapability::AVX2 => 8_000_000_000.0, // 8 GFLOPS estimate for AVX2
            SIMDCapability::AVX => 4_000_000_000.0,  // 4 GFLOPS estimate for AVX
            _ => 1_000_000_000.0,                     // 1 GFLOPS estimate for SSE2
        };

        ((actual_throughput / theoretical_peak) * 100.0).min(100.0) as f32
    }

    /// Get current SIMD capability
    pub fn get_simd_capability(&self) -> SIMDCapability {
        self.simd_capability
    }

    /// Benchmark SIMD performance
    pub fn benchmark_simd_performance(&mut self, test_size: usize) -> Result<SIMDBenchmarkResult> {
        let mut test_data = vec![0.5f32; test_size];

        // Warm up
        for _ in 0..10 {
            self.simd_normalize(&mut test_data)?;
        }

        // Benchmark different operations
        let normalize_stats = self.benchmark_operation("normalize", &mut test_data.clone())?;
        let noise_gate_stats = self.benchmark_operation("noise_gate", &mut test_data.clone())?;

        Ok(SIMDBenchmarkResult {
            simd_capability: self.simd_capability,
            test_size,
            normalize_throughput: normalize_stats.throughput_msamples_per_sec,
            noise_gate_throughput: noise_gate_stats.throughput_msamples_per_sec,
            average_efficiency: (normalize_stats.simd_efficiency + noise_gate_stats.simd_efficiency) / 2.0,
            simd_lanes_used: self.get_simd_lanes(),
        })
    }

    /// Benchmark individual operation
    fn benchmark_operation(&mut self, operation: &str, data: &mut [f32]) -> Result<SIMDProcessingStats> {
        match operation {
            "normalize" => self.simd_normalize(data),
            "noise_gate" => self.simd_noise_gate(data, 0.01, 0.5),
            _ => Err(VoiceStandError::InvalidParameter(format!("Unknown operation: {}", operation))),
        }
    }
}

/// SIMD processing statistics
#[derive(Debug, Clone, Default)]
pub struct SIMDProcessingStats {
    pub samples_processed: usize,
    pub processing_time_us: u32,
    pub simd_lanes_used: u32,
    pub throughput_msamples_per_sec: f64,
    pub simd_efficiency: f32,
}

/// SIMD benchmark results
#[derive(Debug, Clone)]
pub struct SIMDBenchmarkResult {
    pub simd_capability: SIMDCapability,
    pub test_size: usize,
    pub normalize_throughput: f64,
    pub noise_gate_throughput: f64,
    pub average_efficiency: f32,
    pub simd_lanes_used: u32,
}

impl SIMDBenchmarkResult {
    /// Generate benchmark report
    pub fn generate_report(&self) -> String {
        format!(
            "Intel SIMD Benchmark Report\n\
             ===========================\n\
             SIMD Capability: {:?}\n\
             SIMD Lanes: {}\n\
             Test Size: {} samples\n\
             \n\
             Performance:\n\
             - Normalize: {:.2} MSamples/sec\n\
             - Noise Gate: {:.2} MSamples/sec\n\
             \n\
             Average Efficiency: {:.1}%\n\
             \n\
             Hardware Utilization: {}",
            self.simd_capability,
            self.simd_lanes_used,
            self.test_size,
            self.normalize_throughput,
            self.noise_gate_throughput,
            self.average_efficiency,
            if self.average_efficiency > 50.0 { "✅ GOOD" } else { "⚠️ SUBOPTIMAL" }
        )
    }
}