use voicestand_core::{Result, VoiceStandError};
use dasp::{interpolate::linear::Linear, signal, Signal};

/// Audio processing utilities for noise reduction and enhancement
pub struct AudioProcessor {
    sample_rate: u32,
    frame_size: usize,
    noise_gate_threshold: f32,
    noise_reduction_factor: f32,
}

impl AudioProcessor {
    pub fn new(sample_rate: u32, frame_size: usize) -> Self {
        Self {
            sample_rate,
            frame_size,
            noise_gate_threshold: 0.01,
            noise_reduction_factor: 0.5,
        }
    }

    /// Apply noise gate to audio samples
    pub fn apply_noise_gate(&self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            if sample.abs() < self.noise_gate_threshold {
                *sample *= self.noise_reduction_factor;
            }
        }
    }

    /// Normalize audio levels
    pub fn normalize(&self, samples: &mut [f32]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }

        // Find peak amplitude
        let peak = samples.iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);

        if peak > 0.0 && peak != 1.0 {
            let scale = 0.95 / peak; // Leave some headroom
            for sample in samples.iter_mut() {
                *sample *= scale;
            }
        }

        Ok(())
    }

    /// Apply simple high-pass filter to remove DC offset and low-frequency noise
    pub fn high_pass_filter(&self, samples: &mut [f32], cutoff_freq: f32) -> Result<()> {
        if samples.len() < 2 {
            return Ok(());
        }

        let dt = 1.0 / self.sample_rate as f32;
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_freq);
        let alpha = rc / (rc + dt);

        let mut y_prev = samples[0];
        let mut x_prev = samples[0];

        for i in 1..samples.len() {
            let x = samples[i];
            let y = alpha * (y_prev + x - x_prev);
            samples[i] = y;

            y_prev = y;
            x_prev = x;
        }

        Ok(())
    }

    /// Resample audio to target sample rate
    pub fn resample(&self, input: &[f32], input_rate: u32, output_rate: u32) -> Result<Vec<f32>> {
        if input_rate == output_rate {
            return Ok(input.to_vec());
        }

        let ratio = output_rate as f64 / input_rate as f64;
        let output_len = (input.len() as f64 * ratio) as usize;

        // Simple linear interpolation using dasp
        let mut signal = signal::from_iter(input.iter().cloned());
        let interpolator = Linear::new(signal.next(), signal.next());
        let resampled = signal
            .interpolate(interpolator)
            .scale_hz(ratio)
            .take(output_len)
            .collect();

        Ok(resampled)
    }

    /// Apply pre-emphasis filter (commonly used for speech processing)
    pub fn pre_emphasis(&self, samples: &mut [f32], alpha: f32) -> Result<()> {
        if samples.len() < 2 {
            return Ok(());
        }

        for i in (1..samples.len()).rev() {
            samples[i] -= alpha * samples[i - 1];
        }

        Ok(())
    }

    /// Apply windowing function (Hamming window)
    pub fn apply_hamming_window(&self, samples: &mut [f32]) {
        let n = samples.len();
        if n <= 1 {
            return;
        }

        for (i, sample) in samples.iter_mut().enumerate() {
            let window_val = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
            *sample *= window_val;
        }
    }

    /// Calculate spectral centroid for speech quality assessment
    pub fn spectral_centroid(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        // Simple approximation using time-domain zero-crossing rate
        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        // Estimate frequency based on zero-crossings
        (crossings as f32 / 2.0) * (self.sample_rate as f32 / samples.len() as f32)
    }

    /// Enhance speech by reducing background noise
    pub fn enhance_speech(&self, samples: &mut [f32]) -> Result<()> {
        // Apply pre-emphasis
        self.pre_emphasis(samples, 0.97)?;

        // Apply high-pass filter to remove low-frequency noise
        self.high_pass_filter(samples, 80.0)?;

        // Apply noise gate
        self.apply_noise_gate(samples);

        // Normalize
        self.normalize(samples)?;

        Ok(())
    }

    /// Process audio chunk for optimal speech recognition
    pub fn process_for_recognition(&self, samples: &mut [f32]) -> Result<AudioStats> {
        let original_energy = self.calculate_energy(samples);

        // Enhance for speech recognition
        self.enhance_speech(samples)?;

        let processed_energy = self.calculate_energy(samples);
        let snr_improvement = if original_energy > 0.0 {
            20.0 * (processed_energy / original_energy).log10()
        } else {
            0.0
        };

        Ok(AudioStats {
            original_energy,
            processed_energy,
            snr_improvement,
            spectral_centroid: self.spectral_centroid(samples),
            clipping_detected: samples.iter().any(|&x| x.abs() >= 0.99),
        })
    }

    /// Calculate RMS energy
    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Update processing parameters
    pub fn update_parameters(&mut self, noise_gate_threshold: f32, noise_reduction_factor: f32) {
        self.noise_gate_threshold = noise_gate_threshold.clamp(0.0, 1.0);
        self.noise_reduction_factor = noise_reduction_factor.clamp(0.0, 1.0);
    }
}

/// Audio processing statistics
#[derive(Debug, Clone)]
pub struct AudioStats {
    pub original_energy: f32,
    pub processed_energy: f32,
    pub snr_improvement: f32,
    pub spectral_centroid: f32,
    pub clipping_detected: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_gate() {
        let mut processor = AudioProcessor::new(16000, 1024);
        processor.noise_gate_threshold = 0.1;
        processor.noise_reduction_factor = 0.5;

        let mut samples = vec![0.05, 0.15, -0.05, 0.2, -0.15];
        processor.apply_noise_gate(&mut samples);

        // Small samples should be reduced
        assert!(samples[0].abs() < 0.05);
        assert!(samples[2].abs() < 0.05);

        // Large samples should remain unchanged
        assert_eq!(samples[1], 0.15);
        assert_eq!(samples[3], 0.2);
        assert_eq!(samples[4], -0.15);
    }

    #[test]
    fn test_normalization() {
        let processor = AudioProcessor::new(16000, 1024);
        let mut samples = vec![0.1, 0.5, -0.3, 0.8, -0.6];

        processor.normalize(&mut samples).unwrap();

        // Check that peak is close to 0.95
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((peak - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_resampling() {
        let processor = AudioProcessor::new(16000, 1024);
        let input = vec![0.0, 1.0, 0.0, -1.0, 0.0]; // 5 samples

        // Downsample 2:1
        let resampled = processor.resample(&input, 8000, 4000).unwrap();
        assert!(resampled.len() < input.len());

        // Upsample 1:2
        let resampled = processor.resample(&input, 4000, 8000).unwrap();
        assert!(resampled.len() > input.len());
    }

    #[test]
    fn test_spectral_centroid() {
        let processor = AudioProcessor::new(16000, 1024);

        // High-frequency signal (more zero crossings)
        let high_freq: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.5).sin())
            .collect();

        // Low-frequency signal (fewer zero crossings)
        let low_freq: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();

        let high_centroid = processor.spectral_centroid(&high_freq);
        let low_centroid = processor.spectral_centroid(&low_freq);

        assert!(high_centroid > low_centroid);
    }
}