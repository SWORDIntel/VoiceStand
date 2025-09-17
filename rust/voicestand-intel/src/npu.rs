use voicestand_core::{Result, VoiceStandError};
use openvino::{Core, CompiledModel, InferRequest, Tensor, Shape, ElementType, Layout};
use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, warn, error};

/// Intel NPU (11 TOPS) manager for AI workloads
pub struct NPUManager {
    core: Core,
    current_model: Option<CompiledModel>,
    model_cache: Vec<CachedModel>,
    device_name: String,
    throughput_counter: Arc<RwLock<NPUThroughputCounter>>,
}

#[derive(Clone)]
struct CachedModel {
    name: String,
    model: CompiledModel,
    last_used: std::time::Instant,
    inference_time_ms: f32,
}

struct NPUThroughputCounter {
    total_inferences: u64,
    total_tokens: u64,
    start_time: std::time::Instant,
}

impl NPUManager {
    /// Initialize NPU with OpenVINO backend
    pub async fn new() -> Result<Self> {
        let mut core = Core::new(None)
            .map_err(|e| VoiceStandError::Hardware(format!("Failed to create OpenVINO core: {}", e)))?;

        // Check for NPU device
        let devices = core.get_available_devices()
            .map_err(|e| VoiceStandError::Hardware(format!("Failed to get devices: {}", e)))?;

        let npu_device = devices.iter()
            .find(|&device| device.contains("NPU"))
            .ok_or_else(|| VoiceStandError::HardwareNotSupported("NPU device not found".into()))?;

        info!("Found Intel NPU device: {}", npu_device);

        // Configure NPU for optimal performance
        core.set_property(npu_device, &[("PERFORMANCE_HINT", "LATENCY")])
            .map_err(|e| VoiceStandError::Hardware(format!("Failed to set NPU performance hint: {}", e)))?;

        Ok(Self {
            core,
            current_model: None,
            model_cache: Vec::with_capacity(8), // Cache up to 8 models
            device_name: npu_device.clone(),
            throughput_counter: Arc::new(RwLock::new(NPUThroughputCounter {
                total_inferences: 0,
                total_tokens: 0,
                start_time: std::time::Instant::now(),
            })),
        })
    }

    /// Load and compile Whisper model for NPU inference
    pub async fn load_whisper_model<P: AsRef<Path>>(&mut self, model_path: P) -> Result<()> {
        let model_path = model_path.as_ref();
        let model_name = model_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!("Loading Whisper model '{}' onto NPU", model_name);

        // Load model from IR files (.xml and .bin)
        let model = self.core.read_model_from_file(model_path, None)
            .map_err(|e| VoiceStandError::ModelLoad(format!("Failed to read model: {}", e)))?;

        // Compile for NPU with latency optimization
        let compiled_model = self.core.compile_model(&model, &self.device_name, &[
            ("PERFORMANCE_HINT", "LATENCY"),
            ("NUM_STREAMS", "1"), // Single stream for minimal latency
            ("CACHE_MODE", "OPTIMIZE_SIZE"),
        ])
        .map_err(|e| VoiceStandError::ModelCompile(format!("Failed to compile model: {}", e)))?;

        // Cache the compiled model
        let cached_model = CachedModel {
            name: model_name.clone(),
            model: compiled_model.clone(),
            last_used: std::time::Instant::now(),
            inference_time_ms: 0.0,
        };

        // Remove oldest model if cache is full
        if self.model_cache.len() >= 8 {
            self.model_cache.remove(0);
        }

        self.model_cache.push(cached_model);
        self.current_model = Some(compiled_model);

        info!("Successfully loaded Whisper model '{}' onto NPU", model_name);
        Ok(())
    }

    /// Perform NPU inference on audio features
    pub async fn infer_whisper(&self, mel_spectrogram: &[f32], tokens: Option<&[u32]>) -> Result<WhisperInferenceResult> {
        let model = self.current_model.as_ref()
            .ok_or_else(|| VoiceStandError::ModelNotLoaded("No Whisper model loaded".into()))?;

        let start_time = std::time::Instant::now();

        // Create inference request
        let mut infer_request = model.create_infer_request()
            .map_err(|e| VoiceStandError::Inference(format!("Failed to create infer request: {}", e)))?;

        // Prepare input tensor for mel spectrogram (typically [1, 80, 3000] for 30s audio)
        let mel_shape = Shape::new(&[1, 80, mel_spectrogram.len() / 80]);
        let mel_tensor = Tensor::new_with_data(ElementType::F32, &mel_shape, mel_spectrogram)
            .map_err(|e| VoiceStandError::Inference(format!("Failed to create mel tensor: {}", e)))?;

        // Set mel spectrogram input
        infer_request.set_tensor("mel", &mel_tensor)
            .map_err(|e| VoiceStandError::Inference(format!("Failed to set mel tensor: {}", e)))?;

        // Set tokens input if provided (for guided decoding)
        if let Some(token_data) = tokens {
            let token_shape = Shape::new(&[1, token_data.len()]);
            let token_tensor = Tensor::new_with_data(ElementType::I32, &token_shape,
                bytemuck::cast_slice(token_data))
                .map_err(|e| VoiceStandError::Inference(format!("Failed to create token tensor: {}", e)))?;

            infer_request.set_tensor("tokens", &token_tensor)
                .map_err(|e| VoiceStandError::Inference(format!("Failed to set token tensor: {}", e)))?;
        }

        // Run inference on NPU
        infer_request.infer()
            .map_err(|e| VoiceStandError::Inference(format!("NPU inference failed: {}", e)))?;

        // Get output logits
        let output_tensor = infer_request.get_tensor("logits")
            .map_err(|e| VoiceStandError::Inference(format!("Failed to get output tensor: {}", e)))?;

        let logits_data = output_tensor.get_data::<f32>()
            .map_err(|e| VoiceStandError::Inference(format!("Failed to get logits data: {}", e)))?;

        let inference_time = start_time.elapsed();

        // Update throughput counters
        {
            let mut counter = self.throughput_counter.write();
            counter.total_inferences += 1;
            counter.total_tokens += logits_data.len() as u64;
        }

        info!("NPU inference completed in {:.2}ms", inference_time.as_secs_f32() * 1000.0);

        Ok(WhisperInferenceResult {
            logits: logits_data.to_vec(),
            inference_time_ms: inference_time.as_secs_f32() * 1000.0,
            npu_utilization: self.estimate_npu_utilization().await,
            tokens_per_second: logits_data.len() as f32 / inference_time.as_secs_f32(),
        })
    }

    /// Switch between different language models
    pub async fn switch_language_model(&mut self, language: &str) -> Result<()> {
        let model_name = format!("whisper-{}", language);

        // Check if model is already cached
        if let Some(cached) = self.model_cache.iter_mut()
            .find(|m| m.name.contains(&model_name)) {

            cached.last_used = std::time::Instant::now();
            self.current_model = Some(cached.model.clone());
            info!("Switched to cached model: {}", model_name);
            return Ok(());
        }

        // Model not cached, would need to load from disk
        warn!("Language model '{}' not in cache, requires loading", model_name);
        Err(VoiceStandError::ModelNotFound(format!("Model {} not cached", model_name)))
    }

    /// Get NPU performance statistics
    pub async fn get_performance_stats(&self) -> NPUPerformanceStats {
        let counter = self.throughput_counter.read();
        let elapsed = counter.start_time.elapsed();

        NPUPerformanceStats {
            total_inferences: counter.total_inferences,
            total_tokens: counter.total_tokens,
            inferences_per_second: counter.total_inferences as f32 / elapsed.as_secs_f32(),
            tokens_per_second: counter.total_tokens as f32 / elapsed.as_secs_f32(),
            average_inference_time_ms: if counter.total_inferences > 0 {
                elapsed.as_secs_f32() * 1000.0 / counter.total_inferences as f32
            } else { 0.0 },
            npu_utilization_percent: self.estimate_npu_utilization().await,
            model_cache_size: self.model_cache.len(),
            uptime: elapsed,
        }
    }

    /// Estimate NPU utilization based on inference frequency
    async fn estimate_npu_utilization(&self) -> f32 {
        let counter = self.throughput_counter.read();
        let elapsed = counter.start_time.elapsed();

        if elapsed.as_secs() == 0 {
            return 0.0;
        }

        // Rough estimation: assume each inference uses NPU for 50ms on average
        let active_time = counter.total_inferences as f32 * 0.050; // 50ms per inference
        let utilization = (active_time / elapsed.as_secs_f32()).min(1.0);

        utilization * 100.0
    }

    /// Optimize NPU for specific workload type
    pub async fn optimize_for_workload(&mut self, workload: NPUWorkloadType) -> Result<()> {
        let (perf_hint, num_streams) = match workload {
            NPUWorkloadType::RealTime => ("LATENCY", "1"),
            NPUWorkloadType::Batch => ("THROUGHPUT", "4"),
            NPUWorkloadType::PowerEfficient => ("LATENCY", "1"), // Same as real-time for now
        };

        self.core.set_property(&self.device_name, &[
            ("PERFORMANCE_HINT", perf_hint),
            ("NUM_STREAMS", num_streams),
        ])
        .map_err(|e| VoiceStandError::Hardware(format!("Failed to optimize NPU: {}", e)))?;

        info!("NPU optimized for {:?} workload", workload);
        Ok(())
    }

    /// Shutdown NPU manager
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down NPU manager");
        self.model_cache.clear();
        self.current_model = None;
        Ok(())
    }
}

/// NPU workload optimization types
#[derive(Debug, Clone, Copy)]
pub enum NPUWorkloadType {
    RealTime,       // Minimize latency
    Batch,          // Maximize throughput
    PowerEfficient, // Balance performance and power
}

/// Whisper inference result from NPU
#[derive(Debug, Clone)]
pub struct WhisperInferenceResult {
    pub logits: Vec<f32>,
    pub inference_time_ms: f32,
    pub npu_utilization: f32,
    pub tokens_per_second: f32,
}

/// NPU performance statistics
#[derive(Debug, Clone)]
pub struct NPUPerformanceStats {
    pub total_inferences: u64,
    pub total_tokens: u64,
    pub inferences_per_second: f32,
    pub tokens_per_second: f32,
    pub average_inference_time_ms: f32,
    pub npu_utilization_percent: f32,
    pub model_cache_size: usize,
    pub uptime: std::time::Duration,
}

impl NPUPerformanceStats {
    /// Check if NPU is meeting performance targets
    pub fn meets_targets(&self) -> bool {
        self.average_inference_time_ms < 100.0 && // <100ms per inference
        self.npu_utilization_percent > 70.0 &&   // >70% utilization
        self.tokens_per_second > 50.0              // >50 tokens/sec
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "Intel NPU Performance Report\n\
             ============================\n\
             Inferences: {} total ({:.1}/sec)\n\
             Tokens: {} total ({:.1}/sec)\n\
             Average Latency: {:.2}ms\n\
             NPU Utilization: {:.1}%\n\
             Model Cache: {} models loaded\n\
             Uptime: {:.1}s\n\
             Target Compliance: {}",
            self.total_inferences,
            self.inferences_per_second,
            self.total_tokens,
            self.tokens_per_second,
            self.average_inference_time_ms,
            self.npu_utilization_percent,
            self.model_cache_size,
            self.uptime.as_secs_f32(),
            if self.meets_targets() { "✅ PASSED" } else { "❌ FAILED" }
        )
    }
}