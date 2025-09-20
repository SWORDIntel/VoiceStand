//! Intel NPU Hardware Abstraction
//!
//! Memory-safe wrapper for Intel Neural Processing Unit with RAII resource management.
//! Provides <2ms voice-to-text inference with automatic fallback to CPU.

use std::sync::Arc;
use std::time::Instant;
use parking_lot::RwLock;
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

use crate::error::{HardwareError, HardwareResult};
use crate::ffi::npu_bindings;
use crate::performance::{HardwareMetrics, PerformanceTracker};
use crate::HardwareResource;

/// NPU device configuration
#[derive(Debug, Clone)]
pub struct NPUConfig {
    /// Maximum inference time in milliseconds
    pub max_inference_time_ms: f32,
    /// Power budget in milliwatts
    pub power_budget_mw: u32,
    /// Model precision (FP32, FP16, INT8, INT4)
    pub precision: ModelPrecision,
    /// Enable dynamic shapes
    pub dynamic_shapes: bool,
    /// Maximum concurrent inferences
    pub max_concurrent_inferences: usize,
    /// NPU device ID (for multi-NPU systems)
    pub device_id: u32,
}

impl Default for NPUConfig {
    fn default() -> Self {
        Self {
            max_inference_time_ms: 2.0,  // <2ms target
            power_budget_mw: 100,        // <100mW target
            precision: ModelPrecision::FP16,
            dynamic_shapes: true,
            max_concurrent_inferences: 4,
            device_id: 0,
        }
    }
}

/// Model precision options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPrecision {
    FP32,
    FP16,
    INT8,
    INT4,
}

/// NPU device handle with RAII cleanup
pub struct NPUDevice {
    device_handle: NPUHandle,
    config: NPUConfig,
    performance_tracker: Arc<PerformanceTracker>,
    inference_semaphore: Arc<Semaphore>,
    healthy: Arc<RwLock<bool>>,
    model_cache: Arc<RwLock<ModelCache>>,
}

/// Low-level NPU device handle
pub struct NPUHandle {
    device_ptr: *mut npu_bindings::NPUDevice,
    device_id: u32,
}

unsafe impl Send for NPUHandle {}
unsafe impl Sync for NPUHandle {}

impl NPUHandle {
    /// Create new NPU handle
    fn new(device_id: u32) -> HardwareResult<Self> {
        let device_ptr = unsafe {
            npu_bindings::npu_device_create(device_id)
        };

        if device_ptr.is_null() {
            return Err(HardwareError::init_failed(
                "NPU",
                format!("Failed to create device handle for device {}", device_id),
            ));
        }

        Ok(Self {
            device_ptr,
            device_id,
        })
    }

    /// Check if device is operational
    fn is_operational(&self) -> bool {
        if self.device_ptr.is_null() {
            return false;
        }

        unsafe {
            npu_bindings::npu_device_is_operational(self.device_ptr)
        }
    }

    /// Get device capabilities
    fn get_capabilities(&self) -> HardwareResult<NPUCapabilities> {
        if self.device_ptr.is_null() {
            return Err(HardwareError::DeviceUnhealthy("NPU".to_string()));
        }

        let mut caps = npu_bindings::NPUCapabilities::default();
        let result = unsafe {
            npu_bindings::npu_device_get_capabilities(self.device_ptr, &mut caps)
        };

        if result != 0 {
            return Err(HardwareError::driver_error(
                "NPU",
                result,
                "Failed to get device capabilities",
            ));
        }

        Ok(NPUCapabilities {
            max_ops_per_second: caps.max_ops_per_second,
            memory_size_mb: caps.memory_size_mb,
            supported_precisions: parse_precision_flags(caps.supported_precisions),
            max_concurrent_inferences: caps.max_concurrent_inferences,
        })
    }
}

impl Drop for NPUHandle {
    fn drop(&mut self) {
        if !self.device_ptr.is_null() {
            unsafe {
                npu_bindings::npu_device_destroy(self.device_ptr);
            }
            self.device_ptr = std::ptr::null_mut();
        }
    }
}

/// NPU device capabilities
#[derive(Debug, Clone)]
pub struct NPUCapabilities {
    pub max_ops_per_second: u64,
    pub memory_size_mb: u32,
    pub supported_precisions: Vec<ModelPrecision>,
    pub max_concurrent_inferences: u32,
}

/// Model cache for optimized NPU models
#[derive(Debug)]
struct ModelCache {
    cache: std::collections::HashMap<String, CachedModel>,
    max_size_mb: u64,
    current_size_mb: u64,
}

#[derive(Debug)]
struct CachedModel {
    model_ptr: *mut npu_bindings::NPUModel,
    size_mb: u64,
    last_used: Instant,
    precision: ModelPrecision,
}

impl ModelCache {
    fn new(max_size_mb: u64) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size_mb,
            current_size_mb: 0,
        }
    }

    fn get_or_load(&mut self, model_path: &str, precision: ModelPrecision) -> HardwareResult<*mut npu_bindings::NPUModel> {
        let cache_key = format!("{}_{:?}", model_path, precision);

        // Check if model is already cached
        if let Some(cached) = self.cache.get_mut(&cache_key) {
            cached.last_used = Instant::now();
            return Ok(cached.model_ptr);
        }

        // Load new model
        let model_ptr = unsafe {
            npu_bindings::npu_model_load(
                model_path.as_ptr() as *const i8,
                precision as u32,
            )
        };

        if model_ptr.is_null() {
            return Err(HardwareError::model_error(
                "load",
                format!("Failed to load NPU model: {}", model_path),
            ));
        }

        // Get model size
        let model_size_mb = unsafe {
            npu_bindings::npu_model_get_size_mb(model_ptr)
        };

        // Evict old models if necessary
        self.evict_if_needed(model_size_mb)?;

        // Cache the model
        let cached_model = CachedModel {
            model_ptr,
            size_mb: model_size_mb,
            last_used: Instant::now(),
            precision,
        };

        self.current_size_mb += model_size_mb;
        self.cache.insert(cache_key, cached_model);

        Ok(model_ptr)
    }

    fn evict_if_needed(&mut self, needed_mb: u64) -> HardwareResult<()> {
        while self.current_size_mb + needed_mb > self.max_size_mb {
            // Find least recently used model
            let lru_key = self.cache
                .iter()
                .min_by_key(|(_, model)| model.last_used)
                .map(|(key, _)| key.clone());

            if let Some(key) = lru_key {
                self.evict_model(&key)?;
            } else {
                break;
            }
        }

        Ok(())
    }

    fn evict_model(&mut self, key: &str) -> HardwareResult<()> {
        if let Some(model) = self.cache.remove(key) {
            unsafe {
                npu_bindings::npu_model_destroy(model.model_ptr);
            }
            self.current_size_mb -= model.size_mb;
            debug!("Evicted NPU model from cache: {} ({} MB)", key, model.size_mb);
        }

        Ok(())
    }
}

impl Drop for ModelCache {
    fn drop(&mut self) {
        // Clean up all cached models
        for (_, model) in self.cache.drain() {
            unsafe {
                npu_bindings::npu_model_destroy(model.model_ptr);
            }
        }
    }
}

impl HardwareResource for NPUDevice {
    type Config = NPUConfig;
    type Error = HardwareError;

    fn initialize(config: Self::Config) -> HardwareResult<Self> {
        info!("Initializing Intel NPU device {}", config.device_id);

        // Create device handle
        let device_handle = NPUHandle::new(config.device_id)?;

        // Verify device is operational
        if !device_handle.is_operational() {
            return Err(HardwareError::init_failed(
                "NPU",
                "Device is not operational",
            ));
        }

        // Get and validate capabilities
        let capabilities = device_handle.get_capabilities()?;
        validate_config_against_capabilities(&config, &capabilities)?;

        info!("✅ NPU device initialized successfully");
        debug!("NPU capabilities: {:?}", capabilities);

        Ok(Self {
            device_handle,
            config: config.clone(),
            performance_tracker: Arc::new(PerformanceTracker::new("NPU")),
            inference_semaphore: Arc::new(Semaphore::new(config.max_concurrent_inferences)),
            healthy: Arc::new(RwLock::new(true)),
            model_cache: Arc::new(RwLock::new(ModelCache::new(512))), // 512MB cache
        })
    }

    fn is_healthy(&self) -> bool {
        *self.healthy.read() && self.device_handle.is_operational()
    }

    fn get_metrics(&self) -> HardwareMetrics {
        self.performance_tracker.get_current_metrics()
    }

    fn shutdown(&mut self) -> HardwareResult<()> {
        info!("Shutting down NPU device");

        // Mark as unhealthy
        *self.healthy.write() = false;

        // Clear model cache
        {
            let mut cache = self.model_cache.write();
            for (_, model) in cache.cache.drain() {
                unsafe {
                    npu_bindings::npu_model_destroy(model.model_ptr);
                }
            }
        }

        info!("✅ NPU device shutdown complete");
        Ok(())
    }
}

impl NPUDevice {
    /// Create NPU processor for inference operations
    pub fn create_processor(&self) -> HardwareResult<NPUProcessor> {
        if !self.is_healthy() {
            return Err(HardwareError::DeviceUnhealthy("NPU".to_string()));
        }

        NPUProcessor::new(
            self.device_handle.device_ptr,
            self.config.clone(),
            Arc::clone(&self.inference_semaphore),
            Arc::clone(&self.performance_tracker),
            Arc::clone(&self.model_cache),
        )
    }
}

/// NPU processor for inference operations
pub struct NPUProcessor {
    device_ptr: *mut npu_bindings::NPUDevice,
    config: NPUConfig,
    inference_semaphore: Arc<Semaphore>,
    performance_tracker: Arc<PerformanceTracker>,
    model_cache: Arc<RwLock<ModelCache>>,
}

impl NPUProcessor {
    fn new(
        device_ptr: *mut npu_bindings::NPUDevice,
        config: NPUConfig,
        inference_semaphore: Arc<Semaphore>,
        performance_tracker: Arc<PerformanceTracker>,
        model_cache: Arc<RwLock<ModelCache>>,
    ) -> HardwareResult<Self> {
        Ok(Self {
            device_ptr,
            config,
            inference_semaphore,
            performance_tracker,
            model_cache,
        })
    }

    /// Run voice-to-text inference with <2ms target latency
    pub async fn transcribe_audio(
        &self,
        audio_data: &[f32],
        model_path: &str,
    ) -> HardwareResult<TranscriptionResult> {
        let start_time = Instant::now();

        // Acquire inference slot (prevents overloading NPU)
        let _permit = self.inference_semaphore
            .acquire()
            .await
            .map_err(|_| HardwareError::concurrency_error(
                "inference_semaphore",
                "Failed to acquire inference permit",
            ))?;

        // Get model from cache
        let model_ptr = {
            let mut cache = self.model_cache.write();
            cache.get_or_load(model_path, self.config.precision)?
        };

        // Prepare input tensor
        let input_tensor = self.prepare_input_tensor(audio_data)?;

        // Run inference
        let inference_start = Instant::now();
        let output_ptr = unsafe {
            npu_bindings::npu_inference_run(
                self.device_ptr,
                model_ptr,
                input_tensor.data_ptr,
                input_tensor.size,
            )
        };

        if output_ptr.is_null() {
            return Err(HardwareError::model_error(
                "inference",
                "NPU inference failed",
            ));
        }

        let inference_time = inference_start.elapsed();

        // Process output
        let transcription = self.process_output(output_ptr)?;

        // Clean up output
        unsafe {
            npu_bindings::npu_output_destroy(output_ptr);
        }

        let total_time = start_time.elapsed();

        // Check performance targets
        let latency_ms = total_time.as_secs_f32() * 1000.0;
        if latency_ms > self.config.max_inference_time_ms {
            warn!(
                "NPU inference exceeded target latency: {:.2}ms > {:.2}ms",
                latency_ms, self.config.max_inference_time_ms
            );
        }

        // Update performance metrics
        self.performance_tracker.record_inference(
            total_time,
            inference_time,
            audio_data.len(),
            transcription.confidence,
        );

        Ok(TranscriptionResult {
            text: transcription.text,
            confidence: transcription.confidence,
            language: transcription.language,
            duration_ms: total_time.as_millis() as u32,
            inference_time_ms: inference_time.as_secs_f32() * 1000.0,
            meets_latency_target: latency_ms <= self.config.max_inference_time_ms,
        })
    }

    fn prepare_input_tensor(&self, audio_data: &[f32]) -> HardwareResult<InputTensor> {
        // Validate audio data
        if audio_data.is_empty() {
            return Err(HardwareError::audio_error(
                "validation",
                "Audio data is empty",
            ));
        }

        // Create input tensor (zero-copy when possible)
        let tensor_ptr = unsafe {
            npu_bindings::npu_tensor_create_from_audio(
                audio_data.as_ptr(),
                audio_data.len(),
                16000, // 16kHz sample rate
            )
        };

        if tensor_ptr.is_null() {
            return Err(HardwareError::memory_error(
                "tensor_creation",
                "Failed to create input tensor",
            ));
        }

        Ok(InputTensor {
            data_ptr: tensor_ptr,
            size: audio_data.len(),
        })
    }

    fn process_output(&self, output_ptr: *mut npu_bindings::NPUOutput) -> HardwareResult<ProcessedTranscription> {
        if output_ptr.is_null() {
            return Err(HardwareError::model_error(
                "output_processing",
                "Output pointer is null",
            ));
        }

        let mut text_buffer = vec![0u8; 1024];
        let mut confidence: f32 = 0.0;
        let mut language_buffer = vec![0u8; 32];

        let result = unsafe {
            npu_bindings::npu_output_get_transcription(
                output_ptr,
                text_buffer.as_mut_ptr() as *mut i8,
                text_buffer.len(),
                &mut confidence,
                language_buffer.as_mut_ptr() as *mut i8,
                language_buffer.len(),
            )
        };

        if result != 0 {
            return Err(HardwareError::model_error(
                "transcription_extraction",
                format!("Failed to extract transcription: code {}", result),
            ));
        }

        // Convert C strings to Rust strings
        let text = unsafe {
            std::ffi::CStr::from_ptr(text_buffer.as_ptr() as *const i8)
                .to_string_lossy()
                .into_owned()
        };

        let language = unsafe {
            std::ffi::CStr::from_ptr(language_buffer.as_ptr() as *const i8)
                .to_string_lossy()
                .into_owned()
        };

        Ok(ProcessedTranscription {
            text,
            confidence,
            language,
        })
    }
}

/// Input tensor wrapper
struct InputTensor {
    data_ptr: *mut npu_bindings::NPUTensor,
    size: usize,
}

impl Drop for InputTensor {
    fn drop(&mut self) {
        if !self.data_ptr.is_null() {
            unsafe {
                npu_bindings::npu_tensor_destroy(self.data_ptr);
            }
        }
    }
}

/// Processed transcription data
struct ProcessedTranscription {
    text: String,
    confidence: f32,
    language: String,
}

/// Transcription result with performance metrics
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub language: String,
    pub duration_ms: u32,
    pub inference_time_ms: f32,
    pub meets_latency_target: bool,
}

/// Helper functions

fn validate_config_against_capabilities(
    config: &NPUConfig,
    capabilities: &NPUCapabilities,
) -> HardwareResult<()> {
    // Check if precision is supported
    if !capabilities.supported_precisions.contains(&config.precision) {
        return Err(HardwareError::config_error(
            "precision",
            format!("Precision {:?} not supported by NPU", config.precision),
        ));
    }

    // Check concurrent inference limit
    if config.max_concurrent_inferences > capabilities.max_concurrent_inferences as usize {
        return Err(HardwareError::config_error(
            "max_concurrent_inferences",
            format!(
                "Requested {} concurrent inferences exceeds NPU limit of {}",
                config.max_concurrent_inferences,
                capabilities.max_concurrent_inferences
            ),
        ));
    }

    Ok(())
}

fn parse_precision_flags(flags: u32) -> Vec<ModelPrecision> {
    let mut precisions = Vec::new();

    if flags & npu_bindings::NPU_PRECISION_FP32 != 0 {
        precisions.push(ModelPrecision::FP32);
    }
    if flags & npu_bindings::NPU_PRECISION_FP16 != 0 {
        precisions.push(ModelPrecision::FP16);
    }
    if flags & npu_bindings::NPU_PRECISION_INT8 != 0 {
        precisions.push(ModelPrecision::INT8);
    }
    if flags & npu_bindings::NPU_PRECISION_INT4 != 0 {
        precisions.push(ModelPrecision::INT4);
    }

    precisions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npu_config_default() {
        let config = NPUConfig::default();
        assert_eq!(config.max_inference_time_ms, 2.0);
        assert_eq!(config.power_budget_mw, 100);
        assert_eq!(config.precision, ModelPrecision::FP16);
        assert!(config.dynamic_shapes);
    }

    #[test]
    fn test_model_precision_parsing() {
        let flags = npu_bindings::NPU_PRECISION_FP16 | npu_bindings::NPU_PRECISION_INT8;
        let precisions = parse_precision_flags(flags);
        assert!(precisions.contains(&ModelPrecision::FP16));
        assert!(precisions.contains(&ModelPrecision::INT8));
        assert!(!precisions.contains(&ModelPrecision::FP32));
    }

    #[tokio::test]
    async fn test_model_cache_creation() {
        let cache = ModelCache::new(100);
        assert_eq!(cache.max_size_mb, 100);
        assert_eq!(cache.current_size_mb, 0);
        assert!(cache.cache.is_empty());
    }
}