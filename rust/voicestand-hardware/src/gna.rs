//! Intel GNA Hardware Abstraction
//!
//! Memory-safe wrapper for Intel Gaussian Neural Accelerator with RAII resource management.
//! Provides <100mW wake word detection with automatic cleanup.

use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, error, info, warn};

use crate::error::{HardwareError, HardwareResult, HardwareResultExt};
use crate::ffi::gna_bindings;
use crate::performance::{HardwareMetrics, PerformanceTracker};
use crate::HardwareResource;

/// GNA device configuration
#[derive(Debug, Clone)]
pub struct GNAConfig {
    /// Maximum power consumption in milliwatts
    pub max_power_consumption_mw: u32,
    /// Wake word detection threshold (0.0 - 1.0)
    pub detection_threshold: f32,
    /// Audio buffer size in samples
    pub audio_buffer_size: usize,
    /// Sample rate for audio processing
    pub sample_rate: u32,
    /// GNA device ID (for multi-GNA systems)
    pub device_id: u32,
    /// Wake words to detect
    pub wake_words: Vec<String>,
    /// Enable continuous detection mode
    pub continuous_mode: bool,
}

impl Default for GNAConfig {
    fn default() -> Self {
        Self {
            max_power_consumption_mw: 100,  // <100mW target
            detection_threshold: 0.8,      // 80% confidence
            audio_buffer_size: 480,        // 30ms at 16kHz
            sample_rate: 16000,            // 16kHz
            device_id: 0,
            wake_words: vec!["voicestand".to_string()],
            continuous_mode: true,
        }
    }
}

/// GNA device handle with RAII cleanup
pub struct GNADevice {
    device_handle: GNAHandle,
    config: GNAConfig,
    performance_tracker: Arc<PerformanceTracker>,
    detection_semaphore: Arc<Semaphore>,
    healthy: Arc<RwLock<bool>>,
    wake_word_models: Arc<RwLock<WakeWordModelCache>>,
}

/// Low-level GNA device handle
pub struct GNAHandle {
    device_ptr: *mut gna_bindings::GNADevice,
    device_id: u32,
}

unsafe impl Send for GNAHandle {}
unsafe impl Sync for GNAHandle {}

impl GNAHandle {
    /// Create new GNA handle
    fn new(device_id: u32) -> HardwareResult<Self> {
        let device_ptr = unsafe {
            gna_bindings::gna_device_create(device_id)
        };

        if device_ptr.is_null() {
            return Err(HardwareError::init_failed(
                "GNA",
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
            gna_bindings::gna_device_is_operational(self.device_ptr)
        }
    }

    /// Get device capabilities
    fn get_capabilities(&self) -> HardwareResult<GNACapabilities> {
        if self.device_ptr.is_null() {
            return Err(HardwareError::DeviceUnhealthy("GNA".to_string()));
        }

        let mut caps = gna_bindings::GNACapabilities::default();
        let result = unsafe {
            gna_bindings::gna_device_get_capabilities(self.device_ptr, &mut caps)
        };

        if result != 0 {
            return Err(HardwareError::driver_error(
                "GNA",
                result,
                "Failed to get device capabilities",
            ));
        }

        Ok(GNACapabilities {
            max_power_consumption_mw: caps.max_power_consumption_mw,
            min_detection_threshold: caps.min_detection_threshold,
            max_detection_threshold: caps.max_detection_threshold,
            supported_sample_rates: parse_sample_rate_flags(caps.supported_sample_rates),
            max_wake_words: caps.max_wake_words,
            memory_size_kb: caps.memory_size_kb,
        })
    }

    /// Get current power consumption
    fn get_power_consumption(&self) -> HardwareResult<f32> {
        if self.device_ptr.is_null() {
            return Err(HardwareError::DeviceUnhealthy("GNA".to_string()));
        }

        let power_mw = unsafe {
            gna_bindings::gna_device_get_power_consumption(self.device_ptr)
        };

        if power_mw < 0.0 {
            return Err(HardwareError::driver_error(
                "GNA",
                -1,
                "Failed to read power consumption",
            ));
        }

        Ok(power_mw)
    }
}

impl Drop for GNAHandle {
    fn drop(&mut self) {
        if !self.device_ptr.is_null() {
            unsafe {
                gna_bindings::gna_device_destroy(self.device_ptr);
            }
            self.device_ptr = std::ptr::null_mut();
        }
    }
}

/// GNA device capabilities
#[derive(Debug, Clone)]
pub struct GNACapabilities {
    pub max_power_consumption_mw: u32,
    pub min_detection_threshold: f32,
    pub max_detection_threshold: f32,
    pub supported_sample_rates: Vec<u32>,
    pub max_wake_words: u32,
    pub memory_size_kb: u32,
}

/// Wake word model cache
#[derive(Debug)]
struct WakeWordModelCache {
    models: std::collections::HashMap<String, CachedWakeWordModel>,
    max_models: usize,
}

#[derive(Debug)]
struct CachedWakeWordModel {
    model_ptr: *mut gna_bindings::GNAModel,
    wake_word: String,
    last_used: Instant,
    memory_usage_kb: u32,
}

impl WakeWordModelCache {
    fn new(max_models: usize) -> Self {
        Self {
            models: std::collections::HashMap::new(),
            max_models,
        }
    }

    fn get_or_load(&mut self, wake_word: &str) -> HardwareResult<*mut gna_bindings::GNAModel> {
        // Check if model is already cached
        if let Some(cached) = self.models.get_mut(wake_word) {
            cached.last_used = Instant::now();
            return Ok(cached.model_ptr);
        }

        // Load new wake word model
        let model_ptr = unsafe {
            gna_bindings::gna_wake_word_model_load(
                wake_word.as_ptr() as *const i8,
                wake_word.len(),
            )
        };

        if model_ptr.is_null() {
            return Err(HardwareError::model_error(
                "load",
                format!("Failed to load wake word model: {}", wake_word),
            ));
        }

        // Get model memory usage
        let memory_usage_kb = unsafe {
            gna_bindings::gna_model_get_memory_usage_kb(model_ptr)
        };

        // Evict old models if necessary
        self.evict_if_needed()?;

        // Cache the model
        let cached_model = CachedWakeWordModel {
            model_ptr,
            wake_word: wake_word.to_string(),
            last_used: Instant::now(),
            memory_usage_kb,
        };

        self.models.insert(wake_word.to_string(), cached_model);

        Ok(model_ptr)
    }

    fn evict_if_needed(&mut self) -> HardwareResult<()> {
        while self.models.len() >= self.max_models {
            // Find least recently used model
            let lru_key = self.models
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
        if let Some(model) = self.models.remove(key) {
            unsafe {
                gna_bindings::gna_model_destroy(model.model_ptr);
            }
            debug!("Evicted wake word model from cache: {} ({} KB)", key, model.memory_usage_kb);
        }

        Ok(())
    }
}

impl Drop for WakeWordModelCache {
    fn drop(&mut self) {
        // Clean up all cached models
        for (_, model) in self.models.drain() {
            unsafe {
                gna_bindings::gna_model_destroy(model.model_ptr);
            }
        }
    }
}

impl HardwareResource for GNADevice {
    type Config = GNAConfig;
    type Error = HardwareError;

    fn initialize(config: Self::Config) -> HardwareResult<Self> {
        info!("Initializing Intel GNA device {}", config.device_id);

        // Create device handle
        let device_handle = GNAHandle::new(config.device_id)?;

        // Verify device is operational
        if !device_handle.is_operational() {
            return Err(HardwareError::init_failed(
                "GNA",
                "Device is not operational",
            ));
        }

        // Get and validate capabilities
        let capabilities = device_handle.get_capabilities()?;
        validate_config_against_capabilities(&config, &capabilities)?;

        info!("✅ GNA device initialized successfully");
        debug!("GNA capabilities: {:?}", capabilities);

        Ok(Self {
            device_handle,
            config: config.clone(),
            performance_tracker: Arc::new(PerformanceTracker::new("GNA")),
            detection_semaphore: Arc::new(Semaphore::new(1)), // Only one detection at a time
            healthy: Arc::new(RwLock::new(true)),
            wake_word_models: Arc::new(RwLock::new(WakeWordModelCache::new(8))), // 8 wake word models max
        })
    }

    fn is_healthy(&self) -> bool {
        if !*self.healthy.read() || !self.device_handle.is_operational() {
            return false;
        }

        // Check power consumption
        if let Ok(power_mw) = self.device_handle.get_power_consumption() {
            power_mw <= self.config.max_power_consumption_mw as f32
        } else {
            false
        }
    }

    fn get_metrics(&self) -> HardwareMetrics {
        let mut metrics = self.performance_tracker.get_current_metrics();

        // Add GNA-specific metrics
        if let Ok(power_mw) = self.device_handle.get_power_consumption() {
            metrics.add_power_consumption("GNA", power_mw);
        }

        metrics
    }

    fn shutdown(&mut self) -> HardwareResult<()> {
        info!("Shutting down GNA device");

        // Mark as unhealthy
        *self.healthy.write() = false;

        // Clear wake word model cache
        {
            let mut cache = self.wake_word_models.write();
            for (_, model) in cache.models.drain() {
                unsafe {
                    gna_bindings::gna_model_destroy(model.model_ptr);
                }
            }
        }

        info!("✅ GNA device shutdown complete");
        Ok(())
    }
}

impl GNADevice {
    /// Create wake word detector for always-on detection
    pub fn create_wake_word_detector(&self) -> HardwareResult<WakeWordDetector> {
        if !self.is_healthy() {
            return Err(HardwareError::DeviceUnhealthy("GNA".to_string()));
        }

        WakeWordDetector::new(
            self.device_handle.device_ptr,
            self.config.clone(),
            Arc::clone(&self.detection_semaphore),
            Arc::clone(&self.performance_tracker),
            Arc::clone(&self.wake_word_models),
        )
    }
}

/// Wake word detector for always-on detection
pub struct WakeWordDetector {
    device_ptr: *mut gna_bindings::GNADevice,
    config: GNAConfig,
    detection_semaphore: Arc<Semaphore>,
    performance_tracker: Arc<PerformanceTracker>,
    wake_word_models: Arc<RwLock<WakeWordModelCache>>,
    detection_context: Option<DetectionContext>,
}

/// Detection context for active detection session
struct DetectionContext {
    session_ptr: *mut gna_bindings::GNADetectionSession,
    start_time: Instant,
}

impl Drop for DetectionContext {
    fn drop(&mut self) {
        if !self.session_ptr.is_null() {
            unsafe {
                gna_bindings::gna_detection_session_destroy(self.session_ptr);
            }
        }
    }
}

/// Wake word detection result
#[derive(Debug, Clone)]
pub struct WakeWordDetection {
    pub wake_word: String,
    pub confidence: f32,
    pub detection_time_ms: f32,
    pub power_consumption_mw: f32,
    pub meets_power_target: bool,
}

/// Detection event for async wake word monitoring
#[derive(Debug, Clone)]
pub enum DetectionEvent {
    /// Wake word detected
    WakeWordDetected {
        wake_word: String,
        confidence: f32,
        timestamp: Instant,
    },
    /// Detection error occurred
    Error { error: String },
    /// Detection session ended
    SessionEnded,
}

impl WakeWordDetector {
    fn new(
        device_ptr: *mut gna_bindings::GNADevice,
        config: GNAConfig,
        detection_semaphore: Arc<Semaphore>,
        performance_tracker: Arc<PerformanceTracker>,
        wake_word_models: Arc<RwLock<WakeWordModelCache>>,
    ) -> HardwareResult<Self> {
        Ok(Self {
            device_ptr,
            config,
            detection_semaphore,
            performance_tracker,
            wake_word_models,
            detection_context: None,
        })
    }

    /// Start continuous wake word detection
    pub async fn start_continuous_detection(&mut self) -> HardwareResult<mpsc::Receiver<DetectionEvent>> {
        let _permit = self.detection_semaphore
            .acquire()
            .await
            .map_err(|_| HardwareError::concurrency_error(
                "detection_semaphore",
                "Failed to acquire detection permit",
            ))?;

        // Load wake word models
        let model_ptrs = self.load_wake_word_models().await?;

        // Create detection session
        let session_ptr = unsafe {
            gna_bindings::gna_detection_session_create(
                self.device_ptr,
                model_ptrs.as_ptr(),
                model_ptrs.len(),
                self.config.detection_threshold,
                self.config.audio_buffer_size,
                self.config.sample_rate,
            )
        };

        if session_ptr.is_null() {
            return Err(HardwareError::init_failed(
                "GNA",
                "Failed to create detection session",
            ));
        }

        self.detection_context = Some(DetectionContext {
            session_ptr,
            start_time: Instant::now(),
        });

        // Create event channel
        let (tx, rx) = mpsc::channel(32);

        // Start detection loop in background task
        let device_ptr = self.device_ptr;
        let session_ptr_copy = session_ptr;
        let config = self.config.clone();
        let performance_tracker = Arc::clone(&self.performance_tracker);

        tokio::spawn(async move {
            Self::detection_loop(
                device_ptr,
                session_ptr_copy,
                config,
                performance_tracker,
                tx,
            ).await;
        });

        info!("🔊 Started continuous wake word detection");
        Ok(rx)
    }

    /// Detect wake word in audio buffer (single shot)
    pub async fn detect_wake_word(&mut self, audio_data: &[f32]) -> HardwareResult<Option<WakeWordDetection>> {
        let start_time = Instant::now();

        // Validate audio data
        if audio_data.len() != self.config.audio_buffer_size {
            return Err(HardwareError::audio_error(
                "validation",
                format!(
                    "Audio buffer size mismatch: got {}, expected {}",
                    audio_data.len(),
                    self.config.audio_buffer_size
                ),
            ));
        }

        // Load wake word models
        let model_ptrs = self.load_wake_word_models().await?;

        // Run detection
        let detection_start = Instant::now();
        let mut result = gna_bindings::GNADetectionResult::default();

        let detection_result = unsafe {
            gna_bindings::gna_detect_wake_word(
                self.device_ptr,
                model_ptrs.as_ptr(),
                model_ptrs.len(),
                audio_data.as_ptr(),
                audio_data.len(),
                self.config.detection_threshold,
                &mut result,
            )
        };

        let detection_time = detection_start.elapsed();
        let total_time = start_time.elapsed();

        // Check if wake word was detected
        if detection_result == 0 && result.detected {
            // Get wake word string
            let wake_word = unsafe {
                std::ffi::CStr::from_ptr(result.wake_word.as_ptr() as *const i8)
                    .to_string_lossy()
                    .into_owned()
            };

            // Get current power consumption
            let power_mw = unsafe {
                gna_bindings::gna_device_get_power_consumption(self.device_ptr)
            };

            let detection_time_ms = detection_time.as_secs_f32() * 1000.0;
            let meets_power_target = power_mw <= self.config.max_power_consumption_mw as f32;

            // Update performance metrics
            self.performance_tracker.record_wake_word_detection(
                total_time,
                detection_time,
                result.confidence,
                power_mw,
            );

            Ok(Some(WakeWordDetection {
                wake_word,
                confidence: result.confidence,
                detection_time_ms,
                power_consumption_mw: power_mw,
                meets_power_target,
            }))
        } else if detection_result != 0 {
            Err(HardwareError::model_error(
                "wake_word_detection",
                format!("Detection failed with code: {}", detection_result),
            ))
        } else {
            // No wake word detected
            Ok(None)
        }
    }

    /// Stop continuous detection
    pub async fn stop_detection(&mut self) -> HardwareResult<()> {
        if let Some(context) = self.detection_context.take() {
            // Detection context will be cleaned up in Drop
            let session_duration = context.start_time.elapsed();
            info!("🛑 Stopped wake word detection after {:.1}s", session_duration.as_secs_f32());
        }

        Ok(())
    }

    async fn load_wake_word_models(&self) -> HardwareResult<Vec<*mut gna_bindings::GNAModel>> {
        let mut model_ptrs = Vec::new();
        let mut cache = self.wake_word_models.write();

        for wake_word in &self.config.wake_words {
            let model_ptr = cache.get_or_load(wake_word)?;
            model_ptrs.push(model_ptr);
        }

        Ok(model_ptrs)
    }

    async fn detection_loop(
        device_ptr: *mut gna_bindings::GNADevice,
        session_ptr: *mut gna_bindings::GNADetectionSession,
        config: GNAConfig,
        performance_tracker: Arc<PerformanceTracker>,
        tx: mpsc::Sender<DetectionEvent>,
    ) {
        info!("Starting GNA detection loop");

        loop {
            // Check if session is still valid
            let session_active = unsafe {
                gna_bindings::gna_detection_session_is_active(session_ptr)
            };

            if !session_active {
                let _ = tx.send(DetectionEvent::SessionEnded).await;
                break;
            }

            // Poll for detection results
            let mut result = gna_bindings::GNADetectionResult::default();
            let poll_result = unsafe {
                gna_bindings::gna_detection_session_poll(session_ptr, &mut result)
            };

            if poll_result == 0 && result.detected {
                // Wake word detected
                let wake_word = unsafe {
                    std::ffi::CStr::from_ptr(result.wake_word.as_ptr() as *const i8)
                        .to_string_lossy()
                        .into_owned()
                };

                let detection_event = DetectionEvent::WakeWordDetected {
                    wake_word,
                    confidence: result.confidence,
                    timestamp: Instant::now(),
                };

                if tx.send(detection_event).await.is_err() {
                    // Receiver dropped, exit loop
                    break;
                }

                // Record performance metrics
                let power_mw = unsafe {
                    gna_bindings::gna_device_get_power_consumption(device_ptr)
                };

                performance_tracker.record_wake_word_detection(
                    Duration::from_millis(1), // Detection loop iteration
                    Duration::from_millis(1), // Actual detection time (estimated)
                    result.confidence,
                    power_mw,
                );
            } else if poll_result != 0 {
                // Error occurred
                let error_event = DetectionEvent::Error {
                    error: format!("Detection poll failed with code: {}", poll_result),
                };

                if tx.send(error_event).await.is_err() {
                    break;
                }
            }

            // Sleep for a short time to avoid busy polling
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        info!("GNA detection loop ended");
    }
}

/// Helper functions

fn validate_config_against_capabilities(
    config: &GNAConfig,
    capabilities: &GNACapabilities,
) -> HardwareResult<()> {
    // Check power consumption limit
    if config.max_power_consumption_mw > capabilities.max_power_consumption_mw {
        return Err(HardwareError::config_error(
            "max_power_consumption_mw",
            format!(
                "Requested {}mW exceeds GNA limit of {}mW",
                config.max_power_consumption_mw,
                capabilities.max_power_consumption_mw
            ),
        ));
    }

    // Check detection threshold range
    if config.detection_threshold < capabilities.min_detection_threshold
        || config.detection_threshold > capabilities.max_detection_threshold
    {
        return Err(HardwareError::config_error(
            "detection_threshold",
            format!(
                "Detection threshold {} outside valid range [{}, {}]",
                config.detection_threshold,
                capabilities.min_detection_threshold,
                capabilities.max_detection_threshold
            ),
        ));
    }

    // Check sample rate support
    if !capabilities.supported_sample_rates.contains(&config.sample_rate) {
        return Err(HardwareError::config_error(
            "sample_rate",
            format!(
                "Sample rate {}Hz not supported. Supported rates: {:?}",
                config.sample_rate,
                capabilities.supported_sample_rates
            ),
        ));
    }

    // Check wake word count
    if config.wake_words.len() > capabilities.max_wake_words as usize {
        return Err(HardwareError::config_error(
            "wake_words",
            format!(
                "Too many wake words: {} > {}",
                config.wake_words.len(),
                capabilities.max_wake_words
            ),
        ));
    }

    Ok(())
}

fn parse_sample_rate_flags(flags: u32) -> Vec<u32> {
    let mut rates = Vec::new();

    if flags & gna_bindings::GNA_SAMPLE_RATE_8000 != 0 {
        rates.push(8000);
    }
    if flags & gna_bindings::GNA_SAMPLE_RATE_16000 != 0 {
        rates.push(16000);
    }
    if flags & gna_bindings::GNA_SAMPLE_RATE_22050 != 0 {
        rates.push(22050);
    }
    if flags & gna_bindings::GNA_SAMPLE_RATE_44100 != 0 {
        rates.push(44100);
    }
    if flags & gna_bindings::GNA_SAMPLE_RATE_48000 != 0 {
        rates.push(48000);
    }

    rates
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gna_config_default() {
        let config = GNAConfig::default();
        assert_eq!(config.max_power_consumption_mw, 100);
        assert_eq!(config.detection_threshold, 0.8);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.wake_words, vec!["voicestand"]);
        assert!(config.continuous_mode);
    }

    #[test]
    fn test_sample_rate_parsing() {
        let flags = gna_bindings::GNA_SAMPLE_RATE_16000 | gna_bindings::GNA_SAMPLE_RATE_48000;
        let rates = parse_sample_rate_flags(flags);
        assert!(rates.contains(&16000));
        assert!(rates.contains(&48000));
        assert!(!rates.contains(&8000));
    }

    #[tokio::test]
    async fn test_wake_word_model_cache() {
        let mut cache = WakeWordModelCache::new(2);
        assert_eq!(cache.models.len(), 0);
        assert_eq!(cache.max_models, 2);
    }
}