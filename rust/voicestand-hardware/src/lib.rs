//! VoiceStand Hardware Abstraction Layer
//!
//! Memory-safe Rust abstractions for Intel NPU and GNA hardware access.
//! Provides RAII resource management and zero-unwrap error handling.

use std::sync::Arc;
// use thiserror::Error; // Unused - reserved for future error handling
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

pub mod npu;
pub mod gna;
pub mod ffi;
pub mod error;
pub mod performance;

pub use error::{HardwareError, HardwareResult};
pub use npu::{NPUDevice, NPUHandle, NPUProcessor};
pub use gna::{GNADevice, GNAHandle, WakeWordDetector};
pub use performance::{PerformanceMonitor, HardwareMetrics};

/// Hardware resource management trait
///
/// All hardware devices must implement proper RAII cleanup
pub trait HardwareResource {
    type Config;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Initialize hardware resource with configuration
    fn initialize(config: Self::Config) -> Result<Self, Self::Error>
    where
        Self: Sized;

    /// Check if hardware resource is healthy
    fn is_healthy(&self) -> bool;

    /// Get current resource metrics
    fn get_metrics(&self) -> HardwareMetrics;

    /// Safely shutdown and cleanup resources
    fn shutdown(&mut self) -> Result<(), Self::Error>;
}

/// Hardware Manager - Central coordination for NPU and GNA devices
pub struct HardwareManager {
    npu_device: Option<Arc<RwLock<NPUDevice>>>,
    gna_device: Option<Arc<RwLock<GNADevice>>>,
    performance_monitor: PerformanceMonitor,
    initialized: bool,
}

impl HardwareManager {
    /// Create new hardware manager instance
    pub fn new() -> Self {
        Self {
            npu_device: None,
            gna_device: None,
            performance_monitor: PerformanceMonitor::new(),
            initialized: false,
        }
    }

    /// Initialize all available hardware devices
    pub async fn initialize(&mut self) -> HardwareResult<()> {
        info!("Initializing VoiceStand hardware subsystem");

        // Initialize NPU device if available
        match NPUDevice::initialize(Default::default()) {
            Ok(npu) => {
                info!("✅ Intel NPU initialized successfully");
                self.npu_device = Some(Arc::new(RwLock::new(npu)));
            }
            Err(e) => {
                warn!("⚠️ NPU initialization failed: {}", e);
                debug!("Continuing without NPU - will use CPU fallback");
            }
        }

        // Initialize GNA device if available
        match GNADevice::initialize(Default::default()) {
            Ok(gna) => {
                info!("✅ Intel GNA initialized successfully");
                self.gna_device = Some(Arc::new(RwLock::new(gna)));
            }
            Err(e) => {
                warn!("⚠️ GNA initialization failed: {}", e);
                debug!("Continuing without GNA - will use key-only activation");
            }
        }

        // Ensure at least one hardware path is available
        if self.npu_device.is_none() && self.gna_device.is_none() {
            return Err(HardwareError::NoHardwareAvailable);
        }

        // Start performance monitoring
        self.performance_monitor.start().await?;
        self.initialized = true;

        info!("🚀 Hardware subsystem initialization complete");
        Ok(())
    }

    /// Get NPU processor handle (safe access)
    pub async fn get_npu_processor(&self) -> HardwareResult<NPUProcessor> {
        if !self.initialized {
            return Err(HardwareError::NotInitialized);
        }

        let npu_device = self.npu_device
            .as_ref()
            .ok_or(HardwareError::NPUNotAvailable)?;

        let device = npu_device.read().await;
        if !device.is_healthy() {
            return Err(HardwareError::DeviceUnhealthy("NPU".to_string()));
        }

        device.create_processor()
    }

    /// Get GNA wake word detector handle (safe access)
    pub async fn get_wake_word_detector(&self) -> HardwareResult<WakeWordDetector> {
        if !self.initialized {
            return Err(HardwareError::NotInitialized);
        }

        let gna_device = self.gna_device
            .as_ref()
            .ok_or(HardwareError::GNANotAvailable)?;

        let device = gna_device.read().await;
        if !device.is_healthy() {
            return Err(HardwareError::DeviceUnhealthy("GNA".to_string()));
        }

        device.create_wake_word_detector()
    }

    /// Check overall hardware health
    pub async fn check_health(&self) -> HardwareResult<HealthStatus> {
        if !self.initialized {
            return Ok(HealthStatus {
                npu_available: false,
                gna_available: false,
                npu_healthy: false,
                gna_healthy: false,
                overall_healthy: false,
            });
        }

        let mut npu_healthy = false;
        let mut gna_healthy = false;

        // Check NPU health
        if let Some(npu_device) = &self.npu_device {
            let device = npu_device.read().await;
            npu_healthy = device.is_healthy();
        }

        // Check GNA health
        if let Some(gna_device) = &self.gna_device {
            let device = gna_device.read().await;
            gna_healthy = device.is_healthy();
        }

        Ok(HealthStatus {
            npu_available: self.npu_device.is_some(),
            gna_available: self.gna_device.is_some(),
            npu_healthy,
            gna_healthy,
            overall_healthy: npu_healthy || gna_healthy,
        })
    }

    /// Get aggregated hardware metrics
    pub async fn get_metrics(&self) -> HardwareResult<HardwareMetrics> {
        let mut metrics = self.performance_monitor.get_current_metrics().await?;

        // Add NPU metrics if available
        if let Some(npu_device) = &self.npu_device {
            let device = npu_device.read().await;
            let npu_metrics = device.get_metrics();
            metrics.merge_npu_metrics(npu_metrics);
        }

        // Add GNA metrics if available
        if let Some(gna_device) = &self.gna_device {
            let device = gna_device.read().await;
            let gna_metrics = device.get_metrics();
            metrics.merge_gna_metrics(gna_metrics);
        }

        Ok(metrics)
    }

    /// Shutdown all hardware devices safely
    pub async fn shutdown(&mut self) -> HardwareResult<()> {
        info!("Shutting down hardware subsystem");

        // Stop performance monitoring first
        self.performance_monitor.stop().await?;

        // Shutdown GNA device
        if let Some(gna_device) = &self.gna_device {
            let mut device = gna_device.write().await;
            if let Err(e) = device.shutdown() {
                error!("Error shutting down GNA device: {}", e);
            }
        }

        // Shutdown NPU device
        if let Some(npu_device) = &self.npu_device {
            let mut device = npu_device.write().await;
            if let Err(e) = device.shutdown() {
                error!("Error shutting down NPU device: {}", e);
            }
        }

        self.npu_device = None;
        self.gna_device = None;
        self.initialized = false;

        info!("✅ Hardware subsystem shutdown complete");
        Ok(())
    }
}

impl Drop for HardwareManager {
    fn drop(&mut self) {
        if self.initialized {
            // Force synchronous shutdown in drop
            warn!("HardwareManager dropped while initialized - forcing shutdown");

            // Note: We can't call async shutdown() in Drop, so we log the issue
            // Proper shutdown should be called explicitly before dropping
        }
    }
}

/// Hardware health status
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub npu_available: bool,
    pub gna_available: bool,
    pub npu_healthy: bool,
    pub gna_healthy: bool,
    pub overall_healthy: bool,
}

impl HealthStatus {
    /// Check if system can perform voice-to-text
    pub fn can_transcribe(&self) -> bool {
        self.npu_healthy || self.npu_available  // NPU or CPU fallback
    }

    /// Check if system can detect wake words
    pub fn can_wake_word_detect(&self) -> bool {
        self.gna_healthy  // GNA required for wake word detection
    }

    /// Check if push-to-talk is available
    pub fn can_push_to_talk(&self) -> bool {
        true  // Always available via keyboard
    }

    /// Get capability summary string
    pub fn capabilities_summary(&self) -> String {
        let mut caps = Vec::new();

        if self.can_transcribe() {
            if self.npu_healthy {
                caps.push("NPU Voice-to-Text");
            } else {
                caps.push("CPU Voice-to-Text");
            }
        }

        if self.can_wake_word_detect() {
            caps.push("GNA Wake Words");
        }

        if self.can_push_to_talk() {
            caps.push("Push-to-Talk");
        }

        if caps.is_empty() {
            "No capabilities available".to_string()
        } else {
            caps.join(", ")
        }
    }
}

/// Default implementation for HardwareManager
impl Default for HardwareManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hardware_manager_creation() {
        let manager = HardwareManager::new();
        assert!(!manager.initialized);
        assert!(manager.npu_device.is_none());
        assert!(manager.gna_device.is_none());
    }

    #[tokio::test]
    async fn test_health_status_capabilities() {
        let health = HealthStatus {
            npu_available: true,
            gna_available: true,
            npu_healthy: true,
            gna_healthy: true,
            overall_healthy: true,
        };

        assert!(health.can_transcribe());
        assert!(health.can_wake_word_detect());
        assert!(health.can_push_to_talk());

        let summary = health.capabilities_summary();
        assert!(summary.contains("NPU Voice-to-Text"));
        assert!(summary.contains("GNA Wake Words"));
        assert!(summary.contains("Push-to-Talk"));
    }
}