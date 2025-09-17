pub mod npu;
pub mod gna;
pub mod cpu;
pub mod simd;
pub mod hardware_detection;
pub mod thermal_management;

pub use npu::*;
pub use gna::*;
pub use cpu::*;
pub use simd::*;
pub use hardware_detection::*;
pub use thermal_management::*;

use voicestand_core::{Result, VoiceStandError};
use std::sync::Arc;
use parking_lot::RwLock;

/// Intel Meteor Lake hardware acceleration manager
pub struct IntelAcceleration {
    pub npu_manager: Arc<RwLock<NPUManager>>,
    pub gna_controller: Arc<RwLock<GNAController>>,
    pub cpu_optimizer: Arc<RwLock<CPUOptimizer>>,
    pub thermal_manager: Arc<RwLock<ThermalManager>>,
    pub capabilities: HardwareCapabilities,
}

impl IntelAcceleration {
    /// Initialize Intel acceleration with hardware detection
    pub async fn new() -> Result<Self> {
        let capabilities = detect_hardware_capabilities()?;

        let npu_manager = if capabilities.has_npu {
            Arc::new(RwLock::new(NPUManager::new().await?))
        } else {
            return Err(VoiceStandError::HardwareNotSupported("NPU not available".into()));
        };

        let gna_controller = if capabilities.has_gna {
            Arc::new(RwLock::new(GNAController::new().await?))
        } else {
            return Err(VoiceStandError::HardwareNotSupported("GNA not available".into()));
        };

        let cpu_optimizer = Arc::new(RwLock::new(CPUOptimizer::new(&capabilities)?));
        let thermal_manager = Arc::new(RwLock::new(ThermalManager::new()?));

        Ok(Self {
            npu_manager,
            gna_controller,
            cpu_optimizer,
            thermal_manager,
            capabilities,
        })
    }

    /// Get hardware capabilities
    pub fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    /// Shutdown all acceleration components
    pub async fn shutdown(&self) -> Result<()> {
        // Shutdown in reverse order
        self.gna_controller.write().shutdown().await?;
        self.npu_manager.write().shutdown().await?;
        self.cpu_optimizer.write().reset_affinity()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_intel_acceleration_init() {
        let result = IntelAcceleration::new().await;
        // Test should succeed on Intel Meteor Lake systems
        if let Ok(intel_accel) = result {
            assert!(intel_accel.capabilities.has_npu || intel_accel.capabilities.has_gna);
        }
    }
}