pub mod capture;
pub mod processing;
pub mod vad;
pub mod buffer;

pub use capture::*;
pub use processing::*;
pub use vad::*;
pub use buffer::*;

use voicestand_core::{Result, VoiceStandError};
use cpal::traits::{DeviceTrait, HostTrait};

/// Audio subsystem initialization
pub fn initialize_audio() -> Result<()> {
    tracing::info!("Initializing audio subsystem");

    // Initialize CPAL host
    let _host = cpal::default_host();

    tracing::info!("Audio subsystem initialized successfully");
    Ok(())
}

/// Get available audio devices
pub fn get_audio_devices() -> Result<Vec<AudioDeviceInfo>> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    // Input devices
    for device in host.input_devices().map_err(|e| VoiceStandError::audio(format!("Failed to enumerate input devices: {}", e)))? {
        let name = device.name().map_err(|e| VoiceStandError::audio(format!("Failed to get device name: {}", e)))?;
        devices.push(AudioDeviceInfo {
            name,
            is_input: true,
            is_default: false, // Will be updated below
        });
    }

    // Mark default input device
    if let Some(default_device) = host.default_input_device() {
        if let Ok(default_name) = default_device.name() {
            for device in &mut devices {
                if device.name == default_name && device.is_input {
                    device.is_default = true;
                    break;
                }
            }
        }
    }

    Ok(devices)
}

/// Audio device information
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_input: bool,
    pub is_default: bool,
}