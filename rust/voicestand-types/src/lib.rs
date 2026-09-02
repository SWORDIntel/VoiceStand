//! Shared types for VoiceStand
//!
//! This crate contains common types used across all VoiceStand crates
//! to avoid circular dependencies.

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Comprehensive error types for VoiceStand
#[derive(Error, Debug, Clone)]
pub enum VoiceStandError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Speech recognition error: {0}")]
    Speech(String),

    #[error("GUI error: {0}")]
    Gui(String),

    #[error("System error: {0}")]
    System(String),

    #[error("State management error: {0}")]
    State(String),

    #[error("Hardware error: {0}")]
    Hardware(String),

    #[error("Initialization error: {0}")]
    Initialization(String),

    #[error("Event send failed")]
    EventSendFailed,

    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),

    #[error("File I/O error: {0}")]
    Io(String),

    #[error("JSON parsing error: {0}")]
    Json(String),

    #[error("Hardware not supported: {0}")]
    HardwareNotSupported(String),

    #[error("Lock poisoned: {0}")]
    LockPoisoned(String),

    #[error("Intel hardware error: {0}")]
    IntelHardware(String),

    #[error("NPU error: {0}")]
    NPU(String),

    #[error("GNA error: {0}")]
    GNA(String),
}

// Implement From conversions for common error types
impl From<std::io::Error> for VoiceStandError {
    fn from(err: std::io::Error) -> Self {
        VoiceStandError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for VoiceStandError {
    fn from(err: serde_json::Error) -> Self {
        VoiceStandError::Json(err.to_string())
    }
}

/// Result type alias for VoiceStand operations
pub type Result<T> = std::result::Result<T, VoiceStandError>;

impl VoiceStandError {
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn audio(msg: impl Into<String>) -> Self {
        Self::Audio(msg.into())
    }

    pub fn speech(msg: impl Into<String>) -> Self {
        Self::Speech(msg.into())
    }

    pub fn gui(msg: impl Into<String>) -> Self {
        Self::Gui(msg.into())
    }

    pub fn system(msg: impl Into<String>) -> Self {
        Self::System(msg.into())
    }

    pub fn state(msg: impl Into<String>) -> Self {
        Self::State(msg.into())
    }

    pub fn hardware(msg: impl Into<String>) -> Self {
        Self::Hardware(msg.into())
    }

    pub fn initialization(msg: impl Into<String>) -> Self {
        Self::Initialization(msg.into())
    }

    pub fn model_load_failed(msg: impl Into<String>) -> Self {
        Self::ModelLoadFailed(msg.into())
    }

    pub fn hardware_not_supported(msg: impl Into<String>) -> Self {
        Self::HardwareNotSupported(msg.into())
    }

    pub fn lock_poisoned(msg: impl Into<String>) -> Self {
        Self::LockPoisoned(msg.into())
    }

    pub fn intel_hardware(msg: impl Into<String>) -> Self {
        Self::IntelHardware(msg.into())
    }

    pub fn npu(msg: impl Into<String>) -> Self {
        Self::NPU(msg.into())
    }

    pub fn gna(msg: impl Into<String>) -> Self {
        Self::GNA(msg.into())
    }
}

// ============================================================================
// Audio Types
// ============================================================================

/// Audio configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub frames_per_buffer: u32,
    pub buffer_size: usize,
    pub vad_threshold: f32,
    pub device_name: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            frames_per_buffer: 1024,
            buffer_size: 480,
            vad_threshold: 0.3,
            device_name: None,
        }
    }
}

/// Audio capture device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCaptureConfig {
    pub device_name: Option<String>,
    pub sample_rate: u32,
    pub channels: u16,
    pub frames_per_buffer: u32,
    pub latency: f32,
}

impl Default for AudioCaptureConfig {
    fn default() -> Self {
        Self {
            device_name: None,
            sample_rate: 16_000,
            channels: 1,
            frames_per_buffer: 1024,
            latency: 0.1,
        }
    }
}

/// Audio device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioDevice {
    pub name: String,
    pub index: u32,
    pub channels: u16,
    pub sample_rate: u32,
    pub is_default: bool,
}

impl AudioDevice {
    pub fn new(
        name: String,
        index: u32,
        channels: u16,
        sample_rate: u32,
        is_default: bool,
    ) -> Self {
        Self {
            name,
            index,
            channels,
            sample_rate,
            is_default,
        }
    }
}

/// Audio data with metadata
#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub timestamp: SystemTime,
    pub is_speech_end: bool,
}

impl AudioData {
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            timestamp: SystemTime::now(),
            is_speech_end: false,
        }
    }

    pub fn duration(&self) -> Duration {
        let duration_secs =
            self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64);
        Duration::from_secs_f64(duration_secs)
    }

    pub fn with_speech_end(mut self, is_speech_end: bool) -> Self {
        self.is_speech_end = is_speech_end;
        self
    }
}

// ============================================================================
// Speech Recognition Types
// ============================================================================

/// Speech recognition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    pub model_path: String,
    pub language: String,
    pub num_threads: usize,
    pub use_gpu: bool,
    pub max_tokens: usize,
    pub beam_size: usize,
}

impl Default for SpeechConfig {
    fn default() -> Self {
        Self {
            model_path: "models/ggml-tiny.en.bin".to_string(),
            language: "auto".to_string(),
            num_threads: num_cpus::get().clamp(1, 4),
            use_gpu: false,
            max_tokens: 512,
            beam_size: 5,
        }
    }
}

/// Transcription result with confidence and timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub language: String,
    pub duration_ms: u32,
    pub meets_latency_target: bool,
}

impl TranscriptionResult {
    pub fn new(text: String, confidence: f32, language: String, duration_ms: u32) -> Self {
        Self {
            text,
            confidence,
            language,
            duration_ms,
            meets_latency_target: duration_ms <= 10,
        }
    }
}

// ============================================================================
// GUI Types
// ============================================================================

/// GUI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuiConfig {
    pub theme: String,
    pub show_waveform: bool,
    pub auto_scroll: bool,
    pub window_width: i32,
    pub window_height: i32,
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            theme: "system".to_string(),
            show_waveform: true,
            auto_scroll: true,
            window_width: 800,
            window_height: 600,
        }
    }
}

// ============================================================================
// Hotkey Types
// ============================================================================

/// Hotkey configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    pub toggle_recording: String,
    pub push_to_talk: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            toggle_recording: "Ctrl+Alt+Space".to_string(),
            push_to_talk: "Ctrl+Alt+V".to_string(),
        }
    }
}

// ============================================================================
// Event Types
// ============================================================================

/// Application events for communication between components
#[derive(Debug, Clone)]
pub enum AppEvent {
    /// Recording state changed
    RecordingStateChanged(bool),

    /// New audio data available
    AudioDataReceived(AudioData),

    /// Speech detected or ended
    SpeechDetected {
        is_start: bool,
        timestamp: SystemTime,
    },

    /// Transcription result available
    TranscriptionReceived(TranscriptionResult),

    /// Error occurred
    Error(String),

    /// Hotkey pressed
    HotkeyPressed(String),

    /// Configuration updated
    ConfigUpdated,

    /// Application shutdown requested
    Shutdown,

    /// GUI events
    GuiEvent(GuiEvent),
}

/// GUI-specific events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuiEvent {
    /// Show/hide main window
    ToggleWindow,

    /// Show settings dialog
    ShowSettings,

    /// Update transcription display
    UpdateTranscription {
        text: String,
        is_final: bool,
        confidence: f32,
    },

    /// Update waveform display
    UpdateWaveform(Vec<f32>),

    /// Update recording indicator
    UpdateRecordingStatus(bool),

    /// Clear transcription text
    ClearTranscription,

    /// Export transcription
    ExportTranscription(String),
}

impl AppEvent {
    pub fn is_audio_event(&self) -> bool {
        matches!(
            self,
            AppEvent::AudioDataReceived(_) | AppEvent::SpeechDetected { .. }
        )
    }

    pub fn is_transcription_event(&self) -> bool {
        matches!(self, AppEvent::TranscriptionReceived(_))
    }

    pub fn is_gui_event(&self) -> bool {
        matches!(self, AppEvent::GuiEvent(_))
    }

    pub fn is_error_event(&self) -> bool {
        matches!(self, AppEvent::Error(_))
    }
}

// ============================================================================
// System Status Types
// ============================================================================

/// System state enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SystemState {
    NotInitialized,
    Initializing,
    Ready,
    Recording,
    Processing,
    Error,
    ShuttingDown,
}

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub state: SystemState,
    pub components_active: u32,
    pub components_failed: u32,
    pub uptime: Duration,
    pub capabilities: Vec<String>,
}

// ============================================================================
// Voice Command Types
// ============================================================================

/// Voice command structure for command recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCommand {
    pub text: String,
    pub confidence: f32,
    pub timestamp: SystemTime,
    pub command_type: CommandType,
}

/// Command types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandType {
    Start,
    Stop,
    Pause,
    Resume,
    Custom(String),
}

// ============================================================================
// Voice Activity Detection Types
// ============================================================================

/// Voice activity detection state
#[derive(Debug, Clone)]
pub struct VadState {
    pub is_speaking: bool,
    pub consecutive_speech_frames: u32,
    pub consecutive_silence_frames: u32,
    pub speech_start_frame: u64,
    pub speech_end_frame: u64,
    pub frame_count: u64,
    pub energy_threshold: f32,
    pub min_speech_frames: u32,
    pub min_silence_frames: u32,
}

impl Default for VadState {
    fn default() -> Self {
        Self {
            is_speaking: false,
            consecutive_speech_frames: 0,
            consecutive_silence_frames: 0,
            speech_start_frame: 0,
            speech_end_frame: 0,
            frame_count: 0,
            energy_threshold: 0.01,
            min_speech_frames: 5,
            min_silence_frames: 10,
        }
    }
}

impl VadState {
    pub fn update(&mut self, energy: f32) -> bool {
        self.frame_count += 1;
        let is_speech = energy > self.energy_threshold;

        if is_speech {
            self.consecutive_speech_frames += 1;
            self.consecutive_silence_frames = 0;

            if !self.is_speaking && self.consecutive_speech_frames >= self.min_speech_frames {
                self.is_speaking = true;
                self.speech_start_frame = self.frame_count;
                return true; // Speech started
            }
        } else {
            self.consecutive_silence_frames += 1;
            self.consecutive_speech_frames = 0;

            if self.is_speaking && self.consecutive_silence_frames >= self.min_silence_frames {
                self.is_speaking = false;
                self.speech_end_frame = self.frame_count;
                return true; // Speech ended
            }
        }

        false // No state change
    }
}
