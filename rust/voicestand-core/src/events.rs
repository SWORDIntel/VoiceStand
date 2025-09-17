use crate::{AudioData, TranscriptionResult};
use serde::{Deserialize, Serialize};

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
        timestamp: std::time::SystemTime,
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
        matches!(self, AppEvent::AudioDataReceived(_) | AppEvent::SpeechDetected { .. })
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