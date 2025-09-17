use thiserror::Error;

/// Comprehensive error types for VoiceStand
#[derive(Error, Debug)]
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

    #[error("Event send failed")]
    EventSendFailed,

    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),

    #[error("File I/O error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    #[error("JSON parsing error: {source}")]
    Json {
        #[from]
        source: serde_json::Error,
    },

    #[error("Channel receive error: {source}")]
    ChannelReceive {
        #[from]
        source: crossbeam_channel::RecvError,
    },

    #[error("Task join error: {source}")]
    TaskJoin {
        #[from]
        source: tokio::task::JoinError,
    },
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

    pub fn model_load_failed(msg: impl Into<String>) -> Self {
        Self::ModelLoadFailed(msg.into())
    }
}