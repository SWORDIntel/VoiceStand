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

/// Audio-specific error types
#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Audio initialization failed: {0}")]
    InitializationFailed(String),

    #[error("Audio device error: {0}")]
    DeviceError(String),

    #[error("Audio processing error: {stage}: {reason}")]
    ProcessingError { stage: String, reason: String },

    #[error("Audio configuration error: {parameter}: {reason}")]
    ConfigError { parameter: String, reason: String },

    #[error("Audio buffer error: {0}")]
    BufferError(String),

    #[error("Audio format error: {0}")]
    FormatError(String),
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