// Re-export error types from voicestand-types for backward compatibility
pub use voicestand_types::{VoiceStandError, Result};

// Legacy AudioError type - now part of VoiceStandError
pub use voicestand_types::VoiceStandError as AudioError;
