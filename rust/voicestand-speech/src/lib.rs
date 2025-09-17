pub mod recognizer;
pub mod model;
pub mod postprocess;
pub mod features;

pub use recognizer::*;
pub use model::*;
pub use postprocess::*;
pub use features::*;

use voicestand_core::{Result, VoiceStandError, SpeechConfig};

/// Initialize speech recognition subsystem
pub async fn initialize_speech(config: &SpeechConfig) -> Result<()> {
    tracing::info!("Initializing speech recognition subsystem");

    // Check if model file exists
    let model_path = std::path::Path::new(&config.model_path);
    if !model_path.exists() {
        return Err(VoiceStandError::model_load_failed(
            format!("Model file not found: {}", config.model_path)
        ));
    }

    // Initialize Candle device
    let device = candle_core::Device::new_cuda(0)
        .or_else(|_| candle_core::Device::new_metal(0))
        .unwrap_or(candle_core::Device::Cpu);

    tracing::info!("Using device: {:?}", device);

    tracing::info!("Speech recognition subsystem initialized successfully");
    Ok(())
}

/// Get available speech recognition models
pub fn get_available_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            name: "tiny".to_string(),
            size_mb: 39,
            languages: vec!["multilingual".to_string()],
            description: "Fastest, lowest quality".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin".to_string(),
        },
        ModelInfo {
            name: "base".to_string(),
            size_mb: 142,
            languages: vec!["multilingual".to_string()],
            description: "Good balance of speed and quality".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin".to_string(),
        },
        ModelInfo {
            name: "small".to_string(),
            size_mb: 466,
            languages: vec!["multilingual".to_string()],
            description: "Better quality, slower".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin".to_string(),
        },
        ModelInfo {
            name: "medium".to_string(),
            size_mb: 1420,
            languages: vec!["multilingual".to_string()],
            description: "High quality, slower".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin".to_string(),
        },
        ModelInfo {
            name: "large".to_string(),
            size_mb: 2870,
            languages: vec!["multilingual".to_string()],
            description: "Highest quality, slowest".to_string(),
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin".to_string(),
        },
    ]
}

/// Information about available models
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub size_mb: u64,
    pub languages: Vec<String>,
    pub description: String,
    pub url: String,
}