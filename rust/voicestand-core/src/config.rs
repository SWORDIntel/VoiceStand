use crate::{AudioConfig, SpeechConfig, GuiConfig, HotkeyConfig, Result, VoiceStandError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use directories::ProjectDirs;

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceStandConfig {
    pub audio: AudioConfig,
    pub speech: SpeechConfig,
    pub gui: GuiConfig,
    pub hotkeys: HotkeyConfig,
}

impl Default for VoiceStandConfig {
    fn default() -> Self {
        Self {
            audio: AudioConfig::default(),
            speech: SpeechConfig::default(),
            gui: GuiConfig::default(),
            hotkeys: HotkeyConfig::default(),
        }
    }
}

impl VoiceStandConfig {
    /// Load configuration from default location
    pub fn load() -> Result<Self> {
        let config_path = Self::config_file_path()?;
        if config_path.exists() {
            Self::load_from_path(&config_path)
        } else {
            let config = Self::default();
            config.save()?;
            Ok(config)
        }
    }

    /// Load configuration from specific path
    pub fn load_from_path(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to default location
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_file_path()?;
        self.save_to_path(&config_path)
    }

    /// Save configuration to specific path
    pub fn save_to_path(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get the default configuration file path
    pub fn config_file_path() -> Result<PathBuf> {
        let project_dirs = ProjectDirs::from("", "", "VoiceStand")
            .ok_or_else(|| VoiceStandError::config("Failed to determine config directory"))?;

        let config_dir = project_dirs.config_dir();
        Ok(config_dir.join("config.json"))
    }

    /// Get the models directory path
    pub fn models_dir_path() -> Result<PathBuf> {
        let project_dirs = ProjectDirs::from("", "", "VoiceStand")
            .ok_or_else(|| VoiceStandError::config("Failed to determine config directory"))?;

        let config_dir = project_dirs.config_dir();
        Ok(config_dir.join("models"))
    }

    /// Get the data directory path
    pub fn data_dir_path() -> Result<PathBuf> {
        let project_dirs = ProjectDirs::from("", "", "VoiceStand")
            .ok_or_else(|| VoiceStandError::config("Failed to determine data directory"))?;

        Ok(project_dirs.data_dir().to_path_buf())
    }

    /// Validate configuration settings
    pub fn validate(&self) -> Result<()> {
        // Validate audio config
        if self.audio.sample_rate == 0 {
            return Err(VoiceStandError::config("Sample rate must be greater than 0"));
        }
        if self.audio.channels == 0 {
            return Err(VoiceStandError::config("Channels must be greater than 0"));
        }
        if self.audio.vad_threshold < 0.0 || self.audio.vad_threshold > 1.0 {
            return Err(VoiceStandError::config("VAD threshold must be between 0.0 and 1.0"));
        }

        // Validate speech config
        if self.speech.num_threads == 0 {
            return Err(VoiceStandError::config("Number of threads must be greater than 0"));
        }
        if self.speech.max_tokens == 0 {
            return Err(VoiceStandError::config("Max tokens must be greater than 0"));
        }
        if self.speech.beam_size == 0 {
            return Err(VoiceStandError::config("Beam size must be greater than 0"));
        }

        // Validate GUI config
        if self.gui.window_width <= 0 {
            return Err(VoiceStandError::config("Window width must be greater than 0"));
        }
        if self.gui.window_height <= 0 {
            return Err(VoiceStandError::config("Window height must be greater than 0"));
        }

        Ok(())
    }

    /// Update audio configuration
    pub fn update_audio(&mut self, audio: AudioConfig) -> Result<()> {
        self.audio = audio;
        self.validate()?;
        Ok(())
    }

    /// Update speech configuration
    pub fn update_speech(&mut self, speech: SpeechConfig) -> Result<()> {
        self.speech = speech;
        self.validate()?;
        Ok(())
    }

    /// Update GUI configuration
    pub fn update_gui(&mut self, gui: GuiConfig) -> Result<()> {
        self.gui = gui;
        self.validate()?;
        Ok(())
    }

    /// Update hotkey configuration
    pub fn update_hotkeys(&mut self, hotkeys: HotkeyConfig) -> Result<()> {
        self.hotkeys = hotkeys;
        self.validate()?;
        Ok(())
    }
}