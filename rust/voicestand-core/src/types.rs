use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Audio configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub frames_per_buffer: u32,
    pub vad_threshold: f32,
    pub device_name: Option<String>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            frames_per_buffer: 1024,
            vad_threshold: 0.3,
            device_name: None,
        }
    }
}

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
            model_path: "models/ggml-base.bin".to_string(),
            language: "auto".to_string(),
            num_threads: num_cpus::get(),
            use_gpu: false,
            max_tokens: 512,
            beam_size: 5,
        }
    }
}

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
        let duration_secs = self.samples.len() as f64 / (self.sample_rate as f64 * self.channels as f64);
        Duration::from_secs_f64(duration_secs)
    }

    pub fn with_speech_end(mut self, is_speech_end: bool) -> Self {
        self.is_speech_end = is_speech_end;
        self
    }
}

/// Transcription result with confidence and timing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub start_time: f64,
    pub end_time: f64,
    pub is_final: bool,
    pub language: Option<String>,
}

impl TranscriptionResult {
    pub fn new(text: String, confidence: f32, start_time: f64, end_time: f64, is_final: bool) -> Self {
        Self {
            text,
            confidence,
            start_time,
            end_time,
            is_final,
            language: None,
        }
    }

    pub fn with_language(mut self, language: String) -> Self {
        self.language = Some(language);
        self
    }

    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

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

/// System status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub is_listening: bool,
    pub is_processing: bool,
    pub audio_device_connected: bool,
    pub model_loaded: bool,
    pub last_error: Option<String>,
    pub uptime: Duration,
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
    pub fn new(name: String, index: u32, channels: u16, sample_rate: u32, is_default: bool) -> Self {
        Self {
            name,
            index,
            channels,
            sample_rate,
            is_default,
        }
    }
}

/// Audio sample type for type-safe audio processing
pub type AudioSample = f32;

/// Audio frame for pipeline processing
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub data: Vec<AudioSample>,
    pub sample_rate: u32,
    pub channels: u16,
    pub timestamp: SystemTime,
}