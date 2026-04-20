use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use voicestand_core::{Result, SpeechConfig, VoiceStandError};

/// Runtime capability snapshot used to pick the lightest viable transcription strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProfile {
    pub operating_system: String,
    pub architecture: String,
    pub cpu_cores: usize,
    pub total_memory_mb: usize,
    pub supports_cuda: bool,
    pub supports_metal: bool,
    pub supports_openvino: bool,
    pub prefers_low_resource_mode: bool,
}

impl RuntimeProfile {
    pub fn detect() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let operating_system = std::env::consts::OS.to_string();
        let architecture = std::env::consts::ARCH.to_string();

        let total_memory_mb = Self::detect_memory_mb().unwrap_or(4096);

        let supports_cuda = std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
            || PathBuf::from("/dev/nvidiactl").exists();
        let supports_metal = operating_system == "macos";
        let supports_openvino = std::env::var("OPENVINO_LIB_PATH").is_ok()
            || PathBuf::from("/opt/intel/openvino").exists();

        // Keep resource usage low on small systems by default.
        let prefers_low_resource_mode = cpu_cores <= 4 || total_memory_mb <= 6 * 1024;

        Self {
            operating_system,
            architecture,
            cpu_cores,
            total_memory_mb,
            supports_cuda,
            supports_metal,
            supports_openvino,
            prefers_low_resource_mode,
        }
    }

    fn detect_memory_mb() -> Option<usize> {
        #[cfg(target_os = "linux")]
        {
            let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
            let line = meminfo.lines().find(|line| line.starts_with("MemTotal:"))?;
            let kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<usize>().ok())?;
            return Some(kb / 1024);
        }

        #[allow(unreachable_code)]
        None
    }

    /// Generate a platform-appropriate speech config using minimal resources by default.
    pub fn optimize_config(&self, mut base: SpeechConfig) -> SpeechConfig {
        base.num_threads = self.cpu_cores.clamp(1, 8);
        base.use_gpu = self.supports_cuda || self.supports_metal || self.supports_openvino;

        if self.prefers_low_resource_mode {
            base.max_tokens = base.max_tokens.min(192);
            base.beam_size = base.beam_size.min(2);
            if !base.model_path.contains("tiny") && !base.model_path.contains("base") {
                base.model_path = "models/ggml-base.bin".to_string();
            }
        } else {
            base.max_tokens = base.max_tokens.min(384);
            base.beam_size = base.beam_size.min(5);
        }

        base
    }
}

/// Persisted, user-specific voice calibration profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCalibrationProfile {
    pub version: u32,
    pub target_rms: f32,
    pub noise_floor: f32,
    pub speaking_rate_hint_hz: f32,
    pub preferred_vad_threshold: f32,
    pub samples_seen: usize,
}

impl Default for VoiceCalibrationProfile {
    fn default() -> Self {
        Self {
            version: 2,
            target_rms: 0.08,
            noise_floor: 0.01,
            speaking_rate_hint_hz: 4.0,
            preferred_vad_threshold: 0.30,
            samples_seen: 0,
        }
    }
}

impl VoiceCalibrationProfile {
    pub fn profile_path() -> Result<PathBuf> {
        let dirs = directories::ProjectDirs::from("", "", "VoiceStand")
            .ok_or_else(|| VoiceStandError::config("Failed to determine profile directory"))?;
        Ok(dirs.data_dir().join("voice_profile.json"))
    }

    pub fn load_or_default() -> Result<Self> {
        let path = Self::profile_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }

        let content = fs::read_to_string(&path)?;
        let mut parsed: Self = serde_json::from_str(&content)
            .map_err(|e| VoiceStandError::config(format!("Invalid profile JSON: {}", e)))?;

        // Lightweight migration path.
        if parsed.version < 2 {
            parsed.version = 2;
            parsed.preferred_vad_threshold = parsed.preferred_vad_threshold.clamp(0.10, 0.85);
        }

        Ok(parsed)
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::profile_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(self)
            .map_err(|e| VoiceStandError::config(format!("Failed to serialize profile: {}", e)))?;

        let tmp_path = path.with_extension("json.tmp");
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(content.as_bytes())?;
        file.sync_all()?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))?;
        }

        fs::rename(tmp_path, path)?;
        Ok(())
    }
}

/// Online learner that incrementally updates calibration after successful transcriptions.
#[derive(Debug, Clone)]
pub struct AdaptiveVoiceLearner {
    profile: VoiceCalibrationProfile,
    learning_rate: f32,
    rejected_updates: usize,
}

impl AdaptiveVoiceLearner {
    pub fn new(profile: VoiceCalibrationProfile) -> Self {
        Self {
            profile,
            learning_rate: 0.08,
            rejected_updates: 0,
        }
    }

    pub fn profile(&self) -> &VoiceCalibrationProfile {
        &self.profile
    }

    pub fn rejected_updates(&self) -> usize {
        self.rejected_updates
    }

    pub fn update_from_audio(&mut self, audio: &[f32], transcription_confidence: f32) {
        if audio.is_empty() || transcription_confidence < 0.55 {
            return;
        }

        let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt();
        let peak = audio
            .iter()
            .fold(0.0f32, |acc, sample| acc.max(sample.abs()));

        // Reject obviously clipped/noisy windows to reduce drift.
        if peak > 0.98 || rms > 0.60 {
            self.rejected_updates += 1;
            return;
        }

        let zero_crossings = audio
            .windows(2)
            .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
            .count() as f32;
        let speaking_rate_hint_hz = zero_crossings / (audio.len().max(1) as f32 / 16_000.0) / 2.0;

        let w = self.learning_rate;
        self.profile.target_rms =
            (((1.0 - w) * self.profile.target_rms) + (w * rms)).clamp(0.04, 0.25);
        self.profile.noise_floor =
            (((1.0 - w) * self.profile.noise_floor) + (w * (peak * 0.08))).clamp(0.002, 0.08);
        self.profile.speaking_rate_hint_hz = (((1.0 - w) * self.profile.speaking_rate_hint_hz)
            + (w * speaking_rate_hint_hz))
            .clamp(1.0, 12.0);

        self.profile.preferred_vad_threshold =
            (self.profile.noise_floor * 2.1 + 0.15).clamp(0.10, 0.80);
        self.profile.samples_seen += 1;
    }

    pub fn normalized_for_profile(&self, audio: &mut [f32]) {
        if audio.is_empty() {
            return;
        }

        let rms = (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt();
        if rms < 1e-5 {
            return;
        }

        let gain = (self.profile.target_rms / rms).clamp(0.5, 1.8);
        for sample in audio.iter_mut() {
            *sample = (*sample * gain).clamp(-1.0, 1.0);
        }
    }
}

/// Self-calibration helper used at first run or explicit user calibration.
#[derive(Debug, Default)]
pub struct CalibrationSession {
    frames: Vec<f32>,
}

impl CalibrationSession {
    pub fn ingest(&mut self, audio_frame: &[f32]) {
        self.frames.extend_from_slice(audio_frame);
    }

    pub fn is_ready(&self) -> bool {
        self.frames.len() >= 48_000
    }

    pub fn finish(self) -> VoiceCalibrationProfile {
        if self.frames.is_empty() {
            return VoiceCalibrationProfile::default();
        }

        let rms =
            (self.frames.iter().map(|v| v * v).sum::<f32>() / self.frames.len() as f32).sqrt();
        let peak = self
            .frames
            .iter()
            .fold(0.0f32, |acc, sample| acc.max(sample.abs()));

        VoiceCalibrationProfile {
            target_rms: rms.clamp(0.04, 0.20),
            noise_floor: (peak * 0.08).clamp(0.005, 0.08),
            preferred_vad_threshold: (peak * 0.15).clamp(0.15, 0.75),
            samples_seen: self.frames.len(),
            ..VoiceCalibrationProfile::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_profile_never_returns_zero_cores() {
        let runtime = RuntimeProfile::detect();
        assert!(runtime.cpu_cores >= 1);
    }

    #[test]
    fn learner_updates_profile_after_high_confidence_audio() {
        let mut learner = AdaptiveVoiceLearner::new(VoiceCalibrationProfile::default());
        let audio = vec![0.0, 0.2, -0.2, 0.15, -0.1, 0.0].repeat(10_000);
        learner.update_from_audio(&audio, 0.90);
        assert!(learner.profile().samples_seen > 0);
    }

    #[test]
    fn calibration_session_requires_enough_data() {
        let mut session = CalibrationSession::default();
        session.ingest(&vec![0.1f32; 16_000]);
        assert!(!session.is_ready());

        session.ingest(&vec![0.1f32; 32_000]);
        assert!(session.is_ready());
    }
}
