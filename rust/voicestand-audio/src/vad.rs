use crate::AudioError;
use std::time::Instant;
use tracing::{debug, warn};

/// VAD Configuration
#[derive(Debug, Clone)]
pub struct VADConfig {
    pub sample_rate: u32,
    pub frame_size: usize,
    pub energy_threshold: f32,
    pub silence_duration_ms: u64,
    pub voice_duration_ms: u64,
}

/// VAD internal state
#[derive(Debug, Clone)]
struct VADState {
    pub is_speaking: bool,
    pub energy_threshold: f32,
    pub min_speech_frames: u32,
    pub min_silence_frames: u32,
    pub speech_frames: u32,
    pub silence_frames: u32,
}

impl Default for VADState {
    fn default() -> Self {
        Self {
            is_speaking: false,
            energy_threshold: 0.02,
            min_speech_frames: 3,
            min_silence_frames: 10,
            speech_frames: 0,
            silence_frames: 0,
        }
    }
}

impl VADState {
    fn update(&mut self, energy: f32) -> bool {
        let is_above_threshold = energy > self.energy_threshold;
        let state_changed;

        if is_above_threshold {
            self.speech_frames += 1;
            self.silence_frames = 0;

            if !self.is_speaking && self.speech_frames >= self.min_speech_frames {
                self.is_speaking = true;
                state_changed = true;
                debug!("VAD: Speech detected (energy: {:.6})", energy);
            } else {
                state_changed = false;
            }
        } else {
            self.silence_frames += 1;
            self.speech_frames = 0;

            if self.is_speaking && self.silence_frames >= self.min_silence_frames {
                self.is_speaking = false;
                state_changed = true;
                debug!("VAD: Silence detected");
            } else {
                state_changed = false;
            }
        }

        state_changed
    }
}

/// Voice Activity Detection implementation
pub struct VoiceActivityDetector {
    config: VADConfig,
    state: VADState,
    window_size: usize,
    energy_history: Vec<f32>,
    zcr_history: Vec<f32>,
    spectral_centroid_history: Vec<f32>,
}

impl VoiceActivityDetector {
    pub fn new(config: VADConfig) -> Self {
        let mut state = VADState::default();
        state.energy_threshold = config.energy_threshold;

        // Calculate frame counts from duration
        let frames_per_second = config.sample_rate as f32 / config.frame_size as f32;
        state.min_speech_frames = ((config.voice_duration_ms as f32 / 1000.0) * frames_per_second) as u32;
        state.min_silence_frames = ((config.silence_duration_ms as f32 / 1000.0) * frames_per_second) as u32;

        let window_size = (frames_per_second * 2.0) as usize; // 2 second history

        debug!("VAD initialized - threshold: {:.6}, speech_frames: {}, silence_frames: {}, window_size: {}",
               state.energy_threshold, state.min_speech_frames, state.min_silence_frames, window_size);

        Self {
            config,
            state,
            window_size,
            energy_history: Vec::with_capacity(window_size),
            zcr_history: Vec::with_capacity(window_size),
            spectral_centroid_history: Vec::with_capacity(window_size),
        }
    }

    /// Process audio samples and detect voice activity
    pub fn process(&mut self, samples: &[f32]) -> Result<VADResult, AudioError> {
        if samples.is_empty() {
            return Ok(VADResult {
                has_voice: false,
                confidence: 0.0,
                energy_level: 0.0,
                spectral_centroid: 0.0,
                zero_crossing_rate: 0.0,
                state_changed: false,
            });
        }

        let energy = self.calculate_energy(samples);
        let zcr = self.calculate_zero_crossing_rate(samples);
        let spectral_centroid = self.calculate_spectral_centroid(samples);

        // Update history
        self.update_history(energy, zcr, spectral_centroid);

        // Get adaptive thresholds
        let adaptive_energy_threshold = self.calculate_adaptive_energy_threshold();
        let adaptive_zcr_threshold = self.calculate_adaptive_zcr_threshold();
        let adaptive_centroid_threshold = self.calculate_adaptive_centroid_threshold();

        // Enhanced VAD using energy, ZCR, and spectral centroid
        let energy_indicates_speech = energy > adaptive_energy_threshold;
        let zcr_indicates_speech = zcr > 0.02 && zcr < adaptive_zcr_threshold; // Voice has moderate ZCR
        let centroid_indicates_speech = spectral_centroid > adaptive_centroid_threshold;

        // Combine indicators with weights
        let speech_score = (
            if energy_indicates_speech { 0.5 } else { 0.0 } +
            if zcr_indicates_speech { 0.3 } else { 0.0 } +
            if centroid_indicates_speech { 0.2 } else { 0.0 }
        );

        let is_speech = speech_score > 0.4; // Require at least 40% confidence
        let state_changed = self.state.update(energy);

        let confidence = self.calculate_confidence(energy, zcr, spectral_centroid, adaptive_energy_threshold);

        Ok(VADResult {
            has_voice: self.state.is_speaking,
            confidence,
            energy_level: energy,
            spectral_centroid,
            zero_crossing_rate: zcr,
            state_changed,
        })
    }

    /// Calculate RMS energy
    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Calculate zero-crossing rate
    fn calculate_zero_crossing_rate(&self, samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }

        let mut crossings = 0;
        for i in 1..samples.len() {
            if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (samples.len() - 1) as f32
    }

    /// Calculate spectral centroid (approximated using zero-crossing rate)
    fn calculate_spectral_centroid(&self, samples: &[f32]) -> f32 {
        // Simple approximation: higher ZCR indicates higher frequency content
        let zcr = self.calculate_zero_crossing_rate(samples);
        // Convert to rough frequency estimate
        zcr * (self.config.sample_rate as f32 / 4.0)
    }

    /// Update energy, ZCR, and spectral centroid history
    fn update_history(&mut self, energy: f32, zcr: f32, spectral_centroid: f32) {
        if self.energy_history.len() >= self.window_size {
            self.energy_history.remove(0);
            self.zcr_history.remove(0);
            self.spectral_centroid_history.remove(0);
        }

        self.energy_history.push(energy);
        self.zcr_history.push(zcr);
        self.spectral_centroid_history.push(spectral_centroid);
    }

    /// Calculate adaptive energy threshold based on recent history
    fn calculate_adaptive_energy_threshold(&self) -> f32 {
        if self.energy_history.is_empty() {
            return self.state.energy_threshold;
        }

        let mean_energy: f32 = self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32;
        let variance: f32 = self.energy_history.iter()
            .map(|&x| (x - mean_energy).powi(2))
            .sum::<f32>() / self.energy_history.len() as f32;
        let std_dev = variance.sqrt();

        // Adaptive threshold: mean + 2 * std_dev
        (mean_energy + 2.0 * std_dev).max(self.state.energy_threshold)
    }

    /// Calculate adaptive ZCR threshold
    fn calculate_adaptive_zcr_threshold(&self) -> f32 {
        if self.zcr_history.is_empty() {
            return 0.3; // Default ZCR threshold for speech (voice has moderate ZCR)
        }

        let mean_zcr: f32 = self.zcr_history.iter().sum::<f32>() / self.zcr_history.len() as f32;
        (mean_zcr + 0.1).min(0.5) // Voice typically has ZCR between 0.1-0.5
    }

    /// Calculate adaptive spectral centroid threshold
    fn calculate_adaptive_centroid_threshold(&self) -> f32 {
        if self.spectral_centroid_history.is_empty() {
            return 800.0; // Default threshold for voice frequency range
        }

        let mean_centroid: f32 = self.spectral_centroid_history.iter().sum::<f32>() / self.spectral_centroid_history.len() as f32;
        mean_centroid.max(400.0) // Voice typically above 400Hz
    }


    /// Calculate confidence score
    fn calculate_confidence(&self, energy: f32, zcr: f32, spectral_centroid: f32, energy_threshold: f32) -> f32 {
        let energy_confidence = if energy > energy_threshold {
            ((energy / energy_threshold) - 1.0).min(1.0).max(0.0)
        } else {
            0.0
        };

        let zcr_confidence = if zcr > 0.02 && zcr < 0.5 {
            // Voice has moderate ZCR - closer to 0.1-0.3 is better
            let optimal_zcr = 0.2;
            1.0 - (zcr - optimal_zcr).abs() / optimal_zcr
        } else {
            0.0
        }.max(0.0);

        let centroid_confidence = if spectral_centroid > 300.0 && spectral_centroid < 4000.0 {
            // Voice frequency range
            let optimal_centroid = 1000.0;
            1.0 - (spectral_centroid - optimal_centroid).abs() / optimal_centroid
        } else {
            0.0
        }.max(0.0);

        // Weighted average
        (energy_confidence * 0.5 + zcr_confidence * 0.3 + centroid_confidence * 0.2).min(1.0)
    }

    /// Get current VAD configuration
    pub fn config(&self) -> &VADConfig {
        &self.config
    }

    /// Check if currently detecting voice
    pub fn is_voice_detected(&self) -> bool {
        self.state.is_speaking
    }

    /// Reset VAD state
    pub fn reset(&mut self) {
        self.state = VADState::default();
        self.state.energy_threshold = self.config.energy_threshold;
        self.energy_history.clear();
        self.zcr_history.clear();
        self.spectral_centroid_history.clear();
        debug!("VAD state reset");
    }

    /// Update VAD parameters
    pub fn update_parameters(&mut self, energy_threshold: f32, min_speech_frames: u32, min_silence_frames: u32) {
        self.state.energy_threshold = energy_threshold;
        self.state.min_speech_frames = min_speech_frames;
        self.state.min_silence_frames = min_silence_frames;
    }
}

/// VAD processing result
#[derive(Debug, Clone)]
pub struct VADResult {
    pub has_voice: bool,
    pub confidence: f32,
    pub energy_level: f32,
    pub spectral_centroid: f32,
    pub zero_crossing_rate: f32,
    pub state_changed: bool,
}

/// Legacy compatibility
pub type VadResult = VADResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_basic() {
        let config = VADConfig {
            sample_rate: 16000,
            frame_size: 1024,
            energy_threshold: 0.01,
            silence_duration_ms: 500,
            voice_duration_ms: 100,
        };
        let mut vad = VoiceActivityDetector::new(config);

        // Silent samples
        let silent_samples = vec![0.0; 1024];
        let result = vad.process(&silent_samples).unwrap();
        assert!(!result.has_voice);
        assert!(result.energy_level < 0.01);

        // Speech-like samples
        let speech_samples: Vec<f32> = (0..1024)
            .map(|i| 0.5 * (i as f32 * 0.1).sin())
            .collect();
        let result = vad.process(&speech_samples).unwrap();
        // Note: Might need several frames to detect speech
    }

    #[test]
    fn test_energy_calculation() {
        let config = VADConfig {
            sample_rate: 16000,
            frame_size: 1024,
            energy_threshold: 0.01,
            silence_duration_ms: 500,
            voice_duration_ms: 100,
        };
        let vad = VoiceActivityDetector::new(config);

        let samples = vec![0.1, -0.1, 0.2, -0.2];
        let energy = vad.calculate_energy(&samples);

        // RMS of [0.1, -0.1, 0.2, -0.2] = sqrt((0.01 + 0.01 + 0.04 + 0.04) / 4) = sqrt(0.025) ≈ 0.158
        assert!((energy - 0.158).abs() < 0.001);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let config = VADConfig {
            sample_rate: 16000,
            frame_size: 1024,
            energy_threshold: 0.01,
            silence_duration_ms: 500,
            voice_duration_ms: 100,
        };
        let vad = VoiceActivityDetector::new(config);

        // Alternating positive/negative samples
        let samples = vec![0.1, -0.1, 0.1, -0.1];
        let zcr = vad.calculate_zero_crossing_rate(&samples);

        // 3 zero crossings out of 3 intervals = 1.0
        assert!((zcr - 1.0).abs() < 0.001);
    }
}