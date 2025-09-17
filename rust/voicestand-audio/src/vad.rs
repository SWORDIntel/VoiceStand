use voicestand_core::{VadState, Result};

/// Voice Activity Detection implementation
pub struct VoiceActivityDetector {
    state: VadState,
    window_size: usize,
    energy_history: Vec<f32>,
    zcr_history: Vec<f32>,
}

impl VoiceActivityDetector {
    pub fn new(energy_threshold: f32, window_size: usize) -> Self {
        let mut state = VadState::default();
        state.energy_threshold = energy_threshold;

        Self {
            state,
            window_size,
            energy_history: Vec::with_capacity(window_size),
            zcr_history: Vec::with_capacity(window_size),
        }
    }

    /// Process audio samples and detect voice activity
    pub fn process(&mut self, samples: &[f32]) -> VadResult {
        let energy = self.calculate_energy(samples);
        let zcr = self.calculate_zero_crossing_rate(samples);

        // Update history
        self.update_history(energy, zcr);

        // Get adaptive thresholds
        let adaptive_energy_threshold = self.calculate_adaptive_energy_threshold();
        let adaptive_zcr_threshold = self.calculate_adaptive_zcr_threshold();

        // Enhanced VAD using both energy and ZCR
        let is_speech = energy > adaptive_energy_threshold && zcr < adaptive_zcr_threshold;
        let state_changed = self.update_state(is_speech);

        VadResult {
            is_speech: self.state.is_speaking,
            energy,
            zcr,
            state_changed,
            confidence: self.calculate_confidence(energy, zcr),
        }
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

    /// Update energy and ZCR history
    fn update_history(&mut self, energy: f32, zcr: f32) {
        if self.energy_history.len() >= self.window_size {
            self.energy_history.remove(0);
            self.zcr_history.remove(0);
        }

        self.energy_history.push(energy);
        self.zcr_history.push(zcr);
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
            return 0.1; // Default ZCR threshold
        }

        let mean_zcr: f32 = self.zcr_history.iter().sum::<f32>() / self.zcr_history.len() as f32;
        mean_zcr + 0.05 // Slightly above average
    }

    /// Update VAD state and return if state changed
    fn update_state(&mut self, is_speech: bool) -> bool {
        self.state.update(if is_speech { self.state.energy_threshold * 2.0 } else { 0.0 })
    }

    /// Calculate confidence score
    fn calculate_confidence(&self, energy: f32, zcr: f32) -> f32 {
        let energy_confidence = if energy > self.state.energy_threshold {
            (energy / self.state.energy_threshold).min(1.0)
        } else {
            0.0
        };

        let zcr_confidence = if zcr < 0.1 {
            1.0 - zcr / 0.1
        } else {
            0.0
        };

        (energy_confidence + zcr_confidence) / 2.0
    }

    /// Get current VAD state
    pub fn state(&self) -> &VadState {
        &self.state
    }

    /// Reset VAD state
    pub fn reset(&mut self) {
        self.state = VadState::default();
        self.energy_history.clear();
        self.zcr_history.clear();
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
pub struct VadResult {
    pub is_speech: bool,
    pub energy: f32,
    pub zcr: f32,
    pub state_changed: bool,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_basic() {
        let mut vad = VoiceActivityDetector::new(0.01, 10);

        // Silent samples
        let silent_samples = vec![0.0; 1024];
        let result = vad.process(&silent_samples);
        assert!(!result.is_speech);
        assert!(result.energy < 0.01);

        // Speech-like samples
        let speech_samples: Vec<f32> = (0..1024)
            .map(|i| 0.5 * (i as f32 * 0.1).sin())
            .collect();
        let result = vad.process(&speech_samples);
        // Note: Might need several frames to detect speech
    }

    #[test]
    fn test_energy_calculation() {
        let vad = VoiceActivityDetector::new(0.01, 10);

        let samples = vec![0.1, -0.1, 0.2, -0.2];
        let energy = vad.calculate_energy(&samples);

        // RMS of [0.1, -0.1, 0.2, -0.2] = sqrt((0.01 + 0.01 + 0.04 + 0.04) / 4) = sqrt(0.025) ≈ 0.158
        assert!((energy - 0.158).abs() < 0.001);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let vad = VoiceActivityDetector::new(0.01, 10);

        // Alternating positive/negative samples
        let samples = vec![0.1, -0.1, 0.1, -0.1];
        let zcr = vad.calculate_zero_crossing_rate(&samples);

        // 3 zero crossings out of 3 intervals = 1.0
        assert!((zcr - 1.0).abs() < 0.001);
    }
}