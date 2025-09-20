// Voice activation detection
use anyhow::Result;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum ActivationEvent {
    WakeWordDetected,
    VoiceActivityStarted,
    VoiceActivityEnded,
    SilenceDetected,
}

#[derive(Debug)]
pub struct ActivationDetector {
    threshold: f32,
    min_duration: Duration,
    last_activity: Option<Instant>,
}

impl ActivationDetector {
    pub fn new(threshold: f32, min_duration: Duration) -> Self {
        Self {
            threshold,
            min_duration,
            last_activity: None,
        }
    }

    pub fn process_audio(&mut self, audio_data: &[f32]) -> Result<Option<ActivationEvent>> {
        if audio_data.is_empty() {
            return Ok(None);
        }

        // Calculate RMS energy
        let energy = self.calculate_energy(audio_data);
        let now = Instant::now();

        // Check if energy exceeds threshold
        if energy > self.threshold {
            // Voice activity detected
            let event = if self.last_activity.is_none() {
                // First detection - start of voice activity
                self.last_activity = Some(now);
                Some(ActivationEvent::VoiceActivityStarted)
            } else {
                // Continuing voice activity - update timestamp
                self.last_activity = Some(now);
                None // Don't send repeated start events
            };

            Ok(event)
        } else {
            // Low energy - check for end of voice activity
            if let Some(last_active) = self.last_activity {
                if now.duration_since(last_active) >= self.min_duration {
                    // Voice activity has ended
                    self.last_activity = None;
                    Ok(Some(ActivationEvent::VoiceActivityEnded))
                } else {
                    // Still within minimum duration - silence
                    Ok(Some(ActivationEvent::SilenceDetected))
                }
            } else {
                // No recent activity
                Ok(Some(ActivationEvent::SilenceDetected))
            }
        }
    }

    /// Calculate RMS energy of audio samples
    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }
}