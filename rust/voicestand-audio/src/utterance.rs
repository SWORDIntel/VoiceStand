use std::collections::VecDeque;

/// Collects VAD-labelled audio frames into complete speech utterances.
///
/// A short rolling pre-roll keeps the beginning of words that precede the
/// VAD transition. The maximum duration bounds memory if speech never ends.
pub struct UtteranceAssembler {
    sample_rate: u32,
    pre_roll_samples: usize,
    max_samples: usize,
    pre_roll: VecDeque<f32>,
    active: Vec<f32>,
    in_speech: bool,
    last_partial_at: usize,
}

impl UtteranceAssembler {
    pub fn new(sample_rate: u32, pre_roll_ms: u32, max_duration_ms: u32) -> Self {
        Self {
            sample_rate,
            pre_roll_samples: samples_for_ms(sample_rate, pre_roll_ms),
            max_samples: samples_for_ms(sample_rate, max_duration_ms).max(1),
            pre_roll: VecDeque::new(),
            active: Vec::new(),
            in_speech: false,
            last_partial_at: 0,
        }
    }

    /// Push one mono frame. Returns an utterance when speech ends or reaches
    /// the configured maximum duration.
    pub fn push(&mut self, samples: &[f32], has_voice: bool) -> Option<Vec<f32>> {
        if !self.in_speech {
            if has_voice {
                self.in_speech = true;
                self.active.extend(self.pre_roll.drain(..));
                self.active.extend_from_slice(samples);
            } else {
                self.extend_pre_roll(samples);
                return None;
            }
        } else {
            self.active.extend_from_slice(samples);
            if !has_voice {
                return self.finish();
            }
        }

        if self.active.len() >= self.max_samples {
            return self.finish();
        }
        None
    }

    /// Start a push-to-talk utterance and preserve the rolling pre-roll.
    pub fn begin(&mut self) {
        if !self.in_speech {
            self.in_speech = true;
            self.active.extend(self.pre_roll.drain(..));
        }
    }

    /// Finalize the active push-to-talk utterance immediately on key release.
    pub fn end(&mut self) -> Option<Vec<f32>> {
        self.in_speech.then(|| self.finish()).flatten()
    }

    pub fn reset(&mut self) {
        self.pre_roll.clear();
        self.active.clear();
        self.in_speech = false;
        self.last_partial_at = 0;
    }

    pub fn partial_if_due(
        &mut self,
        minimum_samples: usize,
        interval_samples: usize,
    ) -> Option<Vec<f32>> {
        if self.in_speech
            && self.active.len() >= minimum_samples
            && self.active.len().saturating_sub(self.last_partial_at) >= interval_samples
        {
            self.last_partial_at = self.active.len();
            return Some(self.active.clone());
        }
        None
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn extend_pre_roll(&mut self, samples: &[f32]) {
        self.pre_roll.extend(samples.iter().copied());
        while self.pre_roll.len() > self.pre_roll_samples {
            self.pre_roll.pop_front();
        }
    }

    fn finish(&mut self) -> Option<Vec<f32>> {
        self.in_speech = false;
        self.last_partial_at = 0;
        self.pre_roll.clear();
        let utterance = std::mem::take(&mut self.active);
        (!utterance.is_empty()).then_some(utterance)
    }
}

fn samples_for_ms(sample_rate: u32, duration_ms: u32) -> usize {
    (u64::from(sample_rate) * u64::from(duration_ms) / 1_000) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn includes_pre_roll_and_final_silence_frame() {
        let mut assembler = UtteranceAssembler::new(1_000, 4, 1_000);
        assert!(assembler.push(&[1.0, 2.0], false).is_none());
        assert!(assembler.push(&[3.0, 4.0], false).is_none());
        assert!(assembler.push(&[5.0, 6.0], true).is_none());

        assert_eq!(
            assembler.push(&[0.0, 0.0], false),
            Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0])
        );
    }

    #[test]
    fn bounds_continuous_speech() {
        let mut assembler = UtteranceAssembler::new(1_000, 0, 5);
        assert!(assembler.push(&[1.0, 2.0, 3.0], true).is_none());
        assert_eq!(
            assembler.push(&[4.0, 5.0, 6.0], true),
            Some(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
    }

    #[test]
    fn rate_limits_partial_snapshots() {
        let mut assembler = UtteranceAssembler::new(1_000, 0, 10_000);
        assembler.push(&[1.0; 500], true);
        assert!(assembler.partial_if_due(1_000, 1_000).is_none());
        assembler.push(&[1.0; 500], true);
        assert_eq!(assembler.partial_if_due(1_000, 1_000).unwrap().len(), 1_000);
        assembler.push(&[1.0; 500], true);
        assert!(assembler.partial_if_due(1_000, 1_000).is_none());
    }

    #[test]
    fn push_to_talk_release_finalizes_without_waiting_for_vad() {
        let mut assembler = UtteranceAssembler::new(1_000, 2, 10_000);
        assembler.push(&[1.0, 2.0], false);
        assembler.begin();
        assert!(assembler.push(&[3.0, 4.0], true).is_none());
        assert_eq!(assembler.end(), Some(vec![1.0, 2.0, 3.0, 4.0]));
        assert!(assembler.end().is_none());
    }
}
