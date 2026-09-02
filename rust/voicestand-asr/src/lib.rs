//! Production speech-recognition contracts for VoiceStand.
//!
//! This crate deliberately contains no inference engine. Backends such as
//! whisper.cpp and sherpa-onnx implement [`SpeechBackend`] behind their own
//! dependency boundaries, while orchestration depends only on these types.

use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use voicestand_types::{Result, VoiceStandError};

pub mod benchmark;
mod wav;
mod whisper_cpp;
mod zipformer;

pub use wav::read_wav_16khz_mono;
pub use whisper_cpp::WhisperCppBackend;
pub use zipformer::{SherpaOnlineSession, SherpaZipformerBackend, SherpaZipformerConfig};

#[derive(Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Stable identifier for a model and its local artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: String,
    pub path: PathBuf,
}

impl ModelSpec {
    pub fn new(id: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
        }
    }

    /// Validate properties common to every local inference backend.
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(VoiceStandError::model_load_failed(
                "model identifier must not be empty",
            ));
        }

        if !self.path.is_file() {
            return Err(VoiceStandError::model_load_failed(format!(
                "model file does not exist: {}",
                self.path.display()
            )));
        }

        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Decode controls supported by the production orchestration layer.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeOptions {
    pub language: Option<String>,
    pub thread_count: usize,
    pub translate: bool,
    pub emit_partials: bool,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            language: Some("en".to_string()),
            thread_count: 2,
            translate: false,
            emit_partials: false,
        }
    }
}

impl DecodeOptions {
    pub fn validate(&self) -> Result<()> {
        if self.thread_count == 0 {
            return Err(VoiceStandError::speech(
                "decode thread count must be at least one",
            ));
        }

        if self
            .language
            .as_ref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(VoiceStandError::speech(
                "decode language must be omitted or non-empty",
            ));
        }

        Ok(())
    }
}

/// A backend-neutral transcript returned to the session orchestrator.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub confidence: Option<f32>,
    pub audio_duration: Duration,
    pub is_final: bool,
}

impl Transcript {
    pub fn partial_result(
        text: impl Into<String>,
        confidence: Option<f32>,
        audio_duration: Duration,
    ) -> Result<Self> {
        let transcript = Self {
            text: text.into(),
            confidence,
            audio_duration,
            is_final: false,
        };
        transcript.validate()?;
        Ok(transcript)
    }

    pub fn final_result(
        text: impl Into<String>,
        confidence: Option<f32>,
        audio_duration: Duration,
    ) -> Result<Self> {
        let transcript = Self {
            text: text.into(),
            confidence,
            audio_duration,
            is_final: true,
        };
        transcript.validate()?;
        Ok(transcript)
    }

    pub fn validate(&self) -> Result<()> {
        if self.text.trim().is_empty() {
            return Err(VoiceStandError::speech(
                "backend returned an empty transcript",
            ));
        }

        if self
            .confidence
            .is_some_and(|confidence| !(0.0..=1.0).contains(&confidence))
        {
            return Err(VoiceStandError::speech(
                "transcript confidence must be between zero and one",
            ));
        }

        Ok(())
    }
}

/// Replaceable local speech-recognition backend.
///
/// Audio is canonical 16 kHz mono PCM represented as normalized `f32` samples.
/// Implementations should keep the loaded model resident between calls.
pub trait SpeechBackend: Send {
    /// Human-readable backend name used in diagnostics and benchmarks.
    fn name(&self) -> &'static str;

    /// Load or replace the active model.
    fn load(&mut self, model: &ModelSpec) -> Result<()>;

    /// Return whether a model is ready for transcription.
    fn is_loaded(&self) -> bool;

    /// Transcribe one bounded utterance in canonical PCM format.
    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<Transcript>;

    fn transcribe_cancellable(
        &mut self,
        audio: &[f32],
        options: &DecodeOptions,
        cancellation: CancellationToken,
    ) -> Result<Transcript> {
        if cancellation.is_cancelled() {
            return Err(VoiceStandError::speech("transcription cancelled"));
        }
        self.transcribe(audio, options)
    }
}

/// Validate inputs before crossing an inference-engine boundary.
pub fn validate_request(audio: &[f32], options: &DecodeOptions) -> Result<()> {
    options.validate()?;

    if audio.is_empty() {
        return Err(VoiceStandError::speech("audio input must not be empty"));
    }

    if audio.iter().any(|sample| !sample.is_finite()) {
        return Err(VoiceStandError::speech(
            "audio input contains a non-finite sample",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockBackend {
        loaded: bool,
    }

    impl SpeechBackend for MockBackend {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn load(&mut self, model: &ModelSpec) -> Result<()> {
            model.validate()?;
            self.loaded = true;
            Ok(())
        }

        fn is_loaded(&self) -> bool {
            self.loaded
        }

        fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<Transcript> {
            if !self.loaded {
                return Err(VoiceStandError::speech("backend model is not loaded"));
            }
            validate_request(audio, options)?;
            Transcript::final_result(
                "test transcript",
                Some(0.9),
                Duration::from_secs_f64(audio.len() as f64 / 16_000.0),
            )
        }
    }

    #[test]
    fn mock_backend_exercises_contract() {
        let model_path =
            std::env::temp_dir().join(format!("voicestand-asr-model-{}", std::process::id()));
        std::fs::write(&model_path, b"fixture").expect("create model fixture");

        let mut backend = MockBackend::default();
        backend
            .load(&ModelSpec::new("fixture", &model_path))
            .expect("load fixture");
        let transcript = backend
            .transcribe(&[0.0; 1_600], &DecodeOptions::default())
            .expect("transcribe fixture");

        assert!(backend.is_loaded());
        assert_eq!(backend.name(), "mock");
        assert_eq!(transcript.text, "test transcript");
        assert!(transcript.is_final);
        assert_eq!(transcript.audio_duration, Duration::from_millis(100));

        std::fs::remove_file(model_path).expect("remove model fixture");
    }

    #[test]
    fn request_validation_rejects_invalid_audio_and_options() {
        assert!(validate_request(&[], &DecodeOptions::default()).is_err());
        assert!(validate_request(&[f32::NAN], &DecodeOptions::default()).is_err());

        let invalid_options = DecodeOptions {
            thread_count: 0,
            ..DecodeOptions::default()
        };
        assert!(validate_request(&[0.0], &invalid_options).is_err());
    }

    #[test]
    fn transcript_validation_rejects_invalid_results() {
        assert!(Transcript::final_result("  ", None, Duration::ZERO).is_err());
        assert!(Transcript::final_result("text", Some(1.1), Duration::ZERO).is_err());
    }
}
