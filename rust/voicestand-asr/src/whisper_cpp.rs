use std::sync::Once;
use std::time::Duration;

use voicestand_types::{Result, VoiceStandError};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{
    validate_request, CancellationToken, DecodeOptions, ModelSpec, SpeechBackend, Transcript,
};

static INSTALL_LOGGING_HOOKS: Once = Once::new();

/// CPU-first production backend backed by whisper.cpp through `whisper-rs`.
pub struct WhisperCppBackend {
    context: Option<WhisperContext>,
    use_gpu: bool,
}

impl WhisperCppBackend {
    /// Create the normal production backend with CPU inference.
    pub fn cpu() -> Self {
        Self {
            context: None,
            use_gpu: false,
        }
    }

    /// Create a backend that may use a compiled-in GPU implementation.
    ///
    /// CPU remains the default and callers must opt into acceleration.
    pub fn with_gpu(use_gpu: bool) -> Self {
        Self {
            context: None,
            use_gpu,
        }
    }
}

impl Default for WhisperCppBackend {
    fn default() -> Self {
        Self::cpu()
    }
}

impl SpeechBackend for WhisperCppBackend {
    fn name(&self) -> &'static str {
        "whisper.cpp"
    }

    fn load(&mut self, model: &ModelSpec) -> Result<()> {
        model.validate()?;
        INSTALL_LOGGING_HOOKS.call_once(whisper_rs::install_logging_hooks);

        let model_path = model
            .path()
            .to_str()
            .ok_or_else(|| VoiceStandError::model_load_failed("model path is not valid UTF-8"))?;
        let mut parameters = WhisperContextParameters::default();
        parameters.use_gpu(self.use_gpu);

        let context = WhisperContext::new_with_params(model_path, parameters).map_err(|error| {
            VoiceStandError::model_load_failed(format!(
                "whisper.cpp could not load {}: {error}",
                model.path().display()
            ))
        })?;

        self.context = Some(context);
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.context.is_some()
    }

    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<Transcript> {
        self.transcribe_cancellable(audio, options, CancellationToken::default())
    }

    fn transcribe_cancellable(
        &mut self,
        audio: &[f32],
        options: &DecodeOptions,
        cancellation: CancellationToken,
    ) -> Result<Transcript> {
        validate_request(audio, options)?;
        let context = self
            .context
            .as_ref()
            .ok_or_else(|| VoiceStandError::speech("whisper.cpp model is not loaded"))?;
        let mut state = context.create_state().map_err(|error| {
            VoiceStandError::speech(format!("whisper.cpp could not create state: {error}"))
        })?;

        let mut parameters = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        parameters.set_n_threads(options.thread_count.try_into().map_err(|_| {
            VoiceStandError::speech("decode thread count exceeds whisper.cpp limits")
        })?);
        parameters.set_translate(options.translate);
        parameters.set_language(options.language.as_deref());
        parameters.set_print_progress(false);
        parameters.set_print_realtime(false);
        parameters.set_print_special(false);
        parameters.set_print_timestamps(false);
        let abort = cancellation.clone();
        parameters.set_abort_callback_safe(move || abort.is_cancelled());

        state.full(parameters, audio).map_err(|error| {
            if cancellation.is_cancelled() {
                return VoiceStandError::speech("transcription cancelled");
            }
            VoiceStandError::speech(format!("whisper.cpp transcription failed: {error}"))
        })?;

        let mut text = String::new();
        let mut probability_sum = 0.0_f32;
        let mut probability_count = 0_u32;

        for segment in state.as_iter() {
            let segment_text = segment.to_str_lossy().map_err(|error| {
                VoiceStandError::speech(format!("invalid whisper.cpp segment: {error}"))
            })?;
            text.push_str(&segment_text);

            for token_index in 0..segment.n_tokens() {
                if let Some(token) = segment.get_token(token_index) {
                    let probability = token.token_probability();
                    if probability.is_finite() && (0.0..=1.0).contains(&probability) {
                        probability_sum += probability;
                        probability_count += 1;
                    }
                }
            }
        }

        let confidence =
            (probability_count > 0).then_some(probability_sum / probability_count as f32);
        Transcript::final_result(
            text.trim(),
            confidence,
            Duration::from_secs_f64(audio.len() as f64 / 16_000.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_the_default_backend_mode() {
        let backend = WhisperCppBackend::default();
        assert_eq!(backend.name(), "whisper.cpp");
        assert!(!backend.is_loaded());
    }

    #[test]
    fn transcription_requires_a_loaded_model() {
        let mut backend = WhisperCppBackend::cpu();
        let error = backend
            .transcribe(&[0.0; 1_600], &DecodeOptions::default())
            .expect_err("unloaded backend should fail");
        assert!(error.to_string().contains("not loaded"));
    }
}
