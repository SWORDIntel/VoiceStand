use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream};
use voicestand_types::{Result, VoiceStandError};

use crate::Transcript;

const SAMPLE_RATE: i32 = 16_000;
const FINAL_PADDING_SAMPLES: usize = 12_800;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SherpaZipformerConfig {
    pub model_dir: PathBuf,
    pub thread_count: usize,
    pub use_int8: bool,
}

impl SherpaZipformerConfig {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model_dir: model_dir.into(),
            thread_count: 2,
            use_int8: true,
        }
    }

    fn encoder(&self) -> PathBuf {
        self.model_dir.join(if self.use_int8 {
            "encoder-epoch-99-avg-1.int8.onnx"
        } else {
            "encoder-epoch-99-avg-1.onnx"
        })
    }

    // The upstream 20M model's documented int8 configuration keeps its small
    // decoder in fp32 and quantizes the encoder and joiner.
    fn decoder(&self) -> PathBuf {
        self.model_dir.join("decoder-epoch-99-avg-1.onnx")
    }

    fn joiner(&self) -> PathBuf {
        self.model_dir.join(if self.use_int8 {
            "joiner-epoch-99-avg-1.int8.onnx"
        } else {
            "joiner-epoch-99-avg-1.onnx"
        })
    }

    fn tokens(&self) -> PathBuf {
        self.model_dir.join("tokens.txt")
    }

    fn validate(&self) -> Result<()> {
        if self.thread_count == 0 {
            return Err(VoiceStandError::config(
                "Zipformer thread count must be at least one",
            ));
        }
        for path in [self.encoder(), self.decoder(), self.joiner(), self.tokens()] {
            if !path.is_file() {
                return Err(VoiceStandError::model_load_failed(format!(
                    "Zipformer model artifact is missing: {}",
                    path.display()
                )));
            }
            if path.to_str().is_none() {
                return Err(VoiceStandError::model_load_failed(format!(
                    "Zipformer model path is not valid UTF-8: {}",
                    path.display()
                )));
            }
        }
        Ok(())
    }
}

pub struct SherpaZipformerBackend {
    recognizer: Arc<OnlineRecognizer>,
}

impl SherpaZipformerBackend {
    pub fn load(config: &SherpaZipformerConfig) -> Result<Self> {
        config.validate()?;
        let mut recognizer_config = OnlineRecognizerConfig::default();
        recognizer_config.model_config.transducer.encoder = path_string(&config.encoder())?;
        recognizer_config.model_config.transducer.decoder = path_string(&config.decoder())?;
        recognizer_config.model_config.transducer.joiner = path_string(&config.joiner())?;
        recognizer_config.model_config.tokens = path_string(&config.tokens())?;
        recognizer_config.model_config.num_threads =
            config.thread_count.try_into().map_err(|_| {
                VoiceStandError::config("Zipformer thread count exceeds sherpa-onnx limits")
            })?;
        recognizer_config.model_config.provider = Some("cpu".to_string());
        recognizer_config.decoding_method = Some("greedy_search".to_string());
        recognizer_config.max_active_paths = 4;
        recognizer_config.enable_endpoint = true;
        recognizer_config.rule1_min_trailing_silence = 2.4;
        recognizer_config.rule2_min_trailing_silence = 1.2;
        recognizer_config.rule3_min_utterance_length = 20.0;

        let recognizer = OnlineRecognizer::create(&recognizer_config).ok_or_else(|| {
            VoiceStandError::model_load_failed("sherpa-onnx could not create the online recognizer")
        })?;
        Ok(Self {
            recognizer: Arc::new(recognizer),
        })
    }

    pub fn start_session(&self) -> SherpaOnlineSession {
        SherpaOnlineSession {
            stream: self.recognizer.create_stream(),
            recognizer: Arc::clone(&self.recognizer),
            samples_accepted: 0,
            committed_text: String::new(),
            active_text: String::new(),
            last_emitted: String::new(),
            active_segment: None,
            finished: false,
        }
    }
}

pub struct SherpaOnlineSession {
    // Drop the stream before its owning recognizer.
    stream: OnlineStream,
    recognizer: Arc<OnlineRecognizer>,
    samples_accepted: usize,
    committed_text: String,
    active_text: String,
    last_emitted: String,
    active_segment: Option<i32>,
    finished: bool,
}

impl SherpaOnlineSession {
    pub fn accept_audio(&mut self, samples: &[f32]) -> Result<Option<Transcript>> {
        if self.finished {
            return Err(VoiceStandError::speech(
                "cannot append audio to a finished Zipformer session",
            ));
        }
        validate_samples(samples)?;
        self.stream.accept_waveform(SAMPLE_RATE, samples);
        self.samples_accepted += samples.len();
        self.decode_ready();
        self.changed_result(false)
    }

    pub fn finish(&mut self) -> Result<Option<Transcript>> {
        if self.finished {
            return Err(VoiceStandError::speech(
                "Zipformer session has already been finalized",
            ));
        }
        self.finished = true;
        self.stream
            .accept_waveform(SAMPLE_RATE, &[0.0; FINAL_PADDING_SAMPLES]);
        self.stream.input_finished();
        self.decode_ready();
        self.changed_result(true)
    }

    fn decode_ready(&mut self) {
        while self.recognizer.is_ready(&self.stream) {
            self.recognizer.decode(&self.stream);
            if self.recognizer.is_endpoint(&self.stream) {
                if let Some(result) = self.recognizer.get_result(&self.stream) {
                    append_words(&mut self.committed_text, &result.text);
                }
                self.active_text.clear();
                self.active_segment = None;
                self.recognizer.reset(&self.stream);
            }
        }
    }

    fn changed_result(&mut self, is_final: bool) -> Result<Option<Transcript>> {
        if let Some(result) = self.recognizer.get_result(&self.stream) {
            let segment = result.segment.unwrap_or(0);
            if self.active_segment.is_some_and(|active| active != segment)
                && !self.active_text.is_empty()
            {
                append_words(&mut self.committed_text, &self.active_text);
                self.active_text.clear();
            }
            self.active_segment = Some(segment);
            self.active_text.clear();
            self.active_text.push_str(result.text.trim());
        }

        let mut text = self.committed_text.clone();
        append_words(&mut text, &self.active_text);
        if text.is_empty() || (!is_final && text == self.last_emitted) {
            return Ok(None);
        }
        self.last_emitted.clone_from(&text);
        let duration = Duration::from_secs_f64(self.samples_accepted as f64 / SAMPLE_RATE as f64);
        if is_final {
            Transcript::final_result(&text, None, duration).map(Some)
        } else {
            Transcript::partial_result(&text, None, duration).map(Some)
        }
    }
}

fn append_words(target: &mut String, words: &str) {
    let words = words.trim();
    if words.is_empty() {
        return;
    }
    if !target.is_empty() {
        target.push(' ');
    }
    target.push_str(words);
}

fn path_string(path: &Path) -> Result<Option<String>> {
    path.to_str()
        .map(|path| Some(path.to_string()))
        .ok_or_else(|| VoiceStandError::model_load_failed("Zipformer model path is not UTF-8"))
}

fn validate_samples(samples: &[f32]) -> Result<()> {
    if samples.is_empty() {
        return Err(VoiceStandError::speech(
            "streaming audio chunk must not be empty",
        ));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(VoiceStandError::speech(
            "streaming audio contains a non-finite sample",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_streaming_audio() {
        assert!(validate_samples(&[]).is_err());
        assert!(validate_samples(&[f32::NAN]).is_err());
    }

    #[test]
    fn joins_finalized_segments_with_one_separator() {
        let mut text = "first segment".to_string();
        append_words(&mut text, " second segment ");
        assert_eq!(text, "first segment second segment");
    }

    #[test]
    fn rejects_a_missing_model_directory() {
        let error = SherpaZipformerBackend::load(&SherpaZipformerConfig::new(
            "/definitely/missing/voicestand-model",
        ))
        .err()
        .expect("missing model should fail");
        assert!(error.to_string().contains("artifact is missing"));
    }
}
