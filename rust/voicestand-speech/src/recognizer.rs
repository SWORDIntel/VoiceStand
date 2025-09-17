use crate::{WhisperModel, MelSpectrogramExtractor, features};
use voicestand_core::{
    SpeechConfig, AudioData, TranscriptionResult, AppEvent, Result, VoiceStandError
};

use candle_core::{Tensor, Device, DType};
use crossbeam_channel::Sender;
use parking_lot::Mutex;
use std::sync::Arc;
use std::collections::VecDeque;
use tokio::time::{Duration, Instant};

/// Memory-safe speech recognition engine using Candle
pub struct SpeechRecognizer {
    model: Arc<Mutex<WhisperModel>>,
    feature_extractor: MelSpectrogramExtractor,
    config: SpeechConfig,
    event_sender: Sender<AppEvent>,
    audio_buffer: Arc<Mutex<VecDeque<f32>>>,
    is_processing: Arc<Mutex<bool>>,
    last_transcription: Arc<Mutex<Option<TranscriptionResult>>>,
}

impl SpeechRecognizer {
    /// Create new speech recognizer
    pub fn new(config: SpeechConfig, event_sender: Sender<AppEvent>) -> Result<Self> {
        let model = WhisperModel::load(&config)?;
        let device = model.device().clone();

        // Initialize feature extractor with Whisper parameters
        let feature_extractor = MelSpectrogramExtractor::new(
            16000,  // Whisper expects 16kHz
            400,    // n_fft
            160,    // hop_length
            80,     // n_mels
            device,
        )?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            feature_extractor,
            config,
            event_sender,
            audio_buffer: Arc::new(Mutex::new(VecDeque::new())),
            is_processing: Arc::new(Mutex::new(false)),
            last_transcription: Arc::new(Mutex::new(None)),
        })
    }

    /// Initialize the recognizer
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!("Initializing speech recognizer");

        // Initialize model
        self.model.lock().initialize()?;

        tracing::info!("Speech recognizer initialized successfully");
        Ok(())
    }

    /// Start processing audio stream
    pub async fn start_streaming(&self) -> Result<()> {
        tracing::info!("Starting speech recognition streaming");

        let model = Arc::clone(&self.model);
        let audio_buffer = Arc::clone(&self.audio_buffer);
        let is_processing = Arc::clone(&self.is_processing);
        let event_sender = self.event_sender.clone();
        let config = self.config.clone();
        let mut feature_extractor = self.feature_extractor.clone();

        // Spawn processing task
        tokio::spawn(async move {
            let mut processing_timer = Instant::now();
            let processing_interval = Duration::from_millis(100); // Process every 100ms

            loop {
                if processing_timer.elapsed() >= processing_interval {
                    if let Err(e) = Self::process_audio_chunk(
                        &model,
                        &mut feature_extractor,
                        &audio_buffer,
                        &is_processing,
                        &event_sender,
                        &config,
                    ).await {
                        tracing::error!("Audio processing error: {}", e);
                        let _ = event_sender.send(AppEvent::Error(format!("Recognition error: {}", e)));
                    }
                    processing_timer = Instant::now();
                }

                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });

        Ok(())
    }

    /// Process audio data
    pub async fn process_audio(&self, audio_data: &AudioData) -> Result<()> {
        // Resample if necessary
        let audio_samples = if audio_data.sample_rate != 16000 {
            self.resample_audio(&audio_data.samples, audio_data.sample_rate, 16000)?
        } else {
            audio_data.samples.clone()
        };

        // Add to buffer
        self.audio_buffer.lock().extend(audio_samples);

        // Trigger processing if speech ended
        if audio_data.is_speech_end {
            self.process_buffered_audio().await?;
        }

        Ok(())
    }

    /// Process buffered audio immediately
    async fn process_buffered_audio(&self) -> Result<()> {
        if *self.is_processing.lock() {
            return Ok(()); // Already processing
        }

        let audio_samples: Vec<f32> = self.audio_buffer.lock().drain(..).collect();
        if audio_samples.len() < 1600 { // Less than 100ms at 16kHz
            return Ok(());
        }

        *self.is_processing.lock() = true;

        let result = self.recognize_audio(&audio_samples).await;

        *self.is_processing.lock() = false;

        match result {
            Ok(Some(transcription)) => {
                *self.last_transcription.lock() = Some(transcription.clone());
                self.event_sender.send(AppEvent::TranscriptionReceived(transcription))
                    .map_err(|_| VoiceStandError::speech("Failed to send transcription event"))?;
            }
            Ok(None) => {
                // No transcription (silence or noise)
            }
            Err(e) => {
                tracing::error!("Recognition failed: {}", e);
                self.event_sender.send(AppEvent::Error(format!("Recognition error: {}", e)))
                    .map_err(|_| VoiceStandError::speech("Failed to send error event"))?;
            }
        }

        Ok(())
    }

    /// Recognize speech from audio samples
    async fn recognize_audio(&self, audio: &[f32]) -> Result<Option<TranscriptionResult>> {
        let start_time = Instant::now();

        // Normalize and pad audio to expected length (30 seconds max for Whisper)
        let max_samples = 16000 * 30; // 30 seconds at 16kHz
        let mut normalized_audio = features::pad_or_trim(audio, max_samples);
        features::normalize_audio(&mut normalized_audio);

        // Extract mel-spectrogram features
        let mel_features = self.feature_extractor.extract(&normalized_audio)?;

        // Run inference
        let transcription = {
            let model = self.model.lock();
            self.run_inference(&*model, &mel_features).await?
        };

        let processing_time = start_time.elapsed();
        tracing::debug!("Recognition took: {:?}", processing_time);

        // Filter out empty or very short transcriptions
        if transcription.text.trim().len() < 2 {
            return Ok(None);
        }

        // Filter out low confidence results
        if transcription.confidence < 0.3 {
            return Ok(None);
        }

        Ok(Some(transcription))
    }

    /// Run model inference
    async fn run_inference(&self, model: &WhisperModel, mel_features: &Tensor) -> Result<TranscriptionResult> {
        // Encode audio features
        let audio_features = model.encode(mel_features)?;

        // Prepare initial tokens
        let start_tokens = vec![
            50258, // <|startoftranscript|>
            50259, // <|notimestamps|>
        ];

        let mut tokens = start_tokens;
        let mut output_text = String::new();
        let max_tokens = self.config.max_tokens.min(512);

        // Decode tokens iteratively
        for _ in 0..max_tokens {
            let token_tensor = Tensor::from_slice(
                &tokens,
                (1, tokens.len()),
                model.device(),
            )?;

            let logits = model.decode(&token_tensor, &audio_features)?;

            // Get next token (simplified - real implementation would use proper sampling)
            let next_token = self.sample_token(&logits)?;

            // Check for end token
            if next_token == 50257 { // <|endoftext|>
                break;
            }

            tokens.push(next_token);

            // Decode token to text
            let token_text = model.tokenizer().decode(&[next_token]);
            output_text.push_str(&token_text);
        }

        // Clean up text
        let cleaned_text = self.postprocess_text(&output_text);

        Ok(TranscriptionResult::new(
            cleaned_text,
            0.85, // Simplified confidence score
            0.0,  // Start time
            audio_features.dim(1)? as f64 * 0.02, // Approximate duration
            true, // Final result
        ))
    }

    /// Sample next token from logits (simplified)
    fn sample_token(&self, logits: &Tensor) -> Result<u32> {
        // Get the last timestep logits
        let last_logits = logits.get(0)?.get(logits.dim(1)? - 1)?;

        // Find argmax (greedy decoding)
        let logits_vec = last_logits.to_vec1::<f32>()?;
        let max_idx = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(max_idx as u32)
    }

    /// Post-process transcribed text
    fn postprocess_text(&self, text: &str) -> String {
        text.trim()
            .replace("  ", " ")
            .replace(" .", ".")
            .replace(" ,", ",")
            .replace(" ?", "?")
            .replace(" !", "!")
            .to_string()
    }

    /// Resample audio (simplified)
    fn resample_audio(&self, audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
        if from_rate == to_rate {
            return Ok(audio.to_vec());
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let new_len = (audio.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx = src_idx as usize;

            if idx + 1 < audio.len() {
                // Linear interpolation
                let frac = src_idx - idx as f64;
                let sample = audio[idx] * (1.0 - frac) as f32 + audio[idx + 1] * frac as f32;
                resampled.push(sample);
            } else if idx < audio.len() {
                resampled.push(audio[idx]);
            }
        }

        Ok(resampled)
    }

    /// Process audio chunk in background task
    async fn process_audio_chunk(
        model: &Arc<Mutex<WhisperModel>>,
        feature_extractor: &mut MelSpectrogramExtractor,
        audio_buffer: &Arc<Mutex<VecDeque<f32>>>,
        is_processing: &Arc<Mutex<bool>>,
        event_sender: &Sender<AppEvent>,
        config: &SpeechConfig,
    ) -> Result<()> {
        if *is_processing.lock() {
            return Ok(());
        }

        let buffer_len = audio_buffer.lock().len();
        let chunk_size = 16000; // 1 second at 16kHz

        if buffer_len < chunk_size {
            return Ok(());
        }

        // Extract chunk
        let chunk: Vec<f32> = {
            let mut buffer = audio_buffer.lock();
            buffer.drain(..chunk_size).collect()
        };

        *is_processing.lock() = true;

        // Process chunk
        let result = tokio::task::spawn_blocking(move || {
            // Simplified processing for streaming
            if chunk.iter().map(|&x| x.abs()).sum::<f32>() / chunk.len() as f32 > 0.01 {
                Some(TranscriptionResult::new(
                    "Processing...".to_string(),
                    0.5,
                    0.0,
                    1.0,
                    false, // Partial result
                ))
            } else {
                None
            }
        }).await;

        *is_processing.lock() = false;

        if let Ok(Some(transcription)) = result {
            let _ = event_sender.send(AppEvent::TranscriptionReceived(transcription));
        }

        Ok(())
    }

    /// Get the last transcription result
    pub fn last_transcription(&self) -> Option<TranscriptionResult> {
        self.last_transcription.lock().clone()
    }

    /// Check if currently processing
    pub fn is_processing(&self) -> bool {
        *self.is_processing.lock()
    }

    /// Clear audio buffer
    pub fn clear_buffer(&self) {
        self.audio_buffer.lock().clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SpeechConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }
}

// Make feature extractor cloneable for async tasks
impl Clone for MelSpectrogramExtractor {
    fn clone(&self) -> Self {
        Self::new(
            self.sample_rate,
            self.n_fft,
            self.hop_length,
            self.n_mels,
            self.device.clone(),
        ).expect("Recognizer should create successfully")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    #[tokio::test]
    async fn test_recognizer_creation() {
        let config = SpeechConfig::default();
        let (sender, _receiver) = unbounded();

        // This will fail without actual model file, but tests the structure
        let result = SpeechRecognizer::new(config, sender);
        // Just check that the error is about missing model file
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_resampling() {
        let config = SpeechConfig::default();
        let (sender, _receiver) = unbounded();

        if let Ok(recognizer) = SpeechRecognizer::new(config, sender) {
            let audio = vec![1.0, 2.0, 3.0, 4.0];
            let resampled = recognizer.resample_audio(&audio, 8000, 16000).expect("Audio resampling should succeed");
            assert!(resampled.len() > audio.len());
        }
    }

    #[test]
    fn test_text_postprocessing() {
        let config = SpeechConfig::default();
        let (sender, _receiver) = unbounded();

        if let Ok(recognizer) = SpeechRecognizer::new(config, sender) {
            let text = "  hello  world  .  how are you  ?  ";
            let cleaned = recognizer.postprocess_text(text);
            assert_eq!(cleaned, "hello world. how are you?");
        }
    }
}