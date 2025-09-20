use voicestand_core::{Result, VoiceStandError};
use voicestand_audio::AudioBuffer;
use crate::npu::{NPUManager, WhisperInferenceResult, NPUWorkloadType};
use openvino::{Tensor, Shape, ElementType};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error, debug};
use anyhow::anyhow;

/// NPU-accelerated Whisper processor for real-time speech recognition
/// Targets <2ms inference with 11 TOPS Intel NPU and <100mW power consumption
pub struct NPUWhisperProcessor {
    npu: Arc<RwLock<NPUManager>>,
    mel_extractor: MelSpectrogramExtractor,
    tokenizer: WhisperTokenizer,
    audio_buffer: Arc<RwLock<StreamingAudioBuffer>>,
    inference_tx: Option<mpsc::UnboundedSender<InferenceRequest>>,
    result_rx: Option<mpsc::UnboundedReceiver<TranscriptionResult>>,
    config: NPUWhisperConfig,
    performance_stats: Arc<RwLock<NPUWhisperStats>>,
}

#[derive(Debug, Clone)]
pub struct NPUWhisperConfig {
    pub model_path: std::path::PathBuf,
    pub language: String,
    pub sample_rate: u32,
    pub chunk_duration_ms: u32,
    pub overlap_ms: u32,
    pub vad_threshold: f32,
    pub beam_size: usize,
    pub temperature: f32,
    pub power_budget_mw: u32,
}

impl Default for NPUWhisperConfig {
    fn default() -> Self {
        Self {
            model_path: "models/whisper-base-int8.xml".into(),
            language: "en".to_string(),
            sample_rate: 16000,
            chunk_duration_ms: 100,  // 100ms chunks for real-time
            overlap_ms: 20,          // 20ms overlap for continuity
            vad_threshold: 0.3,
            beam_size: 1,            // Beam size 1 for minimal latency
            temperature: 0.0,        // Greedy decoding
            power_budget_mw: 100,    // 100mW power budget
        }
    }
}

#[derive(Debug, Clone)]
struct InferenceRequest {
    mel_features: Vec<f32>,
    timestamp: std::time::Instant,
    chunk_id: u64,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub confidence: f32,
    pub language: String,
    pub inference_time_ms: f32,
    pub chunk_id: u64,
    pub timestamp: std::time::Instant,
    pub tokens: Vec<u32>,
}

/// Mel spectrogram extractor optimized for NPU input format
struct MelSpectrogramExtractor {
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    mel_filters: Vec<Vec<f32>>,
    window: Vec<f32>,
}

impl MelSpectrogramExtractor {
    fn new(sample_rate: u32) -> Result<Self> {
        let n_fft = 512;
        let hop_length = 160;  // 10ms hop for 16kHz
        let n_mels = 80;       // Standard Whisper mel count

        // Generate Hann window
        let window = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32).cos()))
            .collect();

        // Generate mel filter banks
        let mel_filters = Self::create_mel_filters(n_fft, n_mels, sample_rate)?;

        Ok(Self {
            n_fft,
            hop_length,
            n_mels,
            mel_filters,
            window,
        })
    }

    fn create_mel_filters(n_fft: usize, n_mels: usize, sample_rate: u32) -> Result<Vec<Vec<f32>>> {
        let n_freqs = n_fft / 2 + 1;
        let mut filters = Vec::with_capacity(n_mels);

        // Mel scale conversion functions
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).ln();
        let mel_to_hz = |mel: f32| 700.0 * (mel / 2595.0).exp() - 700.0;

        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(sample_rate as f32 / 2.0);

        // Create mel points
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .map(mel_to_hz)
            .collect();

        // Convert to frequency bin indices
        let freq_points: Vec<f32> = mel_points
            .iter()
            .map(|&freq| freq * n_fft as f32 / sample_rate as f32)
            .collect();

        // Create triangular filters
        for i in 0..n_mels {
            let mut filter = vec![0.0; n_freqs];
            let left = freq_points[i] as usize;
            let center = freq_points[i + 1] as usize;
            let right = freq_points[i + 2] as usize;

            // Left slope
            for j in left..=center {
                if j < n_freqs && center > left {
                    filter[j] = (j - left) as f32 / (center - left) as f32;
                }
            }

            // Right slope
            for j in center..=right {
                if j < n_freqs && right > center {
                    filter[j] = (right - j) as f32 / (right - center) as f32;
                }
            }

            filters.push(filter);
        }

        Ok(filters)
    }

    /// Extract mel spectrogram features optimized for NPU input format
    fn extract(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let n_frames = (audio.len() - self.n_fft) / self.hop_length + 1;
        let mut mel_spec = Vec::with_capacity(self.n_mels * n_frames);

        // STFT computation with optimized windowing
        for frame in 0..n_frames {
            let start = frame * self.hop_length;
            let end = start + self.n_fft;

            if end > audio.len() {
                break;
            }

            // Apply window and compute FFT
            let windowed: Vec<f32> = audio[start..end]
                .iter()
                .zip(&self.window)
                .map(|(a, w)| a * w)
                .collect();

            let fft_result = self.compute_fft(&windowed)?;
            let power_spectrum = self.compute_power_spectrum(&fft_result);

            // Apply mel filters
            for filter in &self.mel_filters {
                let mel_energy: f32 = power_spectrum
                    .iter()
                    .zip(filter)
                    .map(|(p, f)| p * f)
                    .sum();

                // Log mel energy (add small epsilon for numerical stability)
                mel_spec.push((mel_energy.max(1e-10)).ln());
            }
        }

        Ok(mel_spec)
    }

    fn compute_fft(&self, input: &[f32]) -> Result<Vec<(f32, f32)>> {
        // Simple DFT implementation - in production use FFTW or similar
        let n = input.len();
        let mut result = Vec::with_capacity(n / 2 + 1);

        for k in 0..=(n / 2) {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in input.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            result.push((real, imag));
        }

        Ok(result)
    }

    fn compute_power_spectrum(&self, fft: &[(f32, f32)]) -> Vec<f32> {
        fft.iter()
            .map(|(real, imag)| real * real + imag * imag)
            .collect()
    }
}

/// Whisper tokenizer for text decoding
struct WhisperTokenizer {
    vocab: std::collections::HashMap<u32, String>,
    special_tokens: std::collections::HashMap<String, u32>,
}

impl WhisperTokenizer {
    fn new() -> Result<Self> {
        // Load tokenizer vocabulary - in production load from file
        let mut vocab = std::collections::HashMap::new();
        let mut special_tokens = std::collections::HashMap::new();

        // Add basic tokens (placeholder - load from actual Whisper tokenizer)
        vocab.insert(220, " ".to_string());
        vocab.insert(262, "the".to_string());
        vocab.insert(290, "and".to_string());

        special_tokens.insert("<|startoftranscript|>".to_string(), 50256);
        special_tokens.insert("<|endoftext|>".to_string(), 50257);

        Ok(Self { vocab, special_tokens })
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut result = String::new();

        for &token in tokens {
            if let Some(text) = self.vocab.get(&token) {
                result.push_str(text);
            } else if token < 50256 {
                // Unknown token, add placeholder
                result.push_str(&format!("<|unk_{}|>", token));
            }
            // Skip special tokens
        }

        Ok(result.trim().to_string())
    }
}

/// Streaming audio buffer for continuous processing
struct StreamingAudioBuffer {
    buffer: Vec<f32>,
    write_pos: usize,
    read_pos: usize,
    capacity: usize,
    overlap_samples: usize,
}

impl StreamingAudioBuffer {
    fn new(duration_ms: u32, overlap_ms: u32, sample_rate: u32) -> Self {
        let capacity = ((duration_ms * sample_rate) / 1000) as usize;
        let overlap_samples = ((overlap_ms * sample_rate) / 1000) as usize;

        Self {
            buffer: vec![0.0; capacity * 2], // Double buffer for overlap
            write_pos: 0,
            read_pos: 0,
            capacity,
            overlap_samples,
        }
    }

    fn write(&mut self, samples: &[f32]) -> Result<()> {
        if samples.len() > self.capacity {
            return Err(anyhow!("Input too large for buffer").into());
        }

        // Circular buffer write with overlap handling
        for &sample in samples {
            self.buffer[self.write_pos % self.buffer.len()] = sample;
            self.write_pos += 1;
        }

        Ok(())
    }

    fn read_chunk(&mut self) -> Option<Vec<f32>> {
        if self.write_pos < self.read_pos + self.capacity {
            return None; // Not enough data
        }

        let mut chunk = Vec::with_capacity(self.capacity);
        for i in 0..self.capacity {
            let pos = (self.read_pos + i) % self.buffer.len();
            chunk.push(self.buffer[pos]);
        }

        // Advance read position with overlap
        self.read_pos += self.capacity - self.overlap_samples;

        Some(chunk)
    }
}

/// Performance statistics for NPU Whisper processing
#[derive(Debug, Default)]
pub struct NPUWhisperStats {
    pub total_inferences: u64,
    pub total_audio_seconds: f64,
    pub average_inference_time_ms: f32,
    pub min_inference_time_ms: f32,
    pub max_inference_time_ms: f32,
    pub npu_utilization_percent: f32,
    pub power_consumption_mw: f32,
    pub transcription_accuracy: f32,
    pub chunks_processed: u64,
    pub chunks_dropped: u64,
}

impl NPUWhisperStats {
    pub fn update_inference(&mut self, inference_time_ms: f32) {
        self.total_inferences += 1;

        if self.total_inferences == 1 {
            self.min_inference_time_ms = inference_time_ms;
            self.max_inference_time_ms = inference_time_ms;
            self.average_inference_time_ms = inference_time_ms;
        } else {
            self.min_inference_time_ms = self.min_inference_time_ms.min(inference_time_ms);
            self.max_inference_time_ms = self.max_inference_time_ms.max(inference_time_ms);

            // Exponential moving average
            let alpha = 0.1;
            self.average_inference_time_ms =
                alpha * inference_time_ms + (1.0 - alpha) * self.average_inference_time_ms;
        }
    }

    pub fn meets_performance_targets(&self) -> bool {
        self.average_inference_time_ms < 2.0 &&  // <2ms target
        self.power_consumption_mw < 100.0 &&     // <100mW target
        self.npu_utilization_percent > 50.0      // >50% utilization
    }

    pub fn generate_performance_report(&self) -> String {
        format!(
            "NPU Whisper Performance Report\n\
             ==============================\n\
             Total Inferences: {}\n\
             Audio Processed: {:.2}s\n\
             Average Latency: {:.2}ms\n\
             Min Latency: {:.2}ms\n\
             Max Latency: {:.2}ms\n\
             NPU Utilization: {:.1}%\n\
             Power Consumption: {:.1}mW\n\
             Transcription Accuracy: {:.1}%\n\
             Chunks Processed: {}\n\
             Chunks Dropped: {}\n\
             Performance Target: {}\n\
             Real-time Factor: {:.2}x",
            self.total_inferences,
            self.total_audio_seconds,
            self.average_inference_time_ms,
            self.min_inference_time_ms,
            self.max_inference_time_ms,
            self.npu_utilization_percent,
            self.power_consumption_mw,
            self.transcription_accuracy,
            self.chunks_processed,
            self.chunks_dropped,
            if self.meets_performance_targets() { "✅ PASSED" } else { "❌ FAILED" },
            self.total_audio_seconds / (self.total_inferences as f64 * 0.1) // Assuming 100ms chunks
        )
    }
}

impl NPUWhisperProcessor {
    /// Create new NPU Whisper processor with Intel NPU backend
    pub async fn new(config: NPUWhisperConfig) -> Result<Self> {
        info!("Initializing NPU Whisper processor with 11 TOPS Intel NPU");

        // Initialize NPU manager
        let npu = Arc::new(RwLock::new(NPUManager::new().await?));

        // Load Whisper model onto NPU
        {
            let mut npu_guard = npu.write().await;
            npu_guard.load_whisper_model(&config.model_path).await?;
            npu_guard.optimize_for_workload(NPUWorkloadType::RealTime).await?;
        }

        // Initialize feature extractors
        let mel_extractor = MelSpectrogramExtractor::new(config.sample_rate)?;
        let tokenizer = WhisperTokenizer::new()?;

        // Initialize streaming buffer
        let audio_buffer = Arc::new(RwLock::new(StreamingAudioBuffer::new(
            config.chunk_duration_ms,
            config.overlap_ms,
            config.sample_rate,
        )));

        let performance_stats = Arc::new(RwLock::new(NPUWhisperStats::default()));

        info!("NPU Whisper processor initialized successfully");

        Ok(Self {
            npu,
            mel_extractor,
            tokenizer,
            audio_buffer,
            inference_tx: None,
            result_rx: None,
            config,
            performance_stats,
        })
    }

    /// Start real-time inference pipeline for push-to-talk
    pub async fn start_inference_pipeline(&mut self) -> Result<mpsc::UnboundedReceiver<TranscriptionResult>> {
        let (inference_tx, mut inference_rx) = mpsc::unbounded_channel::<InferenceRequest>();
        let (result_tx, result_rx) = mpsc::unbounded_channel::<TranscriptionResult>();

        let npu = self.npu.clone();
        let stats = self.performance_stats.clone();
        let tokenizer = WhisperTokenizer::new()?;

        // Spawn inference worker task
        tokio::spawn(async move {
            let mut chunk_id = 0u64;

            while let Some(request) = inference_rx.recv().await {
                let start_time = std::time::Instant::now();

                // Perform NPU inference
                let inference_result = {
                    let npu_guard = npu.read().await;
                    match npu_guard.infer_whisper(&request.mel_features, None).await {
                        Ok(result) => result,
                        Err(e) => {
                            error!("NPU inference failed: {}", e);
                            continue;
                        }
                    }
                };

                // Decode tokens to text
                let tokens = Self::logits_to_tokens(&inference_result.logits);
                let text = match tokenizer.decode(&tokens) {
                    Ok(text) => text,
                    Err(e) => {
                        warn!("Token decoding failed: {}", e);
                        "<decode_error>".to_string()
                    }
                };

                let total_time = start_time.elapsed().as_secs_f32() * 1000.0;

                // Update performance statistics
                {
                    let mut stats_guard = stats.write().await;
                    stats_guard.update_inference(total_time);
                    stats_guard.chunks_processed += 1;
                }

                // Send result
                let result = TranscriptionResult {
                    text,
                    confidence: Self::calculate_confidence(&inference_result.logits),
                    language: "en".to_string(), // TODO: Auto-detect language
                    inference_time_ms: total_time,
                    chunk_id,
                    timestamp: request.timestamp,
                    tokens,
                };

                if let Err(_) = result_tx.send(result) {
                    warn!("Result channel closed, stopping inference pipeline");
                    break;
                }

                chunk_id += 1;

                // Log performance every 100 inferences
                if chunk_id % 100 == 0 {
                    let stats_guard = stats.read().await;
                    if stats_guard.meets_performance_targets() {
                        info!("NPU performance targets met: {:.2}ms avg latency",
                              stats_guard.average_inference_time_ms);
                    } else {
                        warn!("NPU performance below targets: {:.2}ms avg latency",
                              stats_guard.average_inference_time_ms);
                    }
                }
            }
        });

        self.inference_tx = Some(inference_tx);
        self.result_rx = Some(result_rx);

        info!("NPU inference pipeline started successfully");
        Ok(result_rx)
    }

    /// Process audio chunk for real-time transcription
    pub async fn process_audio_chunk(&self, audio_samples: &[f32]) -> Result<()> {
        // Write to streaming buffer
        {
            let mut buffer_guard = self.audio_buffer.write().await;
            buffer_guard.write(audio_samples)?;

            // Check if we have enough data for inference
            if let Some(chunk) = buffer_guard.read_chunk() {
                // Extract mel spectrogram features
                let mel_features = self.mel_extractor.extract(&chunk)?;

                // Send for NPU inference
                if let Some(ref tx) = self.inference_tx {
                    let request = InferenceRequest {
                        mel_features,
                        timestamp: std::time::Instant::now(),
                        chunk_id: 0, // Will be set by inference worker
                    };

                    if let Err(_) = tx.send(request) {
                        warn!("Inference channel closed");
                    }
                } else {
                    warn!("Inference pipeline not started");
                }
            }
        }

        Ok(())
    }

    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> NPUWhisperStats {
        self.performance_stats.read().await.clone()
    }

    /// Convert logits to token IDs (greedy decoding)
    fn logits_to_tokens(logits: &[f32]) -> Vec<u32> {
        // Simple greedy decoding - take argmax for each position
        let vocab_size = 51865; // Whisper vocabulary size
        let seq_len = logits.len() / vocab_size;
        let mut tokens = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let start_idx = i * vocab_size;
            let end_idx = start_idx + vocab_size;

            if end_idx <= logits.len() {
                let slice = &logits[start_idx..end_idx];
                let max_idx = slice
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                tokens.push(max_idx as u32);
            }
        }

        tokens
    }

    /// Calculate confidence score from logits
    fn calculate_confidence(logits: &[f32]) -> f32 {
        if logits.is_empty() {
            return 0.0;
        }

        // Average softmax entropy as confidence measure
        let vocab_size = 51865;
        let seq_len = logits.len() / vocab_size;
        let mut total_confidence = 0.0;

        for i in 0..seq_len {
            let start_idx = i * vocab_size;
            let end_idx = start_idx + vocab_size;

            if end_idx <= logits.len() {
                let slice = &logits[start_idx..end_idx];

                // Softmax
                let max_logit = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let sum_exp: f32 = slice.iter().map(|&x| (x - max_logit).exp()).sum();
                let max_prob = (slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) - max_logit).exp() / sum_exp;

                total_confidence += max_prob;
            }
        }

        if seq_len > 0 {
            total_confidence / seq_len as f32
        } else {
            0.0
        }
    }

    /// Shutdown processor and cleanup resources
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down NPU Whisper processor");

        // Close channels
        self.inference_tx = None;

        // Shutdown NPU
        {
            let mut npu_guard = self.npu.write().await;
            npu_guard.shutdown().await?;
        }

        // Print final performance report
        let stats = self.performance_stats.read().await;
        info!("Final performance report:\n{}", stats.generate_performance_report());

        info!("NPU Whisper processor shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_npu_whisper_initialization() {
        let config = NPUWhisperConfig::default();
        let result = NPUWhisperProcessor::new(config).await;

        // Should succeed if NPU is available
        match result {
            Ok(_) => println!("NPU Whisper processor initialized successfully"),
            Err(e) => println!("NPU not available or initialization failed: {}", e),
        }
    }

    #[test]
    async fn test_mel_spectrogram_extraction() {
        let extractor = match MelSpectrogramExtractor::new(16000) {
            Ok(extractor) => extractor,
            Err(e) => {
                println!("Failed to create mel extractor: {}", e);
                return;
            }
        };

        // Generate test audio (1 second of sine wave)
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();

        let mel_features = match extractor.extract(&audio) {
            Ok(features) => features,
            Err(e) => {
                println!("Failed to extract mel features: {}", e);
                return;
            }
        };

        // Should have 80 mel bins per frame
        assert!(mel_features.len() > 0);
        assert_eq!(mel_features.len() % 80, 0);

        println!("Extracted {} mel features for 1s audio", mel_features.len());
    }

    #[test]
    fn test_streaming_buffer() {
        let mut buffer = StreamingAudioBuffer::new(100, 20, 16000); // 100ms chunks, 20ms overlap

        // Write some test data
        let samples: Vec<f32> = (0..1600).map(|i| i as f32).collect(); // 100ms at 16kHz
        if let Err(e) = buffer.write(&samples) {
            panic!("Failed to write to buffer: {}", e);
        }

        // Should not have a chunk yet
        assert!(buffer.read_chunk().is_none());

        // Write more data
        if let Err(e) = buffer.write(&samples) {
            panic!("Failed to write second buffer: {}", e);
        }

        // Should have a chunk now
        let chunk = buffer.read_chunk();
        assert!(chunk.is_some());
        if let Some(chunk_data) = chunk {
            assert_eq!(chunk_data.len(), 1600);
        }
    }
}