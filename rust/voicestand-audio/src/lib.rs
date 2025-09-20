//! VoiceStand Audio Processing Library
//!
//! Lock-free audio pipeline with <10ms latency for real-time voice processing.
//! Provides memory-safe audio capture, buffering, and processing.

use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

pub mod buffer;
pub mod capture;
pub mod processing;
pub mod vad;
pub mod pipeline;

pub use buffer::{CircularBuffer};
pub use capture::AudioCapture;
pub use processing::{AudioProcessor, AudioStats};
pub use vad::{VoiceActivityDetector, VADConfig, VADResult};
pub use pipeline::{AudioPipeline, PipelineConfig, PipelineEvent};

// Re-export types from voicestand-core for compatibility
pub use voicestand_core::{AudioCaptureConfig, AudioDevice, AudioSample, AudioFrame};

/// Audio processing error types
#[derive(Error, Debug, Clone)]
pub enum AudioError {
    /// Device not available
    #[error("Audio device not available: {device}")]
    DeviceNotAvailable { device: String },

    /// Configuration error
    #[error("Audio configuration error: {parameter} - {reason}")]
    ConfigError { parameter: String, reason: String },

    /// Buffer operation failed
    #[error("Buffer operation failed: {operation} - {reason}")]
    BufferError { operation: String, reason: String },

    /// Capture operation failed
    #[error("Audio capture failed: {reason}")]
    CaptureError { reason: String },

    /// Processing operation failed
    #[error("Audio processing failed: {stage} - {reason}")]
    ProcessingError { stage: String, reason: String },

    /// Latency target not met
    #[error("Latency target exceeded: {actual_ms}ms > {target_ms}ms")]
    LatencyExceeded { actual_ms: f32, target_ms: f32 },

    /// Sample rate conversion failed
    #[error("Sample rate conversion failed: {from_rate} -> {to_rate}")]
    SampleRateError { from_rate: u32, to_rate: u32 },

    /// Channel configuration error
    #[error("Channel configuration error: {channels} channels not supported")]
    ChannelError { channels: u32 },

    /// Format error
    #[error("Audio format error: {format} not supported")]
    FormatError { format: String },

    /// Timeout occurred
    #[error("Audio operation timed out after {timeout_ms}ms: {operation}")]
    Timeout { operation: String, timeout_ms: u64 },

    /// System error
    #[error("Audio system error: {operation}")]
    SystemError {
        operation: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Audio operation result type
pub type AudioResult<T> = Result<T, AudioError>;

/// Audio processing parameters
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Audio format
    pub format: AudioFormat,
    /// Buffer size in samples
    pub buffer_size: usize,
    /// Target latency in milliseconds
    pub target_latency_ms: f32,
    /// Enable voice activity detection
    pub enable_vad: bool,
    /// Enable noise reduction
    pub enable_noise_reduction: bool,
    /// Enable automatic gain control
    pub enable_agc: bool,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,      // 16kHz for speech
            channels: 1,             // Mono
            format: AudioFormat::F32,
            buffer_size: 480,        // 30ms at 16kHz
            target_latency_ms: 10.0, // <10ms target
            enable_vad: true,
            enable_noise_reduction: false,
            enable_agc: false,
        }
    }
}

/// Audio sample formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    /// 32-bit floating point
    F32,
    /// 16-bit signed integer
    I16,
    /// 24-bit signed integer
    I24,
    /// 32-bit signed integer
    I32,
}

impl AudioFormat {
    /// Get sample size in bytes
    pub fn sample_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::I16 => 2,
            Self::I24 => 3,
            Self::I32 => 4,
        }
    }

    /// Check if format is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32)
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I16 => "i16",
            Self::I24 => "i24",
            Self::I32 => "i32",
        }
    }
}

/// Audio sample trait for generic processing
pub trait AudioSample: Copy + Clone + Send + Sync + 'static {
    /// Convert to f32 for processing
    fn to_f32(self) -> f32;
    /// Convert from f32
    fn from_f32(value: f32) -> Self;
    /// Zero value
    fn zero() -> Self;
    /// Check if sample is silence
    fn is_silence(self, threshold: f32) -> bool;
}

impl AudioSample for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value
    }

    fn zero() -> Self {
        0.0
    }

    fn is_silence(self, threshold: f32) -> bool {
        self.abs() < threshold
    }
}

impl AudioSample for i16 {
    fn to_f32(self) -> f32 {
        self as f32 / i16::MAX as f32
    }

    fn from_f32(value: f32) -> Self {
        (value * i16::MAX as f32) as i16
    }

    fn zero() -> Self {
        0
    }

    fn is_silence(self, threshold: f32) -> bool {
        (self as f32).abs() < threshold * i16::MAX as f32
    }
}

impl AudioSample for i32 {
    fn to_f32(self) -> f32 {
        self as f32 / i32::MAX as f32
    }

    fn from_f32(value: f32) -> Self {
        (value * i32::MAX as f32) as i32
    }

    fn zero() -> Self {
        0
    }

    fn is_silence(self, threshold: f32) -> bool {
        (self as f32).abs() < threshold * i32::MAX as f32
    }
}

/// Audio frame containing samples and metadata
#[derive(Debug, Clone)]
pub struct AudioFrame<T: AudioSample> {
    /// Audio samples
    pub samples: Vec<T>,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Timestamp
    pub timestamp: std::time::Instant,
    /// Frame sequence number
    pub sequence: u64,
}

impl<T: AudioSample> AudioFrame<T> {
    /// Create new audio frame
    pub fn new(samples: Vec<T>, sample_rate: u32, channels: u32, sequence: u64) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
            timestamp: std::time::Instant::now(),
            sequence,
        }
    }

    /// Get frame duration in milliseconds
    pub fn duration_ms(&self) -> f32 {
        (self.samples.len() as f32 / self.channels as f32) / self.sample_rate as f32 * 1000.0
    }

    /// Get number of samples per channel
    pub fn samples_per_channel(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Check if frame contains silence
    pub fn is_silence(&self, threshold: f32) -> bool {
        self.samples.iter().all(|&sample| sample.is_silence(threshold))
    }

    /// Convert to f32 format
    pub fn to_f32(&self) -> AudioFrame<f32> {
        AudioFrame {
            samples: self.samples.iter().map(|&s| s.to_f32()).collect(),
            sample_rate: self.sample_rate,
            channels: self.channels,
            timestamp: self.timestamp,
            sequence: self.sequence,
        }
    }

    /// Get peak amplitude
    pub fn peak_amplitude(&self) -> f32 {
        self.samples
            .iter()
            .map(|&s| s.to_f32().abs())
            .fold(0.0f32, f32::max)
    }

    /// Get RMS (root mean square) amplitude
    pub fn rms_amplitude(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = self.samples
            .iter()
            .map(|&s| {
                let f = s.to_f32();
                f * f
            })
            .sum();

        (sum_squares / self.samples.len() as f32).sqrt()
    }

    /// Apply gain to all samples
    pub fn apply_gain(&mut self, gain: f32) {
        for sample in &mut self.samples {
            *sample = T::from_f32(sample.to_f32() * gain);
        }
    }

    /// Mix with another frame (in-place)
    pub fn mix_with(&mut self, other: &AudioFrame<T>, mix_ratio: f32) -> AudioResult<()> {
        if self.samples.len() != other.samples.len() {
            return Err(AudioError::ProcessingError {
                stage: "mixing".to_string(),
                reason: "Frame size mismatch".to_string(),
            });
        }

        if self.sample_rate != other.sample_rate {
            return Err(AudioError::SampleRateError {
                from_rate: other.sample_rate,
                to_rate: self.sample_rate,
            });
        }

        for (self_sample, &other_sample) in self.samples.iter_mut().zip(&other.samples) {
            let mixed = self_sample.to_f32() * (1.0 - mix_ratio) + other_sample.to_f32() * mix_ratio;
            *self_sample = T::from_f32(mixed);
        }

        Ok(())
    }
}

/// Audio metrics for monitoring
#[derive(Debug, Clone)]
pub struct AudioMetrics {
    /// Total frames processed
    pub frames_processed: u64,
    /// Average frame latency in milliseconds
    pub average_latency_ms: f32,
    /// Peak latency in milliseconds
    pub peak_latency_ms: f32,
    /// Buffer underruns
    pub buffer_underruns: u64,
    /// Buffer overruns
    pub buffer_overruns: u64,
    /// Dropped frames
    pub dropped_frames: u64,
    /// Current buffer usage percentage
    pub buffer_usage_percent: f32,
    /// Voice activity detection rate
    pub vad_detection_rate: f32,
    /// Average noise level
    pub average_noise_level: f32,
    /// Peak noise level
    pub peak_noise_level: f32,
}

impl Default for AudioMetrics {
    fn default() -> Self {
        Self {
            frames_processed: 0,
            average_latency_ms: 0.0,
            peak_latency_ms: 0.0,
            buffer_underruns: 0,
            buffer_overruns: 0,
            dropped_frames: 0,
            buffer_usage_percent: 0.0,
            vad_detection_rate: 0.0,
            average_noise_level: 0.0,
            peak_noise_level: 0.0,
        }
    }
}

impl AudioMetrics {
    /// Update latency statistics
    pub fn update_latency(&mut self, latency_ms: f32) {
        if self.frames_processed == 0 {
            self.average_latency_ms = latency_ms;
        } else {
            // Exponential moving average
            self.average_latency_ms = 0.9 * self.average_latency_ms + 0.1 * latency_ms;
        }

        if latency_ms > self.peak_latency_ms {
            self.peak_latency_ms = latency_ms;
        }

        self.frames_processed += 1;
    }

    /// Record buffer underrun
    pub fn record_underrun(&mut self) {
        self.buffer_underruns += 1;
    }

    /// Record buffer overrun
    pub fn record_overrun(&mut self) {
        self.buffer_overruns += 1;
    }

    /// Record dropped frame
    pub fn record_dropped_frame(&mut self) {
        self.dropped_frames += 1;
    }

    /// Update buffer usage
    pub fn update_buffer_usage(&mut self, usage_percent: f32) {
        self.buffer_usage_percent = usage_percent;
    }

    /// Update noise level
    pub fn update_noise_level(&mut self, noise_level: f32) {
        self.average_noise_level = 0.9 * self.average_noise_level + 0.1 * noise_level;
        if noise_level > self.peak_noise_level {
            self.peak_noise_level = noise_level;
        }
    }

    /// Update VAD detection rate
    pub fn update_vad_rate(&mut self, detection_rate: f32) {
        self.vad_detection_rate = detection_rate;
    }

    /// Check if metrics indicate healthy operation
    pub fn is_healthy(&self) -> bool {
        self.average_latency_ms <= 10.0 // <10ms latency target
            && self.buffer_usage_percent < 90.0 // Buffer not too full
            && (self.buffer_underruns + self.buffer_overruns) < self.frames_processed / 1000 // <0.1% buffer issues
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "=== Audio Performance Metrics ===\n\
             Frames Processed: {}\n\
             Average Latency: {:.2}ms\n\
             Peak Latency: {:.2}ms\n\
             Buffer Underruns: {} ({:.3}%)\n\
             Buffer Overruns: {} ({:.3}%)\n\
             Dropped Frames: {} ({:.3}%)\n\
             Buffer Usage: {:.1}%\n\
             VAD Detection Rate: {:.1}%\n\
             Average Noise Level: {:.3}\n\
             Peak Noise Level: {:.3}\n\
             Health Status: {}\n",
            self.frames_processed,
            self.average_latency_ms,
            self.peak_latency_ms,
            self.buffer_underruns,
            if self.frames_processed > 0 { self.buffer_underruns as f32 / self.frames_processed as f32 * 100.0 } else { 0.0 },
            self.buffer_overruns,
            if self.frames_processed > 0 { self.buffer_overruns as f32 / self.frames_processed as f32 * 100.0 } else { 0.0 },
            self.dropped_frames,
            if self.frames_processed > 0 { self.dropped_frames as f32 / self.frames_processed as f32 * 100.0 } else { 0.0 },
            self.buffer_usage_percent,
            self.vad_detection_rate * 100.0,
            self.average_noise_level,
            self.peak_noise_level,
            if self.is_healthy() { "✅ HEALTHY" } else { "⚠️ ISSUES DETECTED" }
        )
    }
}

/// Utility functions
pub mod utils {
    use super::*;

    /// Convert sample rate
    pub fn convert_sample_rate<T: AudioSample>(
        input: &[T],
        from_rate: u32,
        to_rate: u32,
    ) -> AudioResult<Vec<T>> {
        if from_rate == to_rate {
            return Ok(input.to_vec());
        }

        // Simple linear interpolation for sample rate conversion
        let ratio = to_rate as f64 / from_rate as f64;
        let output_len = (input.len() as f64 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let source_index = i as f64 / ratio;
            let index = source_index.floor() as usize;
            let frac = source_index.fract() as f32;

            if index + 1 < input.len() {
                // Linear interpolation
                let sample1 = input[index].to_f32();
                let sample2 = input[index + 1].to_f32();
                let interpolated = sample1 * (1.0 - frac) + sample2 * frac;
                output.push(T::from_f32(interpolated));
            } else if index < input.len() {
                output.push(input[index]);
            } else {
                output.push(T::zero());
            }
        }

        Ok(output)
    }

    /// Apply simple high-pass filter
    pub fn high_pass_filter<T: AudioSample>(
        input: &mut [T],
        cutoff_freq: f32,
        sample_rate: u32,
    ) {
        if input.len() < 2 {
            return;
        }

        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_freq);
        let dt = 1.0 / sample_rate as f32;
        let alpha = rc / (rc + dt);

        let mut prev_input = input[0].to_f32();
        let mut prev_output = 0.0f32;

        for sample in input.iter_mut() {
            let current_input = sample.to_f32();
            let output = alpha * (prev_output + current_input - prev_input);

            *sample = T::from_f32(output);

            prev_input = current_input;
            prev_output = output;
        }
    }

    /// Apply simple low-pass filter
    pub fn low_pass_filter<T: AudioSample>(
        input: &mut [T],
        cutoff_freq: f32,
        sample_rate: u32,
    ) {
        let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_freq);
        let dt = 1.0 / sample_rate as f32;
        let alpha = dt / (rc + dt);

        let mut prev_output = input[0].to_f32();

        for sample in input.iter_mut() {
            let current_input = sample.to_f32();
            let output = prev_output + alpha * (current_input - prev_output);

            *sample = T::from_f32(output);
            prev_output = output;
        }
    }

    /// Calculate frame energy
    pub fn frame_energy<T: AudioSample>(frame: &[T]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = frame
            .iter()
            .map(|&s| {
                let f = s.to_f32();
                f * f
            })
            .sum();

        sum_squares / frame.len() as f32
    }

    /// Detect silence in frame
    pub fn detect_silence<T: AudioSample>(frame: &[T], threshold: f32) -> bool {
        let energy = frame_energy(frame);
        energy < threshold * threshold
    }

    /// Apply automatic gain control
    pub fn apply_agc<T: AudioSample>(
        frame: &mut [T],
        target_level: f32,
        max_gain: f32,
    ) -> f32 {
        let current_level = frame_energy(frame).sqrt();

        if current_level < 1e-6 {
            return 1.0; // No gain for silence
        }

        let gain = (target_level / current_level).min(max_gain);

        for sample in frame.iter_mut() {
            *sample = T::from_f32(sample.to_f32() * gain);
        }

        gain
    }
}

/// Audio device information (legacy compatibility)
#[derive(Debug, Clone)]
pub struct AudioDeviceInfo {
    pub name: String,
    pub is_input: bool,
    pub is_default: bool,
}