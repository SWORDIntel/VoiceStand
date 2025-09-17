use voicestand_core::{Result, VoiceStandError};
use candle_core::{Device, Tensor, DType};
use std::f32::consts::PI;

/// Mel-scale filterbank feature extraction for Whisper
pub struct MelSpectrogramExtractor {
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    mel_filters: Tensor,
    device: Device,
}

impl MelSpectrogramExtractor {
    pub fn new(sample_rate: u32, n_fft: usize, hop_length: usize, n_mels: usize, device: Device) -> Result<Self> {
        let mel_filters = Self::create_mel_filterbank(sample_rate, n_fft, n_mels, &device)?;

        Ok(Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            mel_filters,
            device,
        })
    }

    /// Extract mel-spectrogram features from audio
    pub fn extract(&self, audio: &[f32]) -> Result<Tensor> {
        // Convert audio to tensor
        let audio_tensor = Tensor::from_slice(audio, audio.len(), &self.device)?;

        // Apply STFT
        let spectrogram = self.stft(&audio_tensor)?;

        // Convert to power spectrogram
        let power_spec = self.power_spectrogram(&spectrogram)?;

        // Apply mel filterbank
        let mel_spec = power_spec.matmul(&self.mel_filters.transpose(0, 1)?)?;

        // Convert to log scale
        let log_mel = (mel_spec + 1e-10)?.log()?;

        Ok(log_mel)
    }

    /// Short-Time Fourier Transform implementation
    fn stft(&self, audio: &Tensor) -> Result<Tensor> {
        let audio_len = audio.dim(0)?;
        let num_frames = (audio_len - self.n_fft) / self.hop_length + 1;

        let mut frames = Vec::new();

        for i in 0..num_frames {
            let start = i * self.hop_length;
            let end = start + self.n_fft;

            if end <= audio_len {
                let frame = audio.narrow(0, start, self.n_fft)?;
                let windowed = self.apply_window(&frame)?;
                let fft_result = self.fft(&windowed)?;
                frames.push(fft_result);
            }
        }

        if frames.is_empty() {
            return Err(VoiceStandError::speech("No frames could be extracted"));
        }

        // Stack frames
        Tensor::stack(&frames, 0)
    }

    /// Apply Hann window
    fn apply_window(&self, frame: &Tensor) -> Result<Tensor> {
        let n = frame.dim(0)?;
        let mut window_vals = Vec::with_capacity(n);

        for i in 0..n {
            let val = 0.5 * (1.0 - (2.0 * PI * i as f32 / (n - 1) as f32).cos());
            window_vals.push(val);
        }

        let window = Tensor::from_slice(&window_vals, n, &self.device)?;
        frame.mul(&window)
    }

    /// Simplified FFT implementation (real-valued input)
    fn fft(&self, signal: &Tensor) -> Result<Tensor> {
        let n = signal.dim(0)?;
        let signal_data = signal.to_vec1::<f32>()?;

        // Simplified FFT - in production would use proper FFT library
        let mut real_parts = Vec::with_capacity(n / 2 + 1);
        let mut imag_parts = Vec::with_capacity(n / 2 + 1);

        for k in 0..=n/2 {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for i in 0..n {
                let angle = -2.0 * PI * (k * i) as f32 / n as f32;
                real_sum += signal_data[i] * angle.cos();
                imag_sum += signal_data[i] * angle.sin();
            }

            real_parts.push(real_sum);
            imag_parts.push(imag_sum);
        }

        // Return magnitude spectrum
        let magnitudes: Vec<f32> = real_parts.iter().zip(imag_parts.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .collect();

        Tensor::from_slice(&magnitudes, magnitudes.len(), &self.device)
    }

    /// Convert complex spectrogram to power spectrogram
    fn power_spectrogram(&self, spectrogram: &Tensor) -> Result<Tensor> {
        // For simplified implementation, assume magnitude spectrum is already computed
        spectrogram.sqr()
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize, device: &Device) -> Result<Tensor> {
        let n_freqs = n_fft / 2 + 1;
        let fmax = sample_rate as f32 / 2.0;

        // Create mel-spaced frequencies
        let mel_freqs = Self::mel_frequencies(n_mels + 2, 0.0, fmax);

        // Convert mel frequencies to Hz
        let hz_freqs: Vec<f32> = mel_freqs.iter().map(|&mel| Self::mel_to_hz(mel)).collect();

        // Convert Hz to FFT bin indices
        let bin_indices: Vec<f32> = hz_freqs.iter()
            .map(|&hz| hz * n_fft as f32 / sample_rate as f32)
            .collect();

        // Create filterbank matrix
        let mut filterbank = vec![vec![0.0; n_freqs]; n_mels];

        for m in 0..n_mels {
            let left = bin_indices[m];
            let center = bin_indices[m + 1];
            let right = bin_indices[m + 2];

            for k in 0..n_freqs {
                let freq = k as f32;
                if freq >= left && freq <= center {
                    filterbank[m][k] = (freq - left) / (center - left);
                } else if freq >= center && freq <= right {
                    filterbank[m][k] = (right - freq) / (right - center);
                }
            }
        }

        // Convert to tensor
        let flat_filterbank: Vec<f32> = filterbank.into_iter().flatten().collect();
        Tensor::from_slice(&flat_filterbank, (n_mels, n_freqs), device)
    }

    /// Generate mel-spaced frequencies
    fn mel_frequencies(n_mels: usize, f_min: f32, f_max: f32) -> Vec<f32> {
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        (0..n_mels)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels - 1) as f32)
            .collect()
    }

    /// Convert Hz to mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Get feature dimensions
    pub fn feature_dim(&self) -> usize {
        self.n_mels
    }

    /// Get the number of frames for given audio length
    pub fn num_frames(&self, audio_len: usize) -> usize {
        if audio_len < self.n_fft {
            0
        } else {
            (audio_len - self.n_fft) / self.hop_length + 1
        }
    }
}

/// Default Whisper mel-spectrogram parameters
impl Default for MelSpectrogramExtractor {
    fn default() -> Self {
        let device = Device::Cpu;
        Self::new(16000, 400, 160, 80, device).expect("Default mel-spectrogram should create successfully")
    }
}

/// Padding utilities for audio preprocessing
pub fn pad_or_trim(audio: &[f32], target_length: usize) -> Vec<f32> {
    let mut result = vec![0.0; target_length];
    let copy_len = audio.len().min(target_length);
    result[..copy_len].copy_from_slice(&audio[..copy_len]);
    result
}

/// Normalize audio to [-1, 1] range
pub fn normalize_audio(audio: &mut [f32]) {
    if audio.is_empty() {
        return;
    }

    let max_val = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    if max_val > 0.0 {
        let scale = 1.0 / max_val;
        for sample in audio.iter_mut() {
            *sample *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_conversion() {
        let hz = 1000.0;
        let mel = MelSpectrogramExtractor::hz_to_mel(hz);
        let hz_back = MelSpectrogramExtractor::mel_to_hz(mel);

        assert!((hz - hz_back).abs() < 0.1);
    }

    #[test]
    fn test_mel_frequencies() {
        let freqs = MelSpectrogramExtractor::mel_frequencies(5, 0.0, 8000.0);
        assert_eq!(freqs.len(), 5);
        assert!((freqs[0] - MelSpectrogramExtractor::hz_to_mel(0.0)).abs() < 0.1);
        assert!((freqs[4] - MelSpectrogramExtractor::hz_to_mel(8000.0)).abs() < 0.1);
    }

    #[test]
    fn test_pad_or_trim() {
        let audio = vec![1.0, 2.0, 3.0];

        // Test trimming
        let trimmed = pad_or_trim(&audio, 2);
        assert_eq!(trimmed, vec![1.0, 2.0]);

        // Test padding
        let padded = pad_or_trim(&audio, 5);
        assert_eq!(padded, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_audio() {
        let mut audio = vec![0.5, -1.0, 0.8, -0.3];
        normalize_audio(&mut audio);

        let max_val = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((max_val - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mel_extractor_creation() {
        let device = Device::Cpu;
        let extractor = MelSpectrogramExtractor::new(16000, 400, 160, 80, device);
        assert!(extractor.is_ok());

        let extractor = extractor.expect("Test extractor should create successfully");
        assert_eq!(extractor.feature_dim(), 80);
    }

    #[test]
    fn test_num_frames_calculation() {
        let device = Device::Cpu;
        let extractor = MelSpectrogramExtractor::new(16000, 400, 160, 80, device).expect("Test extractor should create successfully");

        assert_eq!(extractor.num_frames(400), 1);
        assert_eq!(extractor.num_frames(560), 2);
        assert_eq!(extractor.num_frames(200), 0);
    }
}