use crate::{VoiceActivityDetector, VADResult};
use crate::buffer::StreamingBuffer;
use voicestand_core::{AudioConfig, AudioData, AudioCaptureConfig, AudioDevice, Result, VoiceStandError};

use cpal::{Device, Stream, StreamConfig, SampleFormat, SampleRate};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::Sender;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::SystemTime;

/// Memory-safe audio capture system using CPAL
pub struct AudioCapture {
    device: Option<Device>,
    stream: Option<Stream>,
    config: AudioConfig,
    event_sender: Sender<AppEvent>,
    vad: Arc<Mutex<VoiceActivityDetector>>,
    streaming_buffer: Arc<Mutex<StreamingBuffer>>,
    is_recording: Arc<Mutex<bool>>,
}

impl AudioCapture {
    pub fn new(config: AudioConfig, event_sender: Sender<AppEvent>) -> Result<Self> {
        let vad = Arc::new(Mutex::new(VoiceActivityDetector::new(config.vad_threshold, 50)));
        let streaming_buffer = Arc::new(Mutex::new(StreamingBuffer::new(
            config.frames_per_buffer as usize,
            0.2, // 20% overlap
        )));

        Ok(Self {
            device: None,
            stream: None,
            config,
            event_sender,
            vad,
            streaming_buffer,
            is_recording: Arc::new(Mutex::new(false)),
        })
    }

    /// Initialize audio capture with the specified device
    pub fn initialize(&mut self, device_name: Option<String>) -> Result<()> {
        let host = cpal::default_host();

        // Select device
        let device = if let Some(name) = device_name {
            self.find_device_by_name(&host, &name)?
        } else {
            host.default_input_device()
                .ok_or_else(|| VoiceStandError::audio("No default input device available"))?
        };

        // Get supported config
        let supported_configs = device.supported_input_configs()
            .map_err(|e| VoiceStandError::audio(format!("Failed to get supported configs: {}", e)))?;

        let supported_config = supported_configs
            .filter(|config| config.channels() == self.config.channels)
            .find(|config| {
                config.min_sample_rate() <= SampleRate(self.config.sample_rate) &&
                config.max_sample_rate() >= SampleRate(self.config.sample_rate)
            })
            .ok_or_else(|| VoiceStandError::audio("No supported audio configuration found"))?;

        let stream_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.frames_per_buffer),
        };

        tracing::info!(
            "Initializing audio capture: {} channels, {} Hz, {} frames",
            stream_config.channels,
            stream_config.sample_rate.0,
            self.config.frames_per_buffer
        );

        self.device = Some(device);

        Ok(())
    }

    /// Start audio capture
    pub fn start(&mut self) -> Result<()> {
        let device = self.device.as_ref()
            .ok_or_else(|| VoiceStandError::audio("Device not initialized"))?;

        let stream_config = StreamConfig {
            channels: self.config.channels,
            sample_rate: SampleRate(self.config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.frames_per_buffer),
        };

        // Create stream with proper error handling
        let event_sender_data = self.event_sender.clone();
        let event_sender_error = self.event_sender.clone();
        let vad = Arc::clone(&self.vad);
        let streaming_buffer = Arc::clone(&self.streaming_buffer);
        let is_recording = Arc::clone(&self.is_recording);

        let stream = device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !*is_recording.lock() {
                    return;
                }

                // Process audio data safely
                if let Err(e) = Self::process_audio_data(
                    data,
                    &event_sender_data,
                    &vad,
                    &streaming_buffer,
                    stream_config.sample_rate.0,
                    stream_config.channels,
                ) {
                    tracing::error!("Audio processing error: {}", e);
                    let _ = event_sender_data.send(AppEvent::Error(format!("Audio processing error: {}", e)));
                }
            },
            move |err| {
                tracing::error!("Audio stream error: {}", err);
                let _ = event_sender_error.send(AppEvent::Error(format!("Audio stream error: {}", err)));
            },
            None,
        ).map_err(|e| VoiceStandError::audio(format!("Failed to build input stream: {}", e)))?;

        stream.play().map_err(|e| VoiceStandError::audio(format!("Failed to start stream: {}", e)))?;

        self.stream = Some(stream);
        *self.is_recording.lock() = true;

        tracing::info!("Audio capture started");
        Ok(())
    }

    /// Stop audio capture
    pub fn stop(&mut self) -> Result<()> {
        *self.is_recording.lock() = false;

        if let Some(stream) = self.stream.take() {
            stream.pause().map_err(|e| VoiceStandError::audio(format!("Failed to stop stream: {}", e)))?;
        }

        // Clear buffers
        self.streaming_buffer.lock().clear();
        self.vad.lock().reset();

        tracing::info!("Audio capture stopped");
        Ok(())
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        *self.is_recording.lock()
    }

    /// Update VAD parameters
    pub fn update_vad_parameters(&self, threshold: f32, min_speech_frames: u32, min_silence_frames: u32) {
        self.vad.lock().update_parameters(threshold, min_speech_frames, min_silence_frames);
    }

    /// Get audio device information
    pub fn get_device_info(&self) -> Option<String> {
        self.device.as_ref()
            .and_then(|device| device.name().ok())
    }

    /// Find device by name
    fn find_device_by_name(&self, host: &cpal::Host, name: &str) -> Result<Device> {
        for device in host.input_devices()
            .map_err(|e| VoiceStandError::audio(format!("Failed to enumerate devices: {}", e)))? {

            if let Ok(device_name) = device.name() {
                if device_name == name {
                    return Ok(device);
                }
            }
        }

        Err(VoiceStandError::audio(format!("Device '{}' not found", name)))
    }

    /// Process incoming audio data (static method for use in callback)
    fn process_audio_data(
        data: &[f32],
        event_sender: &Sender<AppEvent>,
        vad: &Arc<Mutex<VoiceActivityDetector>>,
        streaming_buffer: &Arc<Mutex<StreamingBuffer>>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<()> {
        // Add to streaming buffer
        streaming_buffer.lock().push(data)?;

        // Process with VAD
        let vad_result = vad.lock().process(data);

        // Send speech detection events
        if vad_result.state_changed {
            let event = AppEvent::SpeechDetected {
                is_start: vad_result.is_speech,
                timestamp: SystemTime::now(),
            };
            event_sender.send(event)
                .map_err(|_| VoiceStandError::audio("Failed to send speech detection event"))?;
        }

        // Create AudioData and send if speech is detected or buffer is ready
        if vad_result.is_speech || streaming_buffer.lock().stats().available_samples >= sample_rate as usize {
            let audio_data = AudioData {
                samples: data.to_vec(),
                sample_rate,
                channels,
                timestamp: SystemTime::now(),
                is_speech_end: !vad_result.is_speech && vad_result.state_changed,
            };

            event_sender.send(AppEvent::AudioDataReceived(audio_data))
                .map_err(|_| VoiceStandError::audio("Failed to send audio data event"))?;
        }

        Ok(())
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        if let Err(e) = self.stop() {
            tracing::error!("Error stopping audio capture during drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    #[test]
    fn test_audio_capture_creation() {
        let config = AudioConfig::default();
        let (sender, _receiver) = unbounded();

        let capture = AudioCapture::new(config, sender);
        assert!(capture.is_ok());
    }

    #[test]
    fn test_audio_capture_state() {
        let config = AudioConfig::default();
        let (sender, _receiver) = unbounded();

        let capture = AudioCapture::new(config, sender).unwrap();
        assert!(!capture.is_recording());
    }
}