// Audio processing pipeline
use crate::{AudioError, AudioFrame, AudioSample, VADConfig, VADResult};
use crate::processing::{AudioProcessor, AudioStats};
use crate::vad::VoiceActivityDetector;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, warn, error};

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub frames_per_buffer: u32,
    pub max_latency_ms: f32,
    pub enable_vad: bool,
    pub enable_noise_reduction: bool,
    pub vad_threshold: f32,
    pub noise_gate_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            frames_per_buffer: 1024,
            max_latency_ms: 10.0,
            enable_vad: true,
            enable_noise_reduction: true,
            vad_threshold: 0.02,
            noise_gate_threshold: 0.01,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PipelineEvent {
    AudioProcessed {
        samples: Vec<f32>,
        stats: AudioStats,
        vad_result: Option<VADResult>,
    },
    VoiceDetected { confidence: f32 },
    SilenceDetected,
    LatencyWarning { actual_ms: f32 },
    ProcessingComplete {
        duration_ms: f32,
        samples_processed: usize,
    },
    Error { error: AudioError },
}

pub struct AudioPipeline {
    config: PipelineConfig,
    processor: AudioProcessor,
    vad: VoiceActivityDetector,
    event_sender: mpsc::UnboundedSender<PipelineEvent>,
    sequence_number: u64,
    total_samples_processed: usize,
}

impl AudioPipeline {
    pub fn new(config: PipelineConfig) -> Result<(Self, mpsc::UnboundedReceiver<PipelineEvent>), AudioError> {
        let (sender, receiver) = mpsc::unbounded_channel();

        // Create audio processor
        let processor = AudioProcessor::new(
            config.sample_rate,
            config.frames_per_buffer as usize,
        );

        // Create VAD with configuration
        let vad_config = VADConfig {
            sample_rate: config.sample_rate,
            frame_size: config.frames_per_buffer as usize,
            energy_threshold: config.vad_threshold,
            silence_duration_ms: 500,
            voice_duration_ms: 100,
        };
        let vad = VoiceActivityDetector::new(vad_config);

        debug!("Audio pipeline initialized - sample_rate: {}, channels: {}, frames_per_buffer: {}",
               config.sample_rate, config.channels, config.frames_per_buffer);

        Ok((Self {
            config,
            processor,
            vad,
            event_sender: sender,
            sequence_number: 0,
            total_samples_processed: 0,
        }, receiver))
    }

    /// Process audio frame with full pipeline
    pub fn process(&mut self, audio_data: &[f32]) -> Result<Vec<f32>, AudioError> {
        let start_time = Instant::now();
        let mut samples = audio_data.to_vec();

        if samples.is_empty() {
            return Ok(samples);
        }

        self.sequence_number += 1;
        self.total_samples_processed += samples.len();

        debug!("Processing audio frame: {} samples, sequence: {}",
               samples.len(), self.sequence_number);

        // Step 1: Audio enhancement and noise reduction
        let stats = if self.config.enable_noise_reduction {
            match self.processor.process_for_recognition(&mut samples) {
                Ok(stats) => {
                    debug!("Audio processing complete - SNR improvement: {:.2}dB, spectral centroid: {:.1}Hz",
                           stats.snr_improvement, stats.spectral_centroid);
                    stats
                }
                Err(e) => {
                    error!("Audio processing failed: {}", e);
                    let _ = self.event_sender.send(PipelineEvent::Error {
                        error: AudioError::ProcessingError {
                            stage: "audio_enhancement".to_string(),
                            reason: e.to_string(),
                        }
                    });
                    // Continue with unprocessed audio
                    AudioStats {
                        original_energy: 0.0,
                        processed_energy: 0.0,
                        snr_improvement: 0.0,
                        spectral_centroid: 0.0,
                        clipping_detected: false,
                    }
                }
            }
        } else {
            // Basic stats without processing
            AudioStats {
                original_energy: self.processor.calculate_energy(&samples),
                processed_energy: self.processor.calculate_energy(&samples),
                snr_improvement: 0.0,
                spectral_centroid: self.processor.spectral_centroid(&samples),
                clipping_detected: samples.iter().any(|&x| x.abs() >= 0.99),
            }
        };

        // Step 2: Voice Activity Detection
        let vad_result = if self.config.enable_vad {
            match self.vad.process(&samples) {
                Ok(result) => {
                    // Send voice/silence events
                    if result.has_voice {
                        debug!("Voice detected - confidence: {:.3}, energy: {:.6}",
                               result.confidence, result.energy_level);
                        let _ = self.event_sender.send(PipelineEvent::VoiceDetected {
                            confidence: result.confidence
                        });
                    } else {
                        let _ = self.event_sender.send(PipelineEvent::SilenceDetected);
                    }
                    Some(result)
                }
                Err(e) => {
                    warn!("VAD processing failed: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 3: Check latency compliance
        let processing_time = start_time.elapsed();
        let processing_ms = processing_time.as_secs_f32() * 1000.0;

        if processing_ms > self.config.max_latency_ms {
            warn!("Processing latency exceeded target: {:.2}ms > {:.2}ms",
                  processing_ms, self.config.max_latency_ms);
            let _ = self.event_sender.send(PipelineEvent::LatencyWarning {
                actual_ms: processing_ms
            });
        }

        // Step 4: Send completion event
        let _ = self.event_sender.send(PipelineEvent::AudioProcessed {
            samples: samples.clone(),
            stats,
            vad_result,
        });

        let _ = self.event_sender.send(PipelineEvent::ProcessingComplete {
            duration_ms: processing_ms,
            samples_processed: samples.len(),
        });

        debug!("Audio pipeline processing complete - {:.2}ms, {} samples",
               processing_ms, samples.len());

        Ok(samples)
    }

    /// Process audio frame and return processing statistics
    pub fn process_with_stats(&mut self, audio_data: &[f32]) -> Result<(Vec<f32>, AudioStats, Option<VADResult>), AudioError> {
        let start_time = Instant::now();
        let mut samples = audio_data.to_vec();

        if samples.is_empty() {
            return Ok((samples, AudioStats {
                original_energy: 0.0,
                processed_energy: 0.0,
                snr_improvement: 0.0,
                spectral_centroid: 0.0,
                clipping_detected: false,
            }, None));
        }

        self.sequence_number += 1;
        self.total_samples_processed += samples.len();

        // Audio processing
        let stats = if self.config.enable_noise_reduction {
            self.processor.process_for_recognition(&mut samples)
                .map_err(|e| AudioError::ProcessingError {
                    stage: "audio_enhancement".to_string(),
                    reason: e.to_string(),
                })?
        } else {
            AudioStats {
                original_energy: self.processor.calculate_energy(&samples),
                processed_energy: self.processor.calculate_energy(&samples),
                snr_improvement: 0.0,
                spectral_centroid: self.processor.spectral_centroid(&samples),
                clipping_detected: samples.iter().any(|&x| x.abs() >= 0.99),
            }
        };

        // Voice Activity Detection
        let vad_result = if self.config.enable_vad {
            Some(self.vad.process(&samples)
                .map_err(|e| AudioError::ProcessingError {
                    stage: "vad".to_string(),
                    reason: e.to_string(),
                })?)
        } else {
            None
        };

        Ok((samples, stats, vad_result))
    }

    /// Get pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Get total samples processed
    pub fn total_samples_processed(&self) -> usize {
        self.total_samples_processed
    }

    /// Get current sequence number
    pub fn sequence_number(&self) -> u64 {
        self.sequence_number
    }

    /// Reset pipeline state
    pub fn reset(&mut self) {
        self.sequence_number = 0;
        self.total_samples_processed = 0;
        debug!("Audio pipeline state reset");
    }

    /// Update pipeline configuration
    pub fn update_config(&mut self, config: PipelineConfig) -> Result<(), AudioError> {
        if config.sample_rate != self.config.sample_rate {
            return Err(AudioError::ConfigError {
                parameter: "sample_rate".to_string(),
                reason: "Cannot change sample rate at runtime".to_string(),
            });
        }

        self.config = config;

        // Update processor parameters
        self.processor.update_parameters(
            self.config.noise_gate_threshold,
            0.5, // noise reduction factor
        );

        debug!("Pipeline configuration updated");
        Ok(())
    }
}