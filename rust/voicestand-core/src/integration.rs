//! VoiceStand Integration Layer
//!
//! Memory-safe integration of NPU, GNA, audio, and state management components.
//! Provides the complete push-to-talk system with fallback mechanisms.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::asr_runtime::AsrMetrics;
use crate::config::VoiceStandConfig;
use crate::transcript_merge::merge_word_overlap;
use crate::transcript_stabilizer::TranscriptStabilizer;
use crate::AsrRuntime;
use voicestand_asr::Transcript;
use voicestand_types::{Result, SystemState, SystemStatus, TranscriptionResult, VoiceStandError};

/// Integration manager for coordinating all subsystems
pub struct VoiceStandIntegration {
    config: VoiceStandConfig,
    hardware_manager: Option<Arc<RwLock<voicestand_hardware::HardwareManager>>>,
    audio_pipeline: Option<Arc<RwLock<voicestand_audio::AudioPipeline>>>,
    audio_capture: Option<voicestand_audio::AudioCapture>,
    utterance_assembler: Option<Arc<parking_lot::Mutex<voicestand_audio::UtteranceAssembler>>>,
    state_coordinator: Option<Arc<RwLock<voicestand_state::VoiceStandCoordinator>>>,
    asr_runtime: Option<AsrRuntime>,
    transcript_stabilizer: Arc<parking_lot::Mutex<TranscriptStabilizer>>,
    latest_partial: Arc<parking_lot::Mutex<Option<CachedPartial>>>,
    ptt_generation: Arc<AtomicU64>,
    ptt_active: Arc<AtomicBool>,
    event_tx: mpsc::Sender<IntegrationEvent>,
    event_rx: Option<mpsc::Receiver<IntegrationEvent>>,
    initialized: bool,
    start_time: Instant,
}

#[derive(Debug, Clone)]
struct CachedPartial {
    generation: u64,
    samples_covered: usize,
    transcript: Transcript,
}

/// Integration events for coordination
#[derive(Debug, Clone)]
pub enum IntegrationEvent {
    /// System initialization started
    InitializationStarted,
    /// Component initialized successfully
    ComponentInitialized { component: String },
    /// Component initialization failed
    ComponentFailed { component: String, error: String },
    /// Audio frame captured
    AudioCaptured {
        frame_size: usize,
        timestamp: Instant,
    },
    /// Raw frame delivered by the live microphone callback.
    LiveAudioFrame { samples: Vec<f32> },
    /// Voice activity detected
    VoiceActivityDetected { confidence: f32 },
    /// Wake word detected
    WakeWordDetected { word: String, confidence: f32 },
    /// Push-to-talk activated
    PTTActivated { timestamp: Instant },
    /// Push-to-talk deactivated
    PTTDeactivated { timestamp: Instant },
    /// Transcription started
    TranscriptionStarted { source: TranscriptionSource },
    /// Best-effort text for the active utterance; superseded by newer partials or the final.
    PartialTranscription {
        text: String,
        stable_prefix_bytes: usize,
        confidence: f32,
    },
    /// Transcription completed
    TranscriptionCompleted { result: TranscriptionResult },
    /// Transcription failed
    TranscriptionFailed { error: String },
    /// System error occurred
    SystemError { error: VoiceStandError },
    /// System shutdown initiated
    ShutdownInitiated,
}

/// Source of transcription request
#[derive(Debug, Clone)]
pub enum TranscriptionSource {
    /// NPU hardware acceleration
    NPU,
    /// CPU fallback processing
    CPU,
    /// Hybrid NPU + CPU processing
    Hybrid,
}

/// Integration statistics
#[derive(Debug, Clone)]
pub struct IntegrationStats {
    /// Total transcriptions processed
    pub total_transcriptions: u64,
    /// NPU transcriptions
    pub npu_transcriptions: u64,
    /// CPU transcriptions (fallback)
    pub cpu_transcriptions: u64,
    /// Failed transcriptions
    pub failed_transcriptions: u64,
    /// Average transcription latency
    pub average_latency_ms: f32,
    /// Peak transcription latency
    pub peak_latency_ms: f32,
    /// System uptime
    pub uptime: Duration,
    /// Components active
    pub components_active: u32,
    /// Components failed
    pub components_failed: u32,
}

impl Default for IntegrationStats {
    fn default() -> Self {
        Self {
            total_transcriptions: 0,
            npu_transcriptions: 0,
            cpu_transcriptions: 0,
            failed_transcriptions: 0,
            average_latency_ms: 0.0,
            peak_latency_ms: 0.0,
            uptime: Duration::ZERO,
            components_active: 0,
            components_failed: 0,
        }
    }
}

impl IntegrationStats {
    /// Get success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_transcriptions == 0 {
            return 1.0;
        }

        let successful = self.total_transcriptions - self.failed_transcriptions;
        successful as f32 / self.total_transcriptions as f32
    }

    /// Get NPU usage rate
    pub fn npu_usage_rate(&self) -> f32 {
        if self.total_transcriptions == 0 {
            return 0.0;
        }

        self.npu_transcriptions as f32 / self.total_transcriptions as f32
    }

    /// Check if integration is healthy
    pub fn is_healthy(&self) -> bool {
        self.success_rate() >= 0.95 // 95% success rate
            && self.average_latency_ms <= 10.0 // <10ms average latency
            && self.components_active > 0 // At least one component active
    }

    /// Update transcription statistics
    pub fn update_transcription(
        &mut self,
        source: &TranscriptionSource,
        latency: Duration,
        success: bool,
    ) {
        self.total_transcriptions += 1;

        match source {
            TranscriptionSource::NPU | TranscriptionSource::Hybrid => {
                self.npu_transcriptions += 1;
            }
            TranscriptionSource::CPU => {
                self.cpu_transcriptions += 1;
            }
        }

        if !success {
            self.failed_transcriptions += 1;
        }

        let latency_ms = latency.as_secs_f32() * 1000.0;

        if self.total_transcriptions == 1 {
            self.average_latency_ms = latency_ms;
        } else {
            // Exponential moving average
            self.average_latency_ms = 0.9 * self.average_latency_ms + 0.1 * latency_ms;
        }

        if latency_ms > self.peak_latency_ms {
            self.peak_latency_ms = latency_ms;
        }
    }

    /// Generate comprehensive report
    pub fn generate_report(&self) -> String {
        format!(
            "=== VoiceStand Integration Statistics ===\n\
             Uptime: {:.1}s\n\
             Total Transcriptions: {}\n\
             NPU Transcriptions: {} ({:.1}%)\n\
             CPU Transcriptions: {} ({:.1}%)\n\
             Failed Transcriptions: {} ({:.1}%)\n\
             Success Rate: {:.1}%\n\
             Average Latency: {:.2}ms\n\
             Peak Latency: {:.2}ms\n\
             Components Active: {}\n\
             Components Failed: {}\n\
             Health Status: {}\n",
            self.uptime.as_secs_f32(),
            self.total_transcriptions,
            self.npu_transcriptions,
            self.npu_usage_rate() * 100.0,
            self.cpu_transcriptions,
            (1.0 - self.npu_usage_rate()) * 100.0,
            self.failed_transcriptions,
            (1.0 - self.success_rate()) * 100.0,
            self.success_rate() * 100.0,
            self.average_latency_ms,
            self.peak_latency_ms,
            self.components_active,
            self.components_failed,
            if self.is_healthy() {
                "✅ HEALTHY"
            } else {
                "⚠️ ISSUES DETECTED"
            }
        )
    }
}

impl VoiceStandIntegration {
    /// Create new integration manager
    pub fn new(config: VoiceStandConfig) -> Result<Self> {
        let (event_tx, event_rx) = mpsc::channel(1000);

        Ok(Self {
            config,
            hardware_manager: None,
            audio_pipeline: None,
            audio_capture: None,
            utterance_assembler: None,
            state_coordinator: None,
            asr_runtime: None,
            transcript_stabilizer: Arc::new(parking_lot::Mutex::new(
                TranscriptStabilizer::default(),
            )),
            latest_partial: Arc::new(parking_lot::Mutex::new(None)),
            ptt_generation: Arc::new(AtomicU64::new(0)),
            ptt_active: Arc::new(AtomicBool::new(false)),
            event_tx,
            event_rx: Some(event_rx),
            initialized: false,
            start_time: Instant::now(),
        })
    }

    /// Initialize all subsystems with comprehensive error handling
    pub async fn initialize(&mut self) -> Result<()> {
        info!("🚀 Initializing VoiceStand integration system");

        self.send_event(IntegrationEvent::InitializationStarted)
            .await?;

        let mut components_initialized = 0u32;
        let mut components_failed = 0u32;

        // Real CPU ASR is a required production component.
        match self.initialize_asr() {
            Ok(()) => {
                self.send_event(IntegrationEvent::ComponentInitialized {
                    component: "CPU ASR".to_string(),
                })
                .await?;
                components_initialized += 1;
            }
            Err(e) => {
                error!("CPU ASR initialization failed: {}", e);
                self.send_event(IntegrationEvent::ComponentFailed {
                    component: "CPU ASR".to_string(),
                    error: e.to_string(),
                })
                .await?;
                return Err(e);
            }
        }

        // Initialize hardware manager (NPU/GNA)
        match self.initialize_hardware().await {
            Ok(()) => {
                self.send_event(IntegrationEvent::ComponentInitialized {
                    component: "Hardware".to_string(),
                })
                .await?;
                components_initialized += 1;
            }
            Err(e) => {
                warn!(
                    "Hardware initialization failed: {} - continuing with CPU fallback",
                    e
                );
                self.send_event(IntegrationEvent::ComponentFailed {
                    component: "Hardware".to_string(),
                    error: e.to_string(),
                })
                .await?;
                components_failed += 1;
            }
        }

        // Initialize audio pipeline
        match self.initialize_audio().await {
            Ok(()) => {
                self.send_event(IntegrationEvent::ComponentInitialized {
                    component: "Audio".to_string(),
                })
                .await?;
                components_initialized += 1;
            }
            Err(e) => {
                error!("Audio initialization failed: {}", e);
                self.send_event(IntegrationEvent::ComponentFailed {
                    component: "Audio".to_string(),
                    error: e.to_string(),
                })
                .await?;
                return Err(e); // Audio is critical
            }
        }

        // Initialize state coordinator
        match self.initialize_state().await {
            Ok(()) => {
                self.send_event(IntegrationEvent::ComponentInitialized {
                    component: "State".to_string(),
                })
                .await?;
                components_initialized += 1;
            }
            Err(e) => {
                error!("State coordinator initialization failed: {}", e);
                self.send_event(IntegrationEvent::ComponentFailed {
                    component: "State".to_string(),
                    error: e.to_string(),
                })
                .await?;
                return Err(e); // State management is critical
            }
        }

        // Check if we have minimum viable system
        if components_initialized == 0 {
            return Err(VoiceStandError::initialization(
                "No components initialized successfully",
            ));
        }

        self.initialized = true;

        info!(
            "✅ VoiceStand integration initialized: {} components active, {} failed",
            components_initialized, components_failed
        );

        Ok(())
    }

    fn initialize_asr(&mut self) -> Result<()> {
        let runtime = AsrRuntime::from_config(&self.config.speech)?;
        info!("CPU ASR ready with {}", runtime.backend_name());
        self.asr_runtime = Some(runtime);
        Ok(())
    }

    /// Initialize hardware subsystem (NPU/GNA)
    async fn initialize_hardware(&mut self) -> Result<()> {
        info!("Initializing hardware subsystem");

        let mut hardware_manager = voicestand_hardware::HardwareManager::new();
        hardware_manager
            .initialize()
            .await
            .map_err(|e| VoiceStandError::hardware(format!("Hardware init failed: {}", e)))?;

        self.hardware_manager = Some(Arc::new(RwLock::new(hardware_manager)));

        info!("✅ Hardware subsystem initialized");
        Ok(())
    }

    /// Initialize audio pipeline
    async fn initialize_audio(&mut self) -> Result<()> {
        info!("Initializing audio pipeline");

        let pipeline_config = voicestand_audio::PipelineConfig {
            sample_rate: self.config.audio.sample_rate,
            channels: self.config.audio.channels as u16,
            frames_per_buffer: self.config.audio.buffer_size as u32,
            max_latency_ms: 10.0,
            enable_vad: true,
            enable_noise_reduction: false,
            vad_threshold: self.config.audio.vad_threshold,
            noise_gate_threshold: 0.01,
        };

        let (audio_pipeline, _event_rx) = voicestand_audio::AudioPipeline::new(pipeline_config)
            .map_err(|e| {
                VoiceStandError::audio(format!("Audio pipeline creation failed: {}", e))
            })?;

        self.audio_pipeline = Some(Arc::new(RwLock::new(audio_pipeline)));
        self.utterance_assembler = Some(Arc::new(parking_lot::Mutex::new(
            voicestand_audio::UtteranceAssembler::new(self.config.audio.sample_rate, 300, 30_000),
        )));

        info!("✅ Audio pipeline initialized");
        Ok(())
    }

    /// Initialize state coordinator
    async fn initialize_state(&mut self) -> Result<()> {
        info!("Initializing state coordinator");

        let state_config = voicestand_state::StateConfig {
            hotkey_config: parse_hotkey_binding(&self.config.hotkeys.push_to_talk)?,
            toggle_hotkey_config: Some(parse_hotkey_binding(
                &self.config.hotkeys.toggle_recording,
            )?),
            target_latency_ms: 10.0,
            enable_fallbacks: true,
            ..Default::default()
        };

        let mut state_coordinator = voicestand_state::VoiceStandCoordinator::new(state_config)
            .map_err(|e| {
                VoiceStandError::state(format!("State coordinator creation failed: {}", e))
            })?;

        state_coordinator
            .initialize()
            .await
            .map_err(|e| VoiceStandError::state(format!("State coordinator init failed: {}", e)))?;

        self.state_coordinator = Some(Arc::new(RwLock::new(state_coordinator)));

        info!("✅ State coordinator initialized");
        Ok(())
    }

    /// Start the integration system
    pub async fn start(&mut self) -> Result<mpsc::Receiver<IntegrationEvent>> {
        if !self.initialized {
            return Err(VoiceStandError::state("System not initialized"));
        }

        info!("🎤 Starting VoiceStand integration system");

        // Take event receiver
        let event_rx = self
            .event_rx
            .take()
            .ok_or_else(|| VoiceStandError::state("Event receiver already taken"))?;

        if let Err(error) = self.start_live_capture() {
            warn!("Live audio capture unavailable: {}", error);
            self.send_event(IntegrationEvent::ComponentFailed {
                component: "Live Audio Capture".to_string(),
                error: error.to_string(),
            })
            .await?;
        }

        // Start state coordinator if available
        if let Some(state_coordinator) = &self.state_coordinator {
            let coordinator = state_coordinator.clone();
            let event_tx = self.event_tx.clone();

            tokio::spawn(async move {
                let mut coordinator_guard = coordinator.write().await;
                if let Ok(mut state_events) = coordinator_guard.start().await {
                    while let Some(state_event) = state_events.recv().await {
                        match state_event {
                            voicestand_state::SystemEvent::PTT(ptt_event) => match ptt_event {
                                voicestand_state::PttEvent::Pressed { timestamp } => {
                                    let _ = event_tx
                                        .send(IntegrationEvent::PTTActivated { timestamp })
                                        .await;
                                }
                                voicestand_state::PttEvent::Released { timestamp } => {
                                    let _ = event_tx
                                        .send(IntegrationEvent::PTTDeactivated { timestamp })
                                        .await;
                                }
                                voicestand_state::PttEvent::ToggleOn { timestamp } => {
                                    let _ = event_tx
                                        .send(IntegrationEvent::PTTActivated { timestamp })
                                        .await;
                                }
                                voicestand_state::PttEvent::ToggleOff { timestamp } => {
                                    let _ = event_tx
                                        .send(IntegrationEvent::PTTDeactivated { timestamp })
                                        .await;
                                }
                                _ => {}
                            },
                            voicestand_state::SystemEvent::Activation(activation_event) => {
                                match activation_event {
                                    voicestand_state::ActivationEvent::WakeWordDetected => {
                                        let _ = event_tx
                                            .send(IntegrationEvent::WakeWordDetected {
                                                word: "wake-word".to_string(),
                                                confidence: 0.9,
                                            })
                                            .await;
                                    }
                                    _ => {}
                                }
                            }
                            voicestand_state::SystemEvent::Error { error } => {
                                let _ = event_tx
                                    .send(IntegrationEvent::SystemError {
                                        error: VoiceStandError::state(error.to_string()),
                                    })
                                    .await;
                            }
                            _ => {}
                        }
                    }
                }
            });
        }

        info!("🚀 VoiceStand integration system started successfully");

        // Create public event channel
        let (public_tx, public_rx) = mpsc::channel(100);

        // Start main integration loop
        let start_time = self.start_time;

        tokio::spawn(async move {
            Self::integration_loop(event_rx, public_tx, start_time).await;
        });

        Ok(public_rx)
    }

    fn start_live_capture(&mut self) -> Result<()> {
        let (sender, receiver) = crossbeam_channel::unbounded();
        let mut capture = voicestand_audio::AudioCapture::new(self.config.audio.clone(), sender)?;
        capture.initialize(self.config.audio.device_name.clone())?;
        capture.start()?;

        let event_tx = self.event_tx.clone();
        std::thread::Builder::new()
            .name("voicestand-audio-events".to_string())
            .spawn(move || {
                while let Ok(event) = receiver.recv() {
                    match event {
                        voicestand_types::AppEvent::AudioDataReceived(audio) => {
                            if event_tx
                                .blocking_send(IntegrationEvent::LiveAudioFrame {
                                    samples: audio.samples,
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        voicestand_types::AppEvent::Error(error) => {
                            if event_tx
                                .blocking_send(IntegrationEvent::SystemError {
                                    error: VoiceStandError::audio(error),
                                })
                                .is_err()
                            {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            })
            .map_err(|error| {
                VoiceStandError::audio(format!("Audio event bridge failed: {error}"))
            })?;

        self.audio_capture = Some(capture);
        Ok(())
    }

    /// Main integration event loop
    async fn integration_loop(
        mut event_rx: mpsc::Receiver<IntegrationEvent>,
        public_tx: mpsc::Sender<IntegrationEvent>,
        start_time: Instant,
    ) {
        info!("Starting VoiceStand integration loop");

        let mut stats = IntegrationStats::default();

        while let Some(event) = event_rx.recv().await {
            // Update statistics
            stats.uptime = start_time.elapsed();

            // Handle integration events
            match &event {
                IntegrationEvent::ComponentInitialized { .. } => {
                    stats.components_active += 1;
                }
                IntegrationEvent::ComponentFailed { .. } => {
                    stats.components_failed += 1;
                }
                IntegrationEvent::PTTActivated { .. } => {
                    info!("🔴 Push-to-talk activated");
                    // Transcription will be handled by the application layer
                    // which has access to the hardware_manager
                }
                IntegrationEvent::PTTDeactivated { .. } => {
                    info!("⚪ Push-to-talk deactivated");
                    // Would stop transcription here
                }
                IntegrationEvent::WakeWordDetected { word, confidence } => {
                    info!(
                        "🔊 Wake word detected: '{}' ({:.1}%)",
                        word,
                        confidence * 100.0
                    );
                    // Would start transcription here
                }
                IntegrationEvent::TranscriptionCompleted { result } => {
                    info!(
                        "✅ Transcription: \"{}\" ({:.1}% confidence)",
                        result.text,
                        result.confidence * 100.0
                    );

                    // Update statistics
                    let source = if result.meets_latency_target {
                        TranscriptionSource::NPU
                    } else {
                        TranscriptionSource::CPU
                    };

                    stats.update_transcription(
                        &source,
                        Duration::from_millis(result.duration_ms as u64),
                        true,
                    );
                }
                IntegrationEvent::TranscriptionFailed { error } => {
                    warn!("❌ Transcription failed: {}", error);
                    stats.update_transcription(
                        &TranscriptionSource::CPU,
                        Duration::from_millis(1000), // Assume 1s for failed transcription
                        false,
                    );
                }
                IntegrationEvent::SystemError { error } => {
                    error!("System error: {}", error);
                }
                _ => {}
            }

            // Forward event to public channel
            if public_tx.send(event).await.is_err() {
                break; // Receiver dropped
            }
        }

        info!("VoiceStand integration loop ended");
        info!("Final statistics:\n{}", stats.generate_report());
    }

    /// Send integration event
    async fn send_event(&self, event: IntegrationEvent) -> Result<()> {
        self.event_tx
            .send(event)
            .await
            .map_err(|_| VoiceStandError::state("Event channel closed"))
    }

    /// Get system status
    pub async fn get_status(&self) -> Result<SystemStatus> {
        if !self.initialized {
            return Ok(SystemStatus {
                state: SystemState::NotInitialized,
                components_active: 0,
                components_failed: 0,
                uptime: Duration::ZERO,
                capabilities: Vec::new(),
            });
        }

        let mut status = SystemStatus {
            state: SystemState::Ready,
            components_active: 0,
            components_failed: 0,
            uptime: self.start_time.elapsed(),
            capabilities: Vec::new(),
        };

        if let Some(runtime) = &self.asr_runtime {
            status.components_active += 1;
            status
                .capabilities
                .push(format!("CPU Voice-to-Text ({})", runtime.backend_name()));
        }

        // Check hardware status
        if let Some(hardware_manager) = &self.hardware_manager {
            if let Ok(hw_guard) = hardware_manager.try_read() {
                if let Ok(health) = hw_guard.check_health().await {
                    status.components_active += 1;

                    if health.can_transcribe() {
                        if health.npu_healthy {
                            status.capabilities.push("NPU Voice-to-Text".to_string());
                        } else {
                            status.capabilities.push("CPU Voice-to-Text".to_string());
                        }
                    }

                    if health.can_wake_word_detect() {
                        status.capabilities.push("GNA Wake Words".to_string());
                    }

                    if health.can_push_to_talk() {
                        status.capabilities.push("Push-to-Talk".to_string());
                    }
                }
            }
        }

        // Check audio status
        if self.audio_pipeline.is_some() {
            status.components_active += 1;
            status.capabilities.push("Audio Capture".to_string());
        }

        // Check state coordinator status
        if let Some(state_coordinator) = &self.state_coordinator {
            if let Ok(coord_guard) = state_coordinator.try_read() {
                if coord_guard.is_healthy() {
                    status.components_active += 1;
                    status.capabilities.push("State Management".to_string());
                }
            }
        }

        Ok(status)
    }

    /// Process voice command (simplified version)
    pub async fn process_voice_command(
        &self,
        audio_data: &[f32],
    ) -> Result<Option<TranscriptionResult>> {
        if !self.initialized {
            return Err(VoiceStandError::state("System not initialized"));
        }

        let start_time = Instant::now();
        self.transcript_stabilizer.lock().reset();

        let runtime = self
            .asr_runtime
            .as_ref()
            .ok_or_else(|| VoiceStandError::speech("CPU ASR is not initialized"))?;
        let transcript = match runtime.transcribe(audio_data.to_vec()).await {
            Ok(transcript) => transcript,
            Err(error) => {
                self.send_event(IntegrationEvent::TranscriptionFailed {
                    error: error.to_string(),
                })
                .await?;
                return Err(error);
            }
        };
        let duration = start_time.elapsed();
        let result = TranscriptionResult {
            text: transcript.text,
            confidence: transcript.confidence.unwrap_or(0.0),
            language: self.config.speech.language.clone(),
            duration_ms: duration.as_millis() as u32,
            meets_latency_target: duration.as_millis() <= 700,
        };

        self.send_event(IntegrationEvent::TranscriptionCompleted {
            result: result.clone(),
        })
        .await?;

        Ok(Some(result))
    }

    async fn publish_cached_final(
        &self,
        samples_covered: usize,
    ) -> Result<Option<TranscriptionResult>> {
        let generation = self.ptt_generation.load(Ordering::Acquire);
        let cached = self.latest_partial.lock().clone();
        let Some(cached) = cached.filter(|cached| {
            cached.generation == generation && cached.samples_covered == samples_covered
        }) else {
            return Ok(None);
        };

        let result = TranscriptionResult {
            text: cached.transcript.text,
            confidence: cached.transcript.confidence.unwrap_or(0.0),
            language: self.config.speech.language.clone(),
            duration_ms: 0,
            meets_latency_target: true,
        };
        self.transcript_stabilizer.lock().reset();
        self.send_event(IntegrationEvent::TranscriptionCompleted {
            result: result.clone(),
        })
        .await?;
        Ok(Some(result))
    }

    async fn finalize_from_tail(&self, audio: &[f32]) -> Result<Option<TranscriptionResult>> {
        const OVERLAP_SAMPLES: usize = 16_000;
        const MAX_UNCOVERED_SAMPLES: usize = 32_000;

        let generation = self.ptt_generation.load(Ordering::Acquire);
        let cached = self.latest_partial.lock().clone();
        let Some(cached) = cached.filter(|cached| {
            cached.generation == generation
                && cached.samples_covered >= OVERLAP_SAMPLES
                && cached.samples_covered < audio.len()
                && audio.len() - cached.samples_covered <= MAX_UNCOVERED_SAMPLES
        }) else {
            return Ok(None);
        };

        let tail_start = cached.samples_covered - OVERLAP_SAMPLES;
        let runtime = self
            .asr_runtime
            .as_ref()
            .ok_or_else(|| VoiceStandError::speech("CPU ASR is not initialized"))?;
        let started = Instant::now();
        let tail = match runtime
            .transcribe_latest(audio[tail_start..].to_vec())
            .await
        {
            Ok(transcript) => transcript,
            Err(_) => return Ok(None),
        };
        let Some(text) = merge_word_overlap(&cached.transcript.text, &tail.text) else {
            return Ok(None);
        };
        let duration = started.elapsed();
        let result = TranscriptionResult {
            text,
            confidence: cached
                .transcript
                .confidence
                .unwrap_or(0.0)
                .min(tail.confidence.unwrap_or(0.0)),
            language: self.config.speech.language.clone(),
            duration_ms: duration.as_millis() as u32,
            meets_latency_target: duration.as_millis() <= 700,
        };
        self.transcript_stabilizer.lock().reset();
        self.send_event(IntegrationEvent::TranscriptionCompleted {
            result: result.clone(),
        })
        .await?;
        Ok(Some(result))
    }

    /// Process one live microphone frame and transcribe once VAD closes an utterance.
    pub async fn process_audio_frame(
        &self,
        audio_data: &[f32],
    ) -> Result<Option<TranscriptionResult>> {
        if self.config.audio.sample_rate != 16_000 || self.config.audio.channels != 1 {
            return Err(VoiceStandError::audio(
                "Live ASR currently requires 16 kHz mono audio",
            ));
        }

        let pipeline = self
            .audio_pipeline
            .as_ref()
            .ok_or_else(|| VoiceStandError::audio("Audio pipeline is not initialized"))?;
        let (processed, _stats, vad_result) = pipeline
            .write()
            .await
            .process_with_stats(audio_data)
            .map_err(|error| VoiceStandError::audio(error.to_string()))?;

        self.send_event(IntegrationEvent::AudioCaptured {
            frame_size: processed.len(),
            timestamp: Instant::now(),
        })
        .await?;

        if let Some(result) = vad_result.filter(|result| result.has_voice) {
            self.send_event(IntegrationEvent::VoiceActivityDetected {
                confidence: result.confidence,
            })
            .await?;
        }

        let assembler = self
            .utterance_assembler
            .as_ref()
            .ok_or_else(|| VoiceStandError::audio("Utterance assembler is not initialized"))?;
        let (utterance, partial) = {
            let mut assembler = assembler.lock();
            let utterance = assembler.push(&processed, self.ptt_active.load(Ordering::Acquire));
            let partial = assembler.partial_if_due(16_000, 16_000);
            (utterance, partial)
        };

        if let Some(samples) = partial {
            let runtime = self
                .asr_runtime
                .as_ref()
                .ok_or_else(|| VoiceStandError::speech("CPU ASR is not initialized"))?
                .clone();
            let event_tx = self.event_tx.clone();
            let stabilizer = Arc::clone(&self.transcript_stabilizer);
            let latest_partial = Arc::clone(&self.latest_partial);
            let generation_counter = Arc::clone(&self.ptt_generation);
            let generation = generation_counter.load(Ordering::Acquire);
            let samples_covered = samples.len();
            tokio::spawn(async move {
                if let Ok(transcript) = runtime.transcribe_latest(samples).await {
                    let confidence = transcript.confidence.unwrap_or(0.0);
                    if confidence < 0.35 {
                        return;
                    }
                    if generation_counter.load(Ordering::Acquire) != generation {
                        return;
                    }
                    *latest_partial.lock() = Some(CachedPartial {
                        generation,
                        samples_covered,
                        transcript: transcript.clone(),
                    });
                    let Some(update) = stabilizer.lock().update(&transcript.text) else {
                        return;
                    };
                    let _ = event_tx
                        .send(IntegrationEvent::PartialTranscription {
                            text: update.text,
                            stable_prefix_bytes: update.stable_prefix_bytes,
                            confidence,
                        })
                        .await;
                }
            });
        }

        match utterance {
            Some(samples) => self.process_voice_command(&samples).await,
            None => Ok(None),
        }
    }

    /// Begin a hold-to-talk session. Repeated key-down events are ignored.
    pub fn begin_ptt(&self) -> Result<()> {
        if self.ptt_active.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        self.ptt_generation.fetch_add(1, Ordering::AcqRel);
        self.latest_partial.lock().take();
        self.transcript_stabilizer.lock().reset();
        self.utterance_assembler
            .as_ref()
            .ok_or_else(|| VoiceStandError::audio("Utterance assembler is not initialized"))?
            .lock()
            .begin();
        Ok(())
    }

    /// End hold-to-talk and force immediate final decoding.
    pub async fn end_ptt(&self) -> Result<Option<TranscriptionResult>> {
        if !self.ptt_active.swap(false, Ordering::AcqRel) {
            return Ok(None);
        }
        let samples = self
            .utterance_assembler
            .as_ref()
            .ok_or_else(|| VoiceStandError::audio("Utterance assembler is not initialized"))?
            .lock()
            .end();
        match samples {
            Some(samples) if !samples.is_empty() => {
                if let Some(result) = self.publish_cached_final(samples.len()).await? {
                    Ok(Some(result))
                } else if let Some(result) = self.finalize_from_tail(&samples).await? {
                    Ok(Some(result))
                } else {
                    self.process_voice_command(&samples).await
                }
            }
            _ => Ok(None),
        }
    }

    pub fn asr_metrics(&self) -> Option<AsrMetrics> {
        self.asr_runtime.as_ref().map(AsrRuntime::metrics)
    }

    /// Shutdown the integration system
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("🛑 Shutting down VoiceStand integration system");

        self.send_event(IntegrationEvent::ShutdownInitiated).await?;

        // Shutdown state coordinator
        if let Some(state_coordinator) = &self.state_coordinator {
            let mut coord_guard = state_coordinator.write().await;
            if let Err(e) = coord_guard.shutdown().await {
                warn!("State coordinator shutdown error: {}", e);
            }
        }

        if let Some(capture) = &mut self.audio_capture {
            capture.stop()?;
        }
        self.audio_capture = None;

        // Shutdown audio pipeline (will be dropped automatically)
        if self.audio_pipeline.is_some() {
            debug!("Audio pipeline will be shutdown automatically on drop");
        }

        // Shutdown hardware manager
        if let Some(hardware_manager) = &self.hardware_manager {
            let mut hw_guard = hardware_manager.write().await;
            if let Err(e) = hw_guard.shutdown().await {
                warn!("Hardware manager shutdown error: {}", e);
            }
        }

        self.initialized = false;

        info!("✅ VoiceStand integration system shutdown complete");
        Ok(())
    }

    /// Check if integration is healthy
    pub fn is_healthy(&self) -> bool {
        self.initialized
            && self.asr_runtime.is_some()
            && self.audio_pipeline.is_some()
            && self.state_coordinator.is_some()
    }
}

fn parse_hotkey_binding(binding: &str) -> Result<voicestand_state::HotkeyConfig> {
    let mut parts = binding
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let key = parts
        .pop()
        .ok_or_else(|| VoiceStandError::config("Push-to-talk binding is empty"))?;
    Ok(voicestand_state::HotkeyConfig {
        modifiers: parts.into_iter().map(str::to_string).collect(),
        key: key.to_string(),
    })
}

impl Drop for VoiceStandIntegration {
    fn drop(&mut self) {
        if self.initialized {
            warn!("VoiceStandIntegration dropped while initialized - should call shutdown() explicitly");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::config::VoiceStandConfig;
    use voicestand_asr::{DecodeOptions, ModelSpec, SpeechBackend, Transcript};

    struct IntegrationMockBackend;

    impl SpeechBackend for IntegrationMockBackend {
        fn name(&self) -> &'static str {
            "integration-mock"
        }

        fn load(&mut self, _model: &ModelSpec) -> Result<()> {
            Ok(())
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn transcribe(&mut self, audio: &[f32], _options: &DecodeOptions) -> Result<Transcript> {
            Transcript::final_result(
                "real backend result",
                Some(0.91),
                Duration::from_secs_f64(audio.len() as f64 / 16_000.0),
            )
        }
    }

    #[test]
    fn test_integration_stats_default() {
        let stats = IntegrationStats::default();
        assert_eq!(stats.total_transcriptions, 0);
        assert_eq!(stats.success_rate(), 1.0);
        assert_eq!(stats.npu_usage_rate(), 0.0);
        assert!(!stats.is_healthy());
    }

    #[tokio::test]
    async fn process_voice_command_uses_configured_asr_runtime() {
        let mut integration =
            VoiceStandIntegration::new(VoiceStandConfig::default()).expect("create integration");
        integration.asr_runtime = Some(
            AsrRuntime::from_loaded_backend(
                Box::new(IntegrationMockBackend),
                DecodeOptions::default(),
            )
            .expect("create ASR runtime"),
        );
        integration.initialized = true;

        let result = integration
            .process_voice_command(&[0.0; 1_600])
            .await
            .expect("process audio")
            .expect("transcription result");

        assert_eq!(result.text, "real backend result");
        assert_eq!(result.confidence, 0.91);

        integration.initialized = false;
    }

    #[tokio::test]
    async fn ptt_release_reuses_an_exact_current_partial() {
        let mut integration =
            VoiceStandIntegration::new(VoiceStandConfig::default()).expect("create integration");
        integration.asr_runtime = Some(
            AsrRuntime::from_loaded_backend(
                Box::new(IntegrationMockBackend),
                DecodeOptions::default(),
            )
            .expect("create ASR runtime"),
        );
        integration.utterance_assembler = Some(Arc::new(parking_lot::Mutex::new(
            voicestand_audio::UtteranceAssembler::new(16_000, 300, 30_000),
        )));
        integration.initialized = true;
        integration.begin_ptt().expect("begin PTT");
        let samples = vec![0.1; 320];
        integration
            .utterance_assembler
            .as_ref()
            .expect("assembler")
            .lock()
            .push(&samples, true);
        *integration.latest_partial.lock() = Some(CachedPartial {
            generation: integration.ptt_generation.load(Ordering::Acquire),
            samples_covered: samples.len(),
            transcript: Transcript::final_result(
                "cached partial",
                Some(0.92),
                Duration::from_millis(20),
            )
            .expect("transcript"),
        });

        let result = integration
            .end_ptt()
            .await
            .expect("end PTT")
            .expect("final transcript");

        assert_eq!(result.text, "cached partial");
        assert_eq!(result.duration_ms, 0);
        assert_eq!(integration.asr_metrics().expect("metrics").started, 0);
    }

    #[tokio::test]
    async fn ptt_release_rejects_a_partial_missing_the_audio_tail() {
        let mut integration =
            VoiceStandIntegration::new(VoiceStandConfig::default()).expect("create integration");
        integration.asr_runtime = Some(
            AsrRuntime::from_loaded_backend(
                Box::new(IntegrationMockBackend),
                DecodeOptions::default(),
            )
            .expect("create ASR runtime"),
        );
        integration.utterance_assembler = Some(Arc::new(parking_lot::Mutex::new(
            voicestand_audio::UtteranceAssembler::new(16_000, 300, 30_000),
        )));
        integration.initialized = true;
        integration.begin_ptt().expect("begin PTT");
        let samples = vec![0.1; 320];
        integration
            .utterance_assembler
            .as_ref()
            .expect("assembler")
            .lock()
            .push(&samples, true);
        *integration.latest_partial.lock() = Some(CachedPartial {
            generation: integration.ptt_generation.load(Ordering::Acquire),
            samples_covered: samples.len() - 1,
            transcript: Transcript::final_result(
                "stale partial",
                Some(0.92),
                Duration::from_millis(20),
            )
            .expect("transcript"),
        });

        let result = integration
            .end_ptt()
            .await
            .expect("end PTT")
            .expect("final transcript");

        assert_eq!(result.text, "real backend result");
        assert_eq!(integration.asr_metrics().expect("metrics").started, 1);
    }

    #[tokio::test]
    async fn ptt_release_decodes_only_a_bounded_overlapping_tail() {
        let mut integration =
            VoiceStandIntegration::new(VoiceStandConfig::default()).expect("create integration");
        integration.asr_runtime = Some(
            AsrRuntime::from_loaded_backend(
                Box::new(IntegrationMockBackend),
                DecodeOptions::default(),
            )
            .expect("create ASR runtime"),
        );
        integration.utterance_assembler = Some(Arc::new(parking_lot::Mutex::new(
            voicestand_audio::UtteranceAssembler::new(16_000, 300, 30_000),
        )));
        integration.initialized = true;
        integration.begin_ptt().expect("begin PTT");
        let samples = vec![0.1; 17_600];
        integration
            .utterance_assembler
            .as_ref()
            .expect("assembler")
            .lock()
            .push(&samples, true);
        *integration.latest_partial.lock() = Some(CachedPartial {
            generation: integration.ptt_generation.load(Ordering::Acquire),
            samples_covered: 16_000,
            transcript: Transcript::final_result(
                "prefix real backend result",
                Some(0.92),
                Duration::from_secs(1),
            )
            .expect("transcript"),
        });

        let result = integration
            .end_ptt()
            .await
            .expect("end PTT")
            .expect("final transcript");

        assert_eq!(result.text, "prefix real backend result");
        let metrics = integration.asr_metrics().expect("metrics");
        assert_eq!(metrics.started, 1);
        assert_eq!(metrics.completed, 1);
    }

    #[tokio::test]
    async fn five_hundred_ptt_sessions_finalize_without_stuck_state() {
        let mut integration =
            VoiceStandIntegration::new(VoiceStandConfig::default()).expect("create integration");
        integration.asr_runtime = Some(
            AsrRuntime::from_loaded_backend(
                Box::new(IntegrationMockBackend),
                DecodeOptions::default(),
            )
            .expect("create ASR runtime"),
        );
        integration.utterance_assembler = Some(Arc::new(parking_lot::Mutex::new(
            voicestand_audio::UtteranceAssembler::new(16_000, 300, 30_000),
        )));
        integration.initialized = true;
        let mut events = integration.event_rx.take().expect("event receiver");
        let drain = tokio::spawn(async move { while events.recv().await.is_some() {} });

        for _ in 0..500 {
            integration.begin_ptt().expect("begin PTT");
            integration
                .utterance_assembler
                .as_ref()
                .expect("assembler")
                .lock()
                .push(&[0.1; 320], true);
            let result = integration
                .end_ptt()
                .await
                .expect("end PTT")
                .expect("final transcript");
            assert_eq!(result.text, "real backend result");
        }

        assert!(!integration.ptt_active.load(Ordering::Acquire));
        let metrics = integration.asr_metrics().expect("ASR metrics");
        assert_eq!(metrics.started, 500);
        assert_eq!(metrics.completed, 500);
        assert_eq!(metrics.cancelled, 0);
        integration.initialized = false;
        drop(integration);
        drain.await.expect("event drain");
    }

    #[test]
    fn test_integration_stats_update() {
        let mut stats = IntegrationStats::default();

        stats.update_transcription(&TranscriptionSource::NPU, Duration::from_millis(5), true);
        assert_eq!(stats.total_transcriptions, 1);
        assert_eq!(stats.npu_transcriptions, 1);
        assert_eq!(stats.success_rate(), 1.0);
        assert_eq!(stats.npu_usage_rate(), 1.0);

        stats.update_transcription(&TranscriptionSource::CPU, Duration::from_millis(15), false);
        assert_eq!(stats.total_transcriptions, 2);
        assert_eq!(stats.failed_transcriptions, 1);
        assert_eq!(stats.success_rate(), 0.5);
        assert_eq!(stats.npu_usage_rate(), 0.5);
    }

    #[tokio::test]
    async fn test_integration_creation() {
        let config = VoiceStandConfig::default();
        let integration = VoiceStandIntegration::new(config);
        assert!(integration.is_ok());

        let integration = integration.unwrap();
        assert!(!integration.initialized);
        assert!(integration.hardware_manager.is_none());
        assert!(integration.audio_pipeline.is_none());
        assert!(integration.state_coordinator.is_none());
    }
}
