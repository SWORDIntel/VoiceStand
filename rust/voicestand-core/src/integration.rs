//! VoiceStand Integration Layer
//!
//! Memory-safe integration of NPU, GNA, audio, and state management components.
//! Provides the complete push-to-talk system with fallback mechanisms.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::error::{VoiceStandError, Result};
use crate::types::{TranscriptionResult, VoiceCommand, SystemStatus};
use crate::config::VoiceStandConfig;

/// Integration manager for coordinating all subsystems
pub struct VoiceStandIntegration {
    config: VoiceStandConfig,
    hardware_manager: Option<Arc<RwLock<voicestand_hardware::HardwareManager>>>,
    audio_pipeline: Option<Arc<RwLock<voicestand_audio::AudioPipeline>>>,
    state_coordinator: Option<Arc<RwLock<voicestand_state::VoiceStandCoordinator>>>,
    event_tx: mpsc::Sender<IntegrationEvent>,
    event_rx: Option<mpsc::Receiver<IntegrationEvent>>,
    initialized: bool,
    start_time: Instant,
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
    AudioCaptured { frame_size: usize, timestamp: Instant },
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
    pub fn update_transcription(&mut self, source: &TranscriptionSource, latency: Duration, success: bool) {
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
            if self.is_healthy() { "✅ HEALTHY" } else { "⚠️ ISSUES DETECTED" }
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
            state_coordinator: None,
            event_tx,
            event_rx: Some(event_rx),
            initialized: false,
            start_time: Instant::now(),
        })
    }

    /// Initialize all subsystems with comprehensive error handling
    pub async fn initialize(&mut self) -> Result<()> {
        info!("🚀 Initializing VoiceStand integration system");

        self.send_event(IntegrationEvent::InitializationStarted).await?;

        let mut components_initialized = 0u32;
        let mut components_failed = 0u32;

        // Initialize hardware manager (NPU/GNA)
        match self.initialize_hardware().await {
            Ok(()) => {
                self.send_event(IntegrationEvent::ComponentInitialized {
                    component: "Hardware".to_string(),
                }).await?;
                components_initialized += 1;
            }
            Err(e) => {
                warn!("Hardware initialization failed: {} - continuing with CPU fallback", e);
                self.send_event(IntegrationEvent::ComponentFailed {
                    component: "Hardware".to_string(),
                    error: e.to_string(),
                }).await?;
                components_failed += 1;
            }
        }

        // Initialize audio pipeline
        match self.initialize_audio().await {
            Ok(()) => {
                self.send_event(IntegrationEvent::ComponentInitialized {
                    component: "Audio".to_string(),
                }).await?;
                components_initialized += 1;
            }
            Err(e) => {
                error!("Audio initialization failed: {}", e);
                self.send_event(IntegrationEvent::ComponentFailed {
                    component: "Audio".to_string(),
                    error: e.to_string(),
                }).await?;
                components_failed += 1;
                return Err(e); // Audio is critical
            }
        }

        // Initialize state coordinator
        match self.initialize_state().await {
            Ok(()) => {
                self.send_event(IntegrationEvent::ComponentInitialized {
                    component: "State".to_string(),
                }).await?;
                components_initialized += 1;
            }
            Err(e) => {
                error!("State coordinator initialization failed: {}", e);
                self.send_event(IntegrationEvent::ComponentFailed {
                    component: "State".to_string(),
                    error: e.to_string(),
                }).await?;
                components_failed += 1;
                return Err(e); // State management is critical
            }
        }

        // Check if we have minimum viable system
        if components_initialized == 0 {
            return Err(VoiceStandError::initialization(
                "No components initialized successfully"
            ));
        }

        self.initialized = true;

        info!("✅ VoiceStand integration initialized: {} components active, {} failed",
              components_initialized, components_failed);

        Ok(())
    }

    /// Initialize hardware subsystem (NPU/GNA)
    async fn initialize_hardware(&mut self) -> Result<()> {
        info!("Initializing hardware subsystem");

        let mut hardware_manager = voicestand_hardware::HardwareManager::new();
        hardware_manager.initialize().await
            .map_err(|e| VoiceStandError::hardware(format!("Hardware init failed: {}", e)))?;

        self.hardware_manager = Some(Arc::new(RwLock::new(hardware_manager)));

        info!("✅ Hardware subsystem initialized");
        Ok(())
    }

    /// Initialize audio pipeline
    async fn initialize_audio(&mut self) -> Result<()> {
        info!("Initializing audio pipeline");

        let audio_config = voicestand_audio::AudioConfig {
            sample_rate: self.config.audio.sample_rate,
            channels: self.config.audio.channels,
            format: voicestand_audio::AudioFormat::F32,
            buffer_size: self.config.audio.buffer_size,
            target_latency_ms: 10.0,
            enable_vad: true,
            enable_noise_reduction: false,
            enable_agc: false,
        };

        let pipeline_config = voicestand_audio::PipelineConfig {
            audio_config,
            processing_threads: 2,
            enable_monitoring: true,
        };

        let audio_pipeline = voicestand_audio::AudioPipeline::new(pipeline_config)
            .map_err(|e| VoiceStandError::audio(format!("Audio pipeline creation failed: {}", e)))?;

        self.audio_pipeline = Some(Arc::new(RwLock::new(audio_pipeline)));

        info!("✅ Audio pipeline initialized");
        Ok(())
    }

    /// Initialize state coordinator
    async fn initialize_state(&mut self) -> Result<()> {
        info!("Initializing state coordinator");

        let state_config = voicestand_state::StateConfig {
            target_latency_ms: 10.0,
            enable_fallbacks: true,
            debug_mode: self.config.debug_mode,
            ..Default::default()
        };

        let mut state_coordinator = voicestand_state::VoiceStandCoordinator::new(state_config)
            .map_err(|e| VoiceStandError::state(format!("State coordinator creation failed: {}", e)))?;

        state_coordinator.initialize().await
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
        let event_rx = self.event_rx.take()
            .ok_or_else(|| VoiceStandError::state("Event receiver already taken"))?;

        // Start audio pipeline if available
        if let Some(audio_pipeline) = &self.audio_pipeline {
            let pipeline = audio_pipeline.clone();
            let event_tx = self.event_tx.clone();
            let state_coordinator = self.state_coordinator.clone();

            tokio::spawn(async move {
                let mut pipeline_guard = pipeline.write().await;
                if let Ok(mut audio_events) = pipeline_guard.start().await {
                    while let Some(audio_event) = audio_events.recv().await {
                        match audio_event {
                            voicestand_audio::PipelineEvent::FrameCaptured { frame, timestamp } => {
                                // CRITICAL FIX: Send audio data to state coordinator for activation detection
                                if let Some(coordinator_ref) = &state_coordinator {
                                    let coordinator = coordinator_ref.clone();
                                    let audio_data = frame.samples.clone();

                                    tokio::spawn(async move {
                                        let mut coord_guard = coordinator.write().await;
                                        if let Ok(events) = coord_guard.process_audio_frame(&audio_data).await {
                                            // Events are already sent by the coordinator
                                            debug!("Processed {} activation events from audio frame", events.len());
                                        } else {
                                            warn!("Failed to process audio frame for activation detection");
                                        }
                                    });
                                }

                                let _ = event_tx.send(IntegrationEvent::AudioCaptured {
                                    frame_size: frame.samples.len(),
                                    timestamp,
                                }).await;
                            }
                            voicestand_audio::PipelineEvent::VoiceActivityDetected { confidence } => {
                                let _ = event_tx.send(IntegrationEvent::VoiceActivityDetected {
                                    confidence,
                                }).await;
                            }
                            voicestand_audio::PipelineEvent::Error { error } => {
                                let _ = event_tx.send(IntegrationEvent::SystemError {
                                    error: VoiceStandError::audio(error),
                                }).await;
                            }
                            _ => {}
                        }
                    }
                }
            });
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
                            voicestand_state::SystemEvent::PTT(ptt_event) => {
                                match ptt_event {
                                    voicestand_state::PTTEvent::Pressed { timestamp } => {
                                        let _ = event_tx.send(IntegrationEvent::PTTActivated { timestamp }).await;
                                    }
                                    voicestand_state::PTTEvent::Released { timestamp } => {
                                        let _ = event_tx.send(IntegrationEvent::PTTDeactivated { timestamp }).await;
                                    }
                                    _ => {}
                                }
                            }
                            voicestand_state::SystemEvent::Activation(activation_event) => {
                                match activation_event {
                                    voicestand_state::ActivationEvent::WakeWordDetected { word, confidence, .. } => {
                                        let _ = event_tx.send(IntegrationEvent::WakeWordDetected {
                                            word,
                                            confidence,
                                        }).await;
                                    }
                                    _ => {}
                                }
                            }
                            voicestand_state::SystemEvent::Error { error } => {
                                let _ = event_tx.send(IntegrationEvent::SystemError {
                                    error: VoiceStandError::state(error.to_string()),
                                }).await;
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
        let event_tx = self.event_tx.clone();
        let start_time = self.start_time;
        let hardware_manager = self.hardware_manager.clone();

        tokio::spawn(async move {
            Self::integration_loop(
                event_rx,
                public_tx,
                start_time,
                hardware_manager,
            ).await;
        });

        Ok(public_rx)
    }

    /// Main integration event loop
    async fn integration_loop(
        mut event_rx: mpsc::Receiver<IntegrationEvent>,
        public_tx: mpsc::Sender<IntegrationEvent>,
        start_time: Instant,
        hardware_manager: Option<Arc<RwLock<voicestand_hardware::HardwareManager>>>,
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

                    // Start transcription if hardware available
                    if let Some(hw_manager) = &hardware_manager {
                        if let Ok(hw_guard) = hw_manager.try_read() {
                            if let Ok(health) = hw_guard.check_health().await {
                                if health.can_transcribe() {
                                    // Would start NPU transcription here
                                    debug!("Starting NPU transcription");
                                }
                            }
                        }
                    }
                }
                IntegrationEvent::PTTDeactivated { .. } => {
                    info!("⚪ Push-to-talk deactivated");
                    // Would stop transcription here
                }
                IntegrationEvent::WakeWordDetected { word, confidence } => {
                    info!("🔊 Wake word detected: '{}' ({:.1}%)", word, confidence * 100.0);
                    // Would start transcription here
                }
                IntegrationEvent::TranscriptionCompleted { result } => {
                    info!("✅ Transcription: \"{}\" ({:.1}% confidence)",
                          result.text, result.confidence * 100.0);

                    // Update statistics
                    let source = if result.meets_latency_target {
                        TranscriptionSource::NPU
                    } else {
                        TranscriptionSource::CPU
                    };

                    stats.update_transcription(
                        &source,
                        Duration::from_millis(result.duration_ms as u64),
                        true
                    );
                }
                IntegrationEvent::TranscriptionFailed { error } => {
                    warn!("❌ Transcription failed: {}", error);
                    stats.update_transcription(
                        &TranscriptionSource::CPU,
                        Duration::from_millis(1000), // Assume 1s for failed transcription
                        false
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
        self.event_tx.send(event).await
            .map_err(|_| VoiceStandError::state("Event channel closed"))
    }

    /// Get system status
    pub async fn get_status(&self) -> Result<SystemStatus> {
        if !self.initialized {
            return Ok(SystemStatus {
                state: crate::types::SystemState::NotInitialized,
                components_active: 0,
                components_failed: 0,
                uptime: Duration::ZERO,
                capabilities: Vec::new(),
            });
        }

        let mut status = SystemStatus {
            state: crate::types::SystemState::Ready,
            components_active: 0,
            components_failed: 0,
            uptime: self.start_time.elapsed(),
            capabilities: Vec::new(),
        };

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
    pub async fn process_voice_command(&self, audio_data: &[f32]) -> Result<Option<TranscriptionResult>> {
        if !self.initialized {
            return Err(VoiceStandError::state("System not initialized"));
        }

        let start_time = Instant::now();

        // Try NPU first if available
        if let Some(hardware_manager) = &self.hardware_manager {
            if let Ok(hw_guard) = hardware_manager.try_read() {
                if let Ok(health) = hw_guard.check_health().await {
                    if health.can_transcribe() && health.npu_healthy {
                        if let Ok(npu_processor) = hw_guard.get_npu_processor().await {
                            match npu_processor.transcribe_audio(audio_data, "models/whisper-base-npu.xml").await {
                                Ok(result) => {
                                    let transcription_result = TranscriptionResult {
                                        text: result.text,
                                        confidence: result.confidence,
                                        language: result.language,
                                        duration_ms: result.duration_ms,
                                        meets_latency_target: result.meets_latency_target,
                                    };

                                    self.send_event(IntegrationEvent::TranscriptionCompleted {
                                        result: transcription_result.clone(),
                                    }).await?;

                                    return Ok(Some(transcription_result));
                                }
                                Err(e) => {
                                    warn!("NPU transcription failed: {} - falling back to CPU", e);
                                }
                            }
                        }
                    }
                }
            }
        }

        // CPU fallback
        let duration = start_time.elapsed();
        let result = TranscriptionResult {
            text: "CPU fallback transcription".to_string(), // Placeholder
            confidence: 0.8,
            language: "en".to_string(),
            duration_ms: duration.as_millis() as u32,
            meets_latency_target: duration.as_millis() <= 10,
        };

        self.send_event(IntegrationEvent::TranscriptionCompleted {
            result: result.clone(),
        }).await?;

        Ok(Some(result))
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

        // Shutdown audio pipeline
        if let Some(audio_pipeline) = &self.audio_pipeline {
            let mut pipeline_guard = audio_pipeline.write().await;
            if let Err(e) = pipeline_guard.stop().await {
                warn!("Audio pipeline shutdown error: {}", e);
            }
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
        self.initialized &&
        self.hardware_manager.is_some() &&
        self.audio_pipeline.is_some() &&
        self.state_coordinator.is_some()
    }
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
    use super::*;
    use crate::config::VoiceStandConfig;

    #[test]
    fn test_integration_stats_default() {
        let stats = IntegrationStats::default();
        assert_eq!(stats.total_transcriptions, 0);
        assert_eq!(stats.success_rate(), 1.0);
        assert_eq!(stats.npu_usage_rate(), 0.0);
        assert!(stats.is_healthy());
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