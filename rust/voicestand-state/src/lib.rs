//! VoiceStand State Management
//!
//! Memory-safe state machine for push-to-talk coordination with dual activation system.
//! Provides <10ms latency response to key presses and voice activation.

use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use tokio::sync::{mpsc, oneshot};
use thiserror::Error;
use tracing::{debug, error, info, warn};

pub mod ptt;
pub mod activation;
pub mod hotkey;
pub mod state_machine;

pub use ptt::{PushToTalkManager, PttEvent};
pub use activation::{ActivationDetector, ActivationEvent};
pub use hotkey::{HotkeyManager, HotkeyConfig, HotkeyEvent};
pub use state_machine::{StateMachine, VoiceStandState, StateTransition};

/// Push-to-talk configuration
#[derive(Debug, Clone)]
pub struct PTTConfig {
    pub enabled: bool,
    pub device_path: Option<String>,
    pub activation_key: Option<String>,
    pub hold_duration_ms: u64,
}

impl Default for PTTConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_path: None,
            activation_key: Some("Space".to_string()),
            hold_duration_ms: 100,
        }
    }
}

/// System activation modes
#[derive(Debug, Clone)]
pub enum ActivationMode {
    PushToTalk,
    VoiceActivated,
    HotkeyToggle,
    AlwaysOn,
    KeyPress,
    WakeWord,
}

/// System state enumeration
#[derive(Debug, Clone)]
pub enum SystemState {
    Idle,
    Listening,
    Processing,
    Error,
}

// Push-to-talk manager is now defined in ptt.rs

/// Activation manager for voice activation
#[derive(Debug)]
pub struct ActivationManager {
    modes: Vec<ActivationMode>,
    detector: ActivationDetector,
    audio_buffer: Vec<f32>,
    last_processed: Instant,
}

impl ActivationManager {
    pub fn new(modes: Vec<ActivationMode>) -> StateResult<Self> {
        use std::time::Duration;

        let detector = ActivationDetector::new(
            0.05, // Energy threshold for voice detection
            Duration::from_millis(200), // Minimum voice duration
        );

        Ok(Self {
            modes,
            detector,
            audio_buffer: Vec::with_capacity(8192),
            last_processed: Instant::now(),
        })
    }

    /// Process audio data for activation detection
    pub async fn process_audio_data(&mut self, audio_data: &[f32]) -> StateResult<Vec<ActivationEvent>> {
        let mut events = Vec::new();

        // Buffer audio data for processing
        self.audio_buffer.extend_from_slice(audio_data);

        // Process in chunks for real-time response
        const CHUNK_SIZE: usize = 1024;
        if self.audio_buffer.len() >= CHUNK_SIZE {
            let chunk: Vec<f32> = self.audio_buffer.drain(..CHUNK_SIZE).collect();

            match self.detector.process_audio(&chunk) {
                Ok(Some(event)) => {
                    events.push(event);
                }
                Ok(None) => {}, // No event
                Err(e) => {
                    return Err(StateError::HardwareError {
                        component: "activation_detector".to_string(),
                        reason: format!("Audio processing failed: {}", e),
                    });
                }
            }
        }

        self.last_processed = Instant::now();
        Ok(events)
    }

    /// Initialize activation manager
    pub async fn initialize(&mut self) -> StateResult<()> {
        info!("Initializing activation manager with modes: {:?}", self.modes);
        Ok(())
    }

    /// Start activation manager
    pub async fn start(&mut self) -> StateResult<mpsc::Receiver<ActivationEvent>> {
        let (_tx, rx) = mpsc::channel(100);
        // This would normally start a background task for activation processing
        // For now we return an empty receiver that will be fed by process_audio_data
        Ok(rx)
    }

    /// Shutdown activation manager
    pub async fn shutdown(&mut self) -> StateResult<()> {
        info!("Shutting down activation manager");
        Ok(())
    }
}

/// State management error types
#[derive(Error, Debug, Clone)]
pub enum StateError {
    /// State machine not initialized
    #[error("State machine not initialized")]
    NotInitialized,

    /// Invalid state transition
    #[error("Invalid state transition: {from} -> {to}")]
    InvalidTransition { from: String, to: String },

    /// Hotkey registration failed
    #[error("Hotkey registration failed: {combination} - {reason}")]
    HotkeyError { combination: String, reason: String },

    /// Activation timeout
    #[error("Activation timeout after {timeout_ms}ms")]
    ActivationTimeout { timeout_ms: u64 },

    /// Configuration error
    #[error("Configuration error: {parameter} - {reason}")]
    ConfigError { parameter: String, reason: String },

    /// Hardware integration error
    #[error("Hardware integration error: {component} - {reason}")]
    HardwareError { component: String, reason: String },

    /// Concurrency error
    #[error("Concurrency error: {operation} - {reason}")]
    ConcurrencyError { operation: String, reason: String },

    /// System error
    #[error("System error: {operation} - {error}")]
    SystemError {
        operation: String,
        error: String,
    },
}

/// State operation result type
pub type StateResult<T> = Result<T, StateError>;

impl From<anyhow::Error> for StateError {
    fn from(error: anyhow::Error) -> Self {
        StateError::SystemError {
            operation: "anyhow_conversion".to_string(),
            error: error.to_string(),
        }
    }
}

/// Main state management configuration
#[derive(Debug, Clone)]
pub struct StateConfig {
    /// Push-to-talk configuration
    pub ptt_config: PTTConfig,
    /// Hotkey configuration
    pub hotkey_config: HotkeyConfig,
    /// Activation modes enabled
    pub activation_modes: Vec<ActivationMode>,
    /// Target response latency in milliseconds
    pub target_latency_ms: f32,
    /// Enable fallback modes
    pub enable_fallbacks: bool,
    /// Debug mode for detailed logging
    pub debug_mode: bool,
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            ptt_config: PTTConfig::default(),
            hotkey_config: HotkeyConfig::default(),
            activation_modes: vec![
                ActivationMode::KeyPress,
                ActivationMode::WakeWord,
            ],
            target_latency_ms: 10.0, // <10ms response target
            enable_fallbacks: true,
            debug_mode: false,
        }
    }
}

/// System event types for state coordination
#[derive(Debug, Clone)]
pub enum SystemEvent {
    /// System starting up
    Startup,
    /// System shutting down
    Shutdown,
    /// Hardware component available
    HardwareAvailable { component: String },
    /// Hardware component failed
    HardwareFailed { component: String, error: String },
    /// Audio system ready
    AudioReady,
    /// Audio system failed
    AudioFailed { error: String },
    /// NPU ready for inference
    NPUReady,
    /// NPU failed
    NPUFailed { error: String },
    /// GNA ready for wake word detection
    GNAReady,
    /// GNA failed
    GNAFailed { error: String },
    /// Push-to-talk event
    PTT(PttEvent),
    /// Hotkey event
    Hotkey(HotkeyEvent),
    /// Activation event
    Activation(ActivationEvent),
    /// State transition event
    StateTransition { from: SystemState, to: SystemState },
    /// Error occurred
    Error { error: StateError },
}

/// System metrics for monitoring
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Total state transitions
    pub total_transitions: u64,
    /// Average transition latency in milliseconds
    pub average_transition_latency_ms: f32,
    /// Peak transition latency in milliseconds
    pub peak_transition_latency_ms: f32,
    /// Push-to-talk activations
    pub ptt_activations: u64,
    /// Wake word activations
    pub wake_word_activations: u64,
    /// Failed activations
    pub failed_activations: u64,
    /// Current state duration
    pub current_state_duration: Duration,
    /// System uptime
    pub uptime: Duration,
    /// Hardware components available
    pub hardware_components_available: u32,
    /// Hardware components failed
    pub hardware_components_failed: u32,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            total_transitions: 0,
            average_transition_latency_ms: 0.0,
            peak_transition_latency_ms: 0.0,
            ptt_activations: 0,
            wake_word_activations: 0,
            failed_activations: 0,
            current_state_duration: Duration::ZERO,
            uptime: Duration::ZERO,
            hardware_components_available: 0,
            hardware_components_failed: 0,
        }
    }
}

impl SystemMetrics {
    /// Update transition metrics
    pub fn update_transition(&mut self, latency: Duration) {
        let latency_ms = latency.as_secs_f32() * 1000.0;

        if self.total_transitions == 0 {
            self.average_transition_latency_ms = latency_ms;
        } else {
            // Exponential moving average
            self.average_transition_latency_ms =
                0.9 * self.average_transition_latency_ms + 0.1 * latency_ms;
        }

        if latency_ms > self.peak_transition_latency_ms {
            self.peak_transition_latency_ms = latency_ms;
        }

        self.total_transitions += 1;
    }

    /// Record PTT activation
    pub fn record_ptt_activation(&mut self) {
        self.ptt_activations += 1;
    }

    /// Record wake word activation
    pub fn record_wake_word_activation(&mut self) {
        self.wake_word_activations += 1;
    }

    /// Record failed activation
    pub fn record_failed_activation(&mut self) {
        self.failed_activations += 1;
    }

    /// Update hardware status
    pub fn update_hardware_status(&mut self, available: u32, failed: u32) {
        self.hardware_components_available = available;
        self.hardware_components_failed = failed;
    }

    /// Update uptime
    pub fn update_uptime(&mut self, uptime: Duration) {
        self.uptime = uptime;
    }

    /// Update current state duration
    pub fn update_state_duration(&mut self, duration: Duration) {
        self.current_state_duration = duration;
    }

    /// Check if metrics indicate healthy operation
    pub fn is_healthy(&self) -> bool {
        self.average_transition_latency_ms <= 10.0 // <10ms transition target
            && self.hardware_components_available > 0 // At least one hardware component
            && (if self.total_transitions > 0 {
                (self.failed_activations as f32 / self.total_transitions as f32) < 0.05 // <5% failure rate
            } else {
                true
            })
    }

    /// Get success rate
    pub fn success_rate(&self) -> f32 {
        if self.total_transitions == 0 {
            return 1.0;
        }

        let successful = self.total_transitions - self.failed_activations;
        successful as f32 / self.total_transitions as f32
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "=== VoiceStand System Metrics ===\n\
             Uptime: {:.1}s\n\
             Total Transitions: {}\n\
             Average Latency: {:.2}ms\n\
             Peak Latency: {:.2}ms\n\
             PTT Activations: {}\n\
             Wake Word Activations: {}\n\
             Failed Activations: {} ({:.1}%)\n\
             Success Rate: {:.1}%\n\
             Hardware Available: {}\n\
             Hardware Failed: {}\n\
             Current State Duration: {:.1}s\n\
             Health Status: {}\n",
            self.uptime.as_secs_f32(),
            self.total_transitions,
            self.average_transition_latency_ms,
            self.peak_transition_latency_ms,
            self.ptt_activations,
            self.wake_word_activations,
            self.failed_activations,
            if self.total_transitions > 0 { self.failed_activations as f32 / self.total_transitions as f32 * 100.0 } else { 0.0 },
            self.success_rate() * 100.0,
            self.hardware_components_available,
            self.hardware_components_failed,
            self.current_state_duration.as_secs_f32(),
            if self.is_healthy() { "✅ HEALTHY" } else { "⚠️ ISSUES DETECTED" }
        )
    }
}

/// Main VoiceStand coordinator
pub struct VoiceStandCoordinator {
    config: StateConfig,
    state_machine: StateMachine,
    ptt_manager: PushToTalkManager,
    activation_manager: ActivationManager,
    hotkey_manager: HotkeyManager,
    metrics: Arc<RwLock<SystemMetrics>>,
    event_tx: mpsc::Sender<SystemEvent>,
    event_rx: Option<mpsc::Receiver<SystemEvent>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    start_time: Instant,
}

impl VoiceStandCoordinator {
    /// Create new VoiceStand coordinator
    pub fn new(config: StateConfig) -> StateResult<Self> {
        let (event_tx, event_rx) = mpsc::channel(1000);

        let (ptt_manager, _ptt_rx) = PushToTalkManager::new();

        Ok(Self {
            state_machine: StateMachine::new(),
            ptt_manager,
            activation_manager: ActivationManager::new(config.activation_modes.clone())?,
            hotkey_manager: HotkeyManager::new(),
            metrics: Arc::new(RwLock::new(SystemMetrics::default())),
            config,
            event_tx,
            event_rx: Some(event_rx),
            shutdown_tx: None,
            start_time: Instant::now(),
        })
    }

    /// Initialize the coordinator and start all subsystems
    pub async fn initialize(&mut self) -> StateResult<()> {
        info!("🚀 Initializing VoiceStand coordinator");

        // Send startup event
        self.send_event(SystemEvent::Startup).await?;

        // Initialize state machine
        self.state_machine.initialize().await?;

        // Initialize hotkey manager
        self.hotkey_manager.initialize().await
            .map_err(|e| StateError::HardwareError {
                component: "hotkey".to_string(),
                reason: e.to_string(),
            })?;

        // Initialize activation manager
        self.activation_manager.initialize().await
            .map_err(|e| StateError::HardwareError {
                component: "activation".to_string(),
                reason: e.to_string(),
            })?;

        // Initialize PTT manager
        self.ptt_manager.initialize().await
            .map_err(|e| StateError::HardwareError {
                component: "ptt".to_string(),
                reason: e.to_string(),
            })?;

        info!("✅ VoiceStand coordinator initialized successfully");
        Ok(())
    }

    /// Start the coordinator event loop
    pub async fn start(&mut self) -> StateResult<mpsc::Receiver<SystemEvent>> {
        let event_rx = self.event_rx.take()
            .ok_or_else(|| StateError::ConcurrencyError {
                operation: "start".to_string(),
                reason: "Event receiver already taken".to_string(),
            })?;

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);

        // Start all subsystems
        let ptt_events = self.ptt_manager.start().await?;
        let activation_events = self.activation_manager.start().await?;
        let hotkey_events = self.hotkey_manager.start().await?;

        // Create event forwarding channels
        let _event_tx = self.event_tx.clone();
        let metrics = Arc::clone(&self.metrics);
        let start_time = self.start_time;

        // Start main event loop
        let main_event_tx = self.event_tx.clone();
        tokio::spawn(async move {
            Self::event_loop(
                event_rx,
                ptt_events,
                activation_events,
                hotkey_events,
                main_event_tx,
                metrics,
                start_time,
                shutdown_rx,
            ).await;
        });

        // Create public event stream
        let (_public_tx, public_rx) = mpsc::channel(100);

        // Forward public events (will be handled in the main event loop)
        // Removed subscribe code as mpsc::Sender doesn't support broadcast

        info!("🎤 VoiceStand coordinator started - ready for voice commands");
        Ok(public_rx)
    }

    /// Main event processing loop
    async fn event_loop(
        mut main_events: mpsc::Receiver<SystemEvent>,
        mut ptt_events: mpsc::UnboundedReceiver<PttEvent>,
        mut activation_events: mpsc::Receiver<ActivationEvent>,
        mut hotkey_events: mpsc::UnboundedReceiver<HotkeyEvent>,
        event_tx: mpsc::Sender<SystemEvent>,
        metrics: Arc<RwLock<SystemMetrics>>,
        start_time: Instant,
        mut shutdown_rx: oneshot::Receiver<()>,
    ) {
        info!("Starting VoiceStand event loop");

        loop {
            tokio::select! {
                // Main system events
                event = main_events.recv() => {
                    match event {
                        Some(event) => {
                            Self::handle_system_event(&event, &metrics, start_time).await;
                        }
                        None => break,
                    }
                }

                // PTT events
                event = ptt_events.recv() => {
                    if let Some(ptt_event) = event {
                        let system_event = SystemEvent::PTT(ptt_event);
                        let _ = event_tx.send(system_event).await;
                    }
                }

                // Activation events
                event = activation_events.recv() => {
                    if let Some(activation_event) = event {
                        let system_event = SystemEvent::Activation(activation_event);
                        let _ = event_tx.send(system_event).await;
                    }
                }

                // Hotkey events
                event = hotkey_events.recv() => {
                    if let Some(hotkey_event) = event {
                        let system_event = SystemEvent::Hotkey(hotkey_event);
                        let _ = event_tx.send(system_event).await;
                    }
                }

                // Shutdown signal
                _ = &mut shutdown_rx => {
                    info!("Received shutdown signal");
                    break;
                }
            }
        }

        info!("VoiceStand event loop ended");
    }

    /// Handle individual system events
    async fn handle_system_event(
        event: &SystemEvent,
        metrics: &Arc<RwLock<SystemMetrics>>,
        start_time: Instant,
    ) {
        match event {
            SystemEvent::Startup => {
                info!("🚀 VoiceStand system starting up");
            }

            SystemEvent::Shutdown => {
                info!("🛑 VoiceStand system shutting down");
            }

            SystemEvent::HardwareAvailable { component } => {
                info!("✅ Hardware component available: {}", component);
                let mut metrics_guard = metrics.write();
                metrics_guard.hardware_components_available += 1;
            }

            SystemEvent::HardwareFailed { component, error } => {
                warn!("❌ Hardware component failed: {} - {}", component, error);
                let mut metrics_guard = metrics.write();
                metrics_guard.hardware_components_failed += 1;
            }

            SystemEvent::PTT(ptt_event) => {
                let mut metrics_guard = metrics.write();
                match ptt_event {
                    PttEvent::Pressed { .. } => {
                        metrics_guard.record_ptt_activation();
                        debug!("🔴 PTT activated");
                    }
                    PttEvent::Released { .. } => {
                        debug!("⚪ PTT released");
                    }
                    PttEvent::Error { .. } => {
                        metrics_guard.record_failed_activation();
                    }
                }
            }

            SystemEvent::Activation(activation_event) => {
                let mut metrics_guard = metrics.write();
                match activation_event {
                    ActivationEvent::WakeWordDetected => {
                        metrics_guard.record_wake_word_activation();
                        debug!("🔊 Wake word detected");
                    }
                    ActivationEvent::VoiceActivityStarted => {
                        debug!("🎤 Voice activity started");
                    }
                    ActivationEvent::VoiceActivityEnded => {
                        debug!("⏹️ Voice activity ended");
                    }
                    ActivationEvent::SilenceDetected => {
                        debug!("🔇 Silence detected");
                    }
                }
            }

            SystemEvent::StateTransition { from, to } => {
                debug!("State transition: {:?} -> {:?}", from, to);
                // Record transition latency (estimated)
                let mut metrics_guard = metrics.write();
                metrics_guard.update_transition(Duration::from_millis(1));
            }

            SystemEvent::Error { error } => {
                error!("System error: {}", error);
                let mut metrics_guard = metrics.write();
                metrics_guard.record_failed_activation();
            }

            _ => {
                // Handle other events as needed
            }
        }

        // Update uptime
        let mut metrics_guard = metrics.write();
        metrics_guard.update_uptime(start_time.elapsed());
    }

    /// Send system event
    async fn send_event(&self, event: SystemEvent) -> StateResult<()> {
        self.event_tx.send(event).await
            .map_err(|_| StateError::ConcurrencyError {
                operation: "send_event".to_string(),
                reason: "Event channel closed".to_string(),
            })
    }

    /// Get current system metrics
    pub fn get_metrics(&self) -> SystemMetrics {
        self.metrics.read().clone()
    }

    /// Get current system state
    pub fn get_state(&self) -> SystemState {
        match self.state_machine.current_state() {
            state_machine::VoiceStandState::Idle => SystemState::Idle,
            state_machine::VoiceStandState::Listening => SystemState::Listening,
            state_machine::VoiceStandState::Processing => SystemState::Processing,
            state_machine::VoiceStandState::Speaking => SystemState::Processing, // Map Speaking to Processing
            state_machine::VoiceStandState::Error => SystemState::Error,
        }
    }

    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        self.metrics.read().is_healthy()
    }

    /// Process audio data for activation detection (CRITICAL FOR PIPELINE INTEGRATION)
    pub async fn process_audio_frame(&mut self, audio_data: &[f32]) -> StateResult<Vec<SystemEvent>> {
        let mut events = Vec::new();

        // Process through activation manager
        let activation_events = self.activation_manager.process_audio_data(audio_data).await?;

        // Convert activation events to system events
        for activation_event in activation_events {
            let system_event = SystemEvent::Activation(activation_event.clone());
            events.push(system_event.clone());
            self.send_event(system_event).await?;
        }

        Ok(events)
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&mut self) -> StateResult<()> {
        info!("🛑 Shutting down VoiceStand coordinator");

        // Send shutdown event
        self.send_event(SystemEvent::Shutdown).await?;

        // Trigger shutdown
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        // Shutdown all subsystems
        self.ptt_manager.shutdown().await?;
        self.activation_manager.shutdown().await?;
        self.hotkey_manager.shutdown().await?;
        self.state_machine.shutdown().await?;

        info!("✅ VoiceStand coordinator shutdown complete");
        Ok(())
    }
}

impl Drop for VoiceStandCoordinator {
    fn drop(&mut self) {
        if self.shutdown_tx.is_some() {
            warn!("VoiceStandCoordinator dropped without explicit shutdown");
        }
    }
}

/// Utility functions
pub mod utils {
    use super::*;

    /// Create default coordinator for quick setup
    pub fn create_default_coordinator() -> StateResult<VoiceStandCoordinator> {
        VoiceStandCoordinator::new(StateConfig::default())
    }

    /// Create coordinator with custom hotkey
    pub fn create_coordinator_with_hotkey(hotkey: &str) -> StateResult<VoiceStandCoordinator> {
        let mut config = StateConfig::default();
        config.hotkey_config.key = hotkey.to_string();
        VoiceStandCoordinator::new(config)
    }

    /// Check system compatibility
    pub async fn check_system_compatibility() -> StateResult<SystemCompatibility> {
        let mut compatibility = SystemCompatibility::default();

        // Check for NPU availability
        compatibility.npu_available = check_npu_availability().await;

        // Check for GNA availability
        compatibility.gna_available = check_gna_availability().await;

        // Check audio system
        compatibility.audio_available = check_audio_availability().await;

        // Check hotkey system
        compatibility.hotkey_available = check_hotkey_availability().await;

        Ok(compatibility)
    }

    async fn check_npu_availability() -> bool {
        // Would check for Intel NPU hardware
        false // Placeholder
    }

    async fn check_gna_availability() -> bool {
        // Would check for Intel GNA hardware
        false // Placeholder
    }

    async fn check_audio_availability() -> bool {
        // Would check for audio input devices
        true // Usually available
    }

    async fn check_hotkey_availability() -> bool {
        // Would check for global hotkey support
        true // Usually available
    }
}

/// System compatibility information
#[derive(Debug, Clone)]
pub struct SystemCompatibility {
    pub npu_available: bool,
    pub gna_available: bool,
    pub audio_available: bool,
    pub hotkey_available: bool,
}

impl Default for SystemCompatibility {
    fn default() -> Self {
        Self {
            npu_available: false,
            gna_available: false,
            audio_available: true,
            hotkey_available: true,
        }
    }
}

impl SystemCompatibility {
    /// Check if system meets minimum requirements
    pub fn meets_minimum_requirements(&self) -> bool {
        self.audio_available && self.hotkey_available
    }

    /// Check if system supports advanced features
    pub fn supports_advanced_features(&self) -> bool {
        self.npu_available || self.gna_available
    }

    /// Get capability summary
    pub fn capabilities_summary(&self) -> String {
        let mut capabilities = Vec::new();

        if self.audio_available {
            capabilities.push("Audio Capture");
        }

        if self.hotkey_available {
            capabilities.push("Global Hotkeys");
        }

        if self.npu_available {
            capabilities.push("NPU Voice-to-Text");
        } else {
            capabilities.push("CPU Voice-to-Text");
        }

        if self.gna_available {
            capabilities.push("GNA Wake Words");
        }

        capabilities.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_config_default() {
        let config = StateConfig::default();
        assert_eq!(config.target_latency_ms, 10.0);
        assert!(config.enable_fallbacks);
        assert!(!config.debug_mode);
        assert_eq!(config.activation_modes.len(), 2);
    }

    #[test]
    fn test_system_metrics_default() {
        let metrics = SystemMetrics::default();
        assert_eq!(metrics.total_transitions, 0);
        assert_eq!(metrics.ptt_activations, 0);
        assert_eq!(metrics.wake_word_activations, 0);
        assert!(metrics.is_healthy());
    }

    #[test]
    fn test_system_metrics_transitions() {
        let mut metrics = SystemMetrics::default();
        metrics.update_transition(Duration::from_millis(5));
        assert_eq!(metrics.average_transition_latency_ms, 5.0);
        assert_eq!(metrics.peak_transition_latency_ms, 5.0);
        assert_eq!(metrics.total_transitions, 1);

        metrics.update_transition(Duration::from_millis(15));
        assert!(metrics.average_transition_latency_ms > 5.0);
        assert_eq!(metrics.peak_transition_latency_ms, 15.0);
    }

    #[test]
    fn test_system_metrics_success_rate() {
        let mut metrics = SystemMetrics::default();
        assert_eq!(metrics.success_rate(), 1.0);

        metrics.total_transitions = 10;
        metrics.failed_activations = 2;
        assert_eq!(metrics.success_rate(), 0.8);
    }

    #[test]
    fn test_system_compatibility() {
        let compat = SystemCompatibility::default();
        assert!(compat.meets_minimum_requirements());
        assert!(!compat.supports_advanced_features());

        let summary = compat.capabilities_summary();
        assert!(summary.contains("Audio Capture"));
        assert!(summary.contains("Global Hotkeys"));
        assert!(summary.contains("CPU Voice-to-Text"));
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = VoiceStandCoordinator::new(StateConfig::default());
        assert!(coordinator.is_ok());

        let coordinator = match coordinator {
            Ok(coord) => coord,
            Err(e) => {
                panic!("Failed to create coordinator: {}", e);
            }
        };
        assert_eq!(coordinator.get_state(), SystemState::Idle);
    }

    #[tokio::test]
    async fn test_utils_default_coordinator() {
        let coordinator = utils::create_default_coordinator();
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_utils_custom_hotkey() {
        let coordinator = utils::create_coordinator_with_hotkey("ctrl+alt+v");
        assert!(coordinator.is_ok());
    }
}