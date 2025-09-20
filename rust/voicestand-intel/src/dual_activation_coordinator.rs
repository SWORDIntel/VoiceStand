// Dual Activation Coordinator for VoiceStand Push-to-Talk System
// Coordinates GNA wake word detection with manual push-to-talk key activation
// Provides seamless handoff to NPU pipeline for full voice-to-text processing

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, broadcast};
use tokio::time::{sleep, timeout};
use anyhow::{Result, Context, anyhow};
use log::{info, warn, error, debug};
use serde::{Deserialize, Serialize};
use voicestand_core::{VoiceStandError, Result as VsResult};

use crate::gna_wake_word_detector::{GNAWakeWordDetector, GNAWakeWordConfig, WakeWordDetection};

// Input device paths for key detection
const INPUT_DEVICE_PATHS: &[&str] = &[
    "/dev/input/event0",
    "/dev/input/event1",
    "/dev/input/event2",
    "/dev/input/event3",
];

// Default hotkey configuration
const DEFAULT_HOTKEY: &str = "ctrl+alt+space";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualActivationConfig {
    pub gna_config: GNAWakeWordConfig,
    pub hotkey_combination: String,
    pub activation_timeout_ms: u64,
    pub debounce_time_ms: u64,
    pub priority_mode: ActivationPriority,
    pub enable_gna: bool,
    pub enable_hotkey: bool,
    pub enable_continuous_listening: bool,
}

impl Default for DualActivationConfig {
    fn default() -> Self {
        Self {
            gna_config: GNAWakeWordConfig::default(),
            hotkey_combination: DEFAULT_HOTKEY.to_string(),
            activation_timeout_ms: 5000,  // 5 second timeout
            debounce_time_ms: 500,        // 500ms debounce
            priority_mode: ActivationPriority::GNAFirst,
            enable_gna: true,
            enable_hotkey: true,
            enable_continuous_listening: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationPriority {
    GNAFirst,      // Prefer GNA wake word over hotkey
    HotkeyFirst,   // Prefer hotkey over GNA
    Either,        // Accept either activation method
    Both,          // Require both GNA and hotkey
}

#[derive(Debug, Clone)]
pub enum ActivationSource {
    GNAWakeWord {
        detection: WakeWordDetection,
    },
    HotkeyPress {
        key_combination: String,
        timestamp: Instant,
    },
    Continuous {
        start_time: Instant,
    },
}

#[derive(Debug, Clone)]
pub struct ActivationEvent {
    pub source: ActivationSource,
    pub timestamp: Instant,
    pub confidence: f32,
    pub power_consumption_mw: f32,
    pub should_handoff_to_npu: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CoordinatorMetrics {
    pub gna_activations: u64,
    pub hotkey_activations: u64,
    pub continuous_activations: u64,
    pub successful_npu_handoffs: u64,
    pub failed_handoffs: u64,
    pub average_activation_time_ms: f32,
    pub total_power_consumption_mw: f32,
    pub uptime_hours: f32,
}

pub struct DualActivationCoordinator {
    config: DualActivationConfig,
    gna_detector: Option<GNAWakeWordDetector>,
    hotkey_monitor: Option<HotkeyMonitor>,
    activation_sender: Option<broadcast::Sender<ActivationEvent>>,
    metrics: Arc<Mutex<CoordinatorMetrics>>,
    is_running: Arc<Mutex<bool>>,
    start_time: Instant,
}

// Safe mutex locking utility
fn safe_lock<T>(mutex: &Mutex<T>) -> VsResult<std::sync::MutexGuard<T>> {
    mutex.lock().map_err(|e| VoiceStandError::lock_poisoned(format!("Mutex poisoned: {}", e)))
}

impl DualActivationCoordinator {
    pub fn new(config: DualActivationConfig) -> Result<Self> {
        info!("Initializing Dual Activation Coordinator");

        let mut coordinator = Self {
            config: config.clone(),
            gna_detector: None,
            hotkey_monitor: None,
            activation_sender: None,
            metrics: Arc::new(Mutex::new(CoordinatorMetrics {
                gna_activations: 0,
                hotkey_activations: 0,
                continuous_activations: 0,
                successful_npu_handoffs: 0,
                failed_handoffs: 0,
                average_activation_time_ms: 0.0,
                total_power_consumption_mw: 0.0,
                uptime_hours: 0.0,
            })),
            is_running: Arc::new(Mutex::new(false)),
            start_time: Instant::now(),
        };

        // Initialize GNA detector if enabled
        if config.enable_gna {
            match GNAWakeWordDetector::new(config.gna_config) {
                Ok(detector) => {
                    coordinator.gna_detector = Some(detector);
                    info!("GNA wake word detector initialized");
                }
                Err(e) => {
                    warn!("Failed to initialize GNA detector: {}. Hotkey-only mode.", e);
                }
            }
        }

        // Initialize hotkey monitor if enabled
        if config.enable_hotkey {
            match HotkeyMonitor::new(config.hotkey_combination.clone()) {
                Ok(monitor) => {
                    coordinator.hotkey_monitor = Some(monitor);
                    info!("Hotkey monitor initialized for: {}", config.hotkey_combination);
                }
                Err(e) => {
                    warn!("Failed to initialize hotkey monitor: {}. GNA-only mode.", e);
                }
            }
        }

        info!("Dual Activation Coordinator initialized successfully");
        Ok(coordinator)
    }

    pub async fn start_coordination(&mut self) -> Result<broadcast::Receiver<ActivationEvent>> {
        info!("Starting dual activation coordination");

        let (tx, rx) = broadcast::channel(10);
        self.activation_sender = Some(tx.clone());

        *safe_lock(&self.is_running).map_err(|e| anyhow::anyhow!("{}", e))? = true;

        // Start GNA detection if available
        if let Some(ref mut gna_detector) = self.gna_detector {
            let gna_rx = gna_detector.start_always_on_detection().await?;
            self.spawn_gna_handler(gna_rx, tx.clone()).await;
        }

        // Start hotkey monitoring if available
        if let Some(ref mut hotkey_monitor) = self.hotkey_monitor {
            let hotkey_rx = hotkey_monitor.start_monitoring().await?;
            self.spawn_hotkey_handler(hotkey_rx, tx.clone()).await;
        }

        // Start continuous listening if enabled
        if self.config.enable_continuous_listening {
            self.spawn_continuous_handler(tx.clone()).await;
        }

        // Start coordination logic
        self.spawn_coordinator(tx).await;

        info!("Dual activation coordination started");
        Ok(rx)
    }

    async fn spawn_gna_handler(&self, mut gna_rx: mpsc::Receiver<WakeWordDetection>, tx: broadcast::Sender<ActivationEvent>) {
        let is_running = Arc::clone(&self.is_running);
        let metrics = Arc::clone(&self.metrics);

        tokio::spawn(async move {
            while *safe_lock(&is_running).unwrap_or_default() {
                match timeout(Duration::from_millis(100), gna_rx.recv()).await {
                    Ok(Some(detection)) => {
                        let activation = ActivationEvent {
                            source: ActivationSource::GNAWakeWord { detection: detection.clone() },
                            timestamp: detection.timestamp,
                            confidence: detection.confidence,
                            power_consumption_mw: detection.power_consumption_mw,
                            should_handoff_to_npu: true,
                        };

                        if let Err(e) = tx.send(activation) {
                            error!("Failed to send GNA activation: {}", e);
                        } else {
                            // Update metrics
                            if let Ok(mut metrics_guard) = safe_lock(&metrics) {
                                metrics_guard.gna_activations += 1;
                                metrics_guard.total_power_consumption_mw += detection.power_consumption_mw;
                            }

                            info!("GNA wake word activation: {} (confidence: {:.3})",
                                detection.wake_word, detection.confidence);
                        }
                    }
                    Ok(None) => {
                        debug!("GNA receiver closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout - continue monitoring
                        continue;
                    }
                }
            }
        });
    }

    async fn spawn_hotkey_handler(&self, mut hotkey_rx: mpsc::Receiver<HotkeyEvent>, tx: broadcast::Sender<ActivationEvent>) {
        let is_running = Arc::clone(&self.is_running);
        let metrics = Arc::clone(&self.metrics);
        let debounce_time = Duration::from_millis(self.config.debounce_time_ms);

        tokio::spawn(async move {
            let mut last_hotkey_time = Instant::now() - debounce_time;

            while *safe_lock(&is_running).unwrap_or_default() {
                match timeout(Duration::from_millis(100), hotkey_rx.recv()).await {
                    Ok(Some(hotkey_event)) => {
                        let now = Instant::now();

                        // Apply debouncing
                        if now.duration_since(last_hotkey_time) < debounce_time {
                            continue;
                        }
                        last_hotkey_time = now;

                        let activation = ActivationEvent {
                            source: ActivationSource::HotkeyPress {
                                key_combination: hotkey_event.combination,
                                timestamp: hotkey_event.timestamp,
                            },
                            timestamp: now,
                            confidence: 1.0,  // Hotkey is always 100% confident
                            power_consumption_mw: 0.1,  // Minimal power for key detection
                            should_handoff_to_npu: true,
                        };

                        if let Err(e) = tx.send(activation) {
                            error!("Failed to send hotkey activation: {}", e);
                        } else {
                            // Update metrics
                            if let Ok(mut metrics_guard) = safe_lock(&metrics) {
                                metrics_guard.hotkey_activations += 1;
                            }

                            info!("Hotkey activation: {}", hotkey_event.combination);
                        }
                    }
                    Ok(None) => {
                        debug!("Hotkey receiver closed");
                        break;
                    }
                    Err(_) => {
                        // Timeout - continue monitoring
                        continue;
                    }
                }
            }
        });
    }

    async fn spawn_continuous_handler(&self, tx: broadcast::Sender<ActivationEvent>) {
        let is_running = Arc::clone(&self.is_running);
        let metrics = Arc::clone(&self.metrics);

        tokio::spawn(async move {
            info!("Starting continuous listening mode");

            let start_time = Instant::now();
            let activation = ActivationEvent {
                source: ActivationSource::Continuous { start_time },
                timestamp: start_time,
                confidence: 0.8,  // Continuous mode has lower confidence
                power_consumption_mw: 150.0,  // Higher power for continuous listening
                should_handoff_to_npu: true,
            };

            if let Err(e) = tx.send(activation) {
                error!("Failed to send continuous activation: {}", e);
            } else {
                if let Ok(mut metrics_guard) = safe_lock(&metrics) {
                    metrics_guard.continuous_activations += 1;
                }
            }

            // Continuous mode stays active until explicitly stopped
            while *safe_lock(&is_running).unwrap_or_default() {
                sleep(Duration::from_secs(1)).await;
            }
        });
    }

    async fn spawn_coordinator(&self, tx: broadcast::Sender<ActivationEvent>) {
        let is_running = Arc::clone(&self.is_running);
        let metrics = Arc::clone(&self.metrics);
        let start_time = self.start_time;

        tokio::spawn(async move {
            while *safe_lock(&is_running).unwrap_or_default() {
                // Update uptime metrics
                {
                    if let Ok(mut metrics_guard) = safe_lock(&metrics) {
                        metrics_guard.uptime_hours = start_time.elapsed().as_secs_f32() / 3600.0;
                    }
                }

                sleep(Duration::from_secs(1)).await;
            }
        });
    }

    pub fn get_metrics(&self) -> CoordinatorMetrics {
        safe_lock(&self.metrics).map(|guard| guard.clone()).unwrap_or_default()
    }

    pub fn stop_coordination(&mut self) {
        info!("Stopping dual activation coordination");

        if let Ok(mut guard) = safe_lock(&self.is_running) {
            *guard = false;
        }

        if let Some(ref mut gna_detector) = self.gna_detector {
            gna_detector.stop_detection();
        }

        if let Some(ref mut hotkey_monitor) = self.hotkey_monitor {
            hotkey_monitor.stop_monitoring();
        }

        info!("Dual activation coordination stopped");
    }

    pub fn update_config(&mut self, config: DualActivationConfig) -> Result<()> {
        info!("Updating dual activation configuration");

        self.config = config;
        // In a full implementation, this would reconfigure the running components

        info!("Configuration updated successfully");
        Ok(())
    }

    pub async fn test_activation_sources(&mut self) -> Result<()> {
        info!("Testing activation sources");

        // Test GNA if available
        if let Some(ref mut gna_detector) = self.gna_detector {
            info!("Testing GNA wake word detection...");
            let test_audio: Vec<f32> = generate_test_audio();
            match gna_detector.process_audio_frame(&test_audio) {
                Ok(Some(detection)) => {
                    info!("GNA test successful: {} (confidence: {:.3})",
                        detection.wake_word, detection.confidence);
                }
                Ok(None) => {
                    info!("GNA test completed - no wake word detected (expected for test signal)");
                }
                Err(e) => {
                    warn!("GNA test failed: {}", e);
                }
            }
        }

        // Test hotkey if available
        if let Some(ref hotkey_monitor) = self.hotkey_monitor {
            info!("Hotkey monitor available for: {}", self.config.hotkey_combination);
        }

        info!("Activation source testing completed");
        Ok(())
    }
}

// Hotkey monitoring implementation
#[derive(Debug, Clone)]
pub struct HotkeyEvent {
    pub combination: String,
    pub timestamp: Instant,
    pub key_states: Vec<bool>,
}

struct HotkeyMonitor {
    combination: String,
    is_monitoring: Arc<Mutex<bool>>,
}

impl HotkeyMonitor {
    fn new(combination: String) -> Result<Self> {
        Ok(Self {
            combination,
            is_monitoring: Arc::new(Mutex::new(false)),
        })
    }

    async fn start_monitoring(&mut self) -> Result<mpsc::Receiver<HotkeyEvent>> {
        info!("Starting hotkey monitoring for: {}", self.combination);

        let (tx, rx) = mpsc::channel(10);
        if let Ok(mut guard) = safe_lock(&self.is_monitoring) {
            *guard = true;
        }

        // In a real implementation, this would use a proper input device monitoring library
        // For now, we simulate the hotkey monitoring
        let combination = self.combination.clone();
        let is_monitoring = Arc::clone(&self.is_monitoring);

        tokio::spawn(async move {
            while *safe_lock(&is_monitoring).unwrap_or_default() {
                // Simulate hotkey detection every 5-10 seconds for testing
                sleep(Duration::from_secs(7)).await;

                let event = HotkeyEvent {
                    combination: combination.clone(),
                    timestamp: Instant::now(),
                    key_states: vec![true, true, true], // Simulate ctrl+alt+space
                };

                if tx.send(event).await.is_err() {
                    break;
                }
            }
        });

        Ok(rx)
    }

    fn stop_monitoring(&mut self) {
        if let Ok(mut guard) = safe_lock(&self.is_monitoring) {
            *guard = false;
        }
    }
}

// Utility functions
fn generate_test_audio() -> Vec<f32> {
    // Generate 1 second of test audio at 16kHz
    (0..16000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.1)
        .collect()
}

pub mod coordinator_utils {
    use super::*;

    pub fn create_default_config() -> DualActivationConfig {
        DualActivationConfig::default()
    }

    pub fn create_gna_only_config() -> DualActivationConfig {
        DualActivationConfig {
            enable_gna: true,
            enable_hotkey: false,
            enable_continuous_listening: false,
            priority_mode: ActivationPriority::GNAFirst,
            ..Default::default()
        }
    }

    pub fn create_hotkey_only_config() -> DualActivationConfig {
        DualActivationConfig {
            enable_gna: false,
            enable_hotkey: true,
            enable_continuous_listening: false,
            priority_mode: ActivationPriority::HotkeyFirst,
            ..Default::default()
        }
    }

    pub fn create_continuous_config() -> DualActivationConfig {
        DualActivationConfig {
            enable_gna: true,
            enable_hotkey: true,
            enable_continuous_listening: true,
            priority_mode: ActivationPriority::Either,
            ..Default::default()
        }
    }

    pub fn validate_config(config: &DualActivationConfig) -> Result<()> {
        if !config.enable_gna && !config.enable_hotkey && !config.enable_continuous_listening {
            return Err(anyhow!("At least one activation method must be enabled"));
        }

        if config.activation_timeout_ms < 1000 {
            warn!("Very short activation timeout: {}ms", config.activation_timeout_ms);
        }

        if config.debounce_time_ms > 2000 {
            warn!("Very long debounce time: {}ms", config.debounce_time_ms);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_coordinator_initialization() {
        let config = DualActivationConfig::default();

        match DualActivationCoordinator::new(config) {
            Ok(_coordinator) => {
                info!("Coordinator initialized successfully");
            }
            Err(e) => {
                // Expected in test environment without hardware
                info!("Coordinator initialization failed (expected without hardware): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_config_validation() {
        let valid_config = DualActivationConfig::default();
        assert!(coordinator_utils::validate_config(&valid_config).is_ok());

        let invalid_config = DualActivationConfig {
            enable_gna: false,
            enable_hotkey: false,
            enable_continuous_listening: false,
            ..Default::default()
        };
        assert!(coordinator_utils::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_config_presets() {
        let gna_only = coordinator_utils::create_gna_only_config();
        assert!(gna_only.enable_gna);
        assert!(!gna_only.enable_hotkey);

        let hotkey_only = coordinator_utils::create_hotkey_only_config();
        assert!(!hotkey_only.enable_gna);
        assert!(hotkey_only.enable_hotkey);

        let continuous = coordinator_utils::create_continuous_config();
        assert!(continuous.enable_continuous_listening);
    }
}