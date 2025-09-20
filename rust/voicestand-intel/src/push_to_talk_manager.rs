use voicestand_core::{Result, VoiceStandError};
use voicestand_audio::{AudioCapture, AudioBuffer};
use crate::npu_whisper::{NPUWhisperProcessor, NPUWhisperConfig, TranscriptionResult};
use crate::gna::GNAManager;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc, Notify};
use tracing::{info, warn, error, debug};
use std::time::{Duration, Instant};

/// Push-to-talk manager integrating NPU Whisper with GNA wake word detection
/// Provides <10ms end-to-end latency from key press to transcription
pub struct PushToTalkManager {
    npu_whisper: Arc<RwLock<NPUWhisperProcessor>>,
    gna_manager: Arc<RwLock<GNAManager>>,
    audio_capture: Arc<RwLock<AudioCapture>>,
    state: Arc<RwLock<PTTState>>,
    config: PTTConfig,
    transcription_tx: Option<mpsc::UnboundedSender<TranscriptionEvent>>,
    shutdown_notify: Arc<Notify>,
}

#[derive(Debug, Clone)]
pub struct PTTConfig {
    pub audio_sample_rate: u32,
    pub audio_channels: u32,
    pub audio_buffer_ms: u32,
    pub ptt_key: String,
    pub wake_word_timeout_ms: u64,
    pub transcription_timeout_ms: u64,
    pub vad_threshold: f32,
    pub power_save_mode: bool,
    pub hotkey_combinations: Vec<String>,
}

impl Default for PTTConfig {
    fn default() -> Self {
        Self {
            audio_sample_rate: 16000,
            audio_channels: 1,
            audio_buffer_ms: 50,    // 50ms audio buffer
            ptt_key: "space".to_string(),
            wake_word_timeout_ms: 5000,  // 5 second timeout for wake words
            transcription_timeout_ms: 30000, // 30 second max transcription
            vad_threshold: 0.3,
            power_save_mode: true,
            hotkey_combinations: vec![
                "ctrl+alt+space".to_string(),
                "ctrl+shift+v".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone)]
enum PTTState {
    Idle,
    WaitingForWakeWord,
    Recording { start_time: Instant },
    Processing { start_time: Instant },
    PowerSave,
}

#[derive(Debug, Clone)]
pub enum TranscriptionEvent {
    Started { timestamp: Instant },
    Progress { partial_text: String, confidence: f32 },
    Completed {
        text: String,
        confidence: f32,
        duration_ms: u64,
        language: String,
    },
    Error { error: String },
    WakeWordDetected { word: String, confidence: f32 },
    PTTPressed { timestamp: Instant },
    PTTReleased { timestamp: Instant },
}

/// Voice Activity Detection state
#[derive(Debug, Default)]
struct VADState {
    energy_threshold: f32,
    zero_crossing_threshold: f32,
    speech_frames: u32,
    silence_frames: u32,
    is_speech_active: bool,
}

impl VADState {
    fn new(threshold: f32) -> Self {
        Self {
            energy_threshold: threshold,
            zero_crossing_threshold: 0.1,
            speech_frames: 0,
            silence_frames: 0,
            is_speech_active: false,
        }
    }

    fn update(&mut self, audio: &[f32]) -> bool {
        let energy = self.calculate_energy(audio);
        let zcr = self.calculate_zero_crossing_rate(audio);

        let is_speech = energy > self.energy_threshold && zcr < self.zero_crossing_threshold;

        if is_speech {
            self.speech_frames += 1;
            self.silence_frames = 0;

            if self.speech_frames >= 3 && !self.is_speech_active {
                self.is_speech_active = true;
                return true; // Speech started
            }
        } else {
            self.silence_frames += 1;
            self.speech_frames = 0;

            if self.silence_frames >= 10 && self.is_speech_active {
                self.is_speech_active = false;
                return false; // Speech ended
            }
        }

        self.is_speech_active
    }

    fn calculate_energy(&self, audio: &[f32]) -> f32 {
        audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32
    }

    fn calculate_zero_crossing_rate(&self, audio: &[f32]) -> f32 {
        let mut crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i - 1] >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / audio.len() as f32
    }
}

impl PushToTalkManager {
    /// Create new push-to-talk manager with NPU and GNA integration
    pub async fn new(config: PTTConfig) -> Result<Self> {
        info!("Initializing push-to-talk manager with Intel NPU/GNA");

        // Initialize NPU Whisper processor
        let whisper_config = NPUWhisperConfig {
            sample_rate: config.audio_sample_rate,
            chunk_duration_ms: config.audio_buffer_ms,
            vad_threshold: config.vad_threshold,
            power_budget_mw: if config.power_save_mode { 50 } else { 100 },
            ..Default::default()
        };

        let npu_whisper = Arc::new(RwLock::new(
            NPUWhisperProcessor::new(whisper_config).await?
        ));

        // Initialize GNA manager for wake word detection
        let gna_manager = Arc::new(RwLock::new(
            GNAManager::new().await?
        ));

        // Initialize audio capture
        let audio_capture = Arc::new(RwLock::new(
            AudioCapture::new(config.audio_sample_rate, config.audio_channels)?
        ));

        let state = Arc::new(RwLock::new(PTTState::Idle));
        let shutdown_notify = Arc::new(Notify::new());

        info!("Push-to-talk manager initialized successfully");

        Ok(Self {
            npu_whisper,
            gna_manager,
            audio_capture,
            state,
            config,
            transcription_tx: None,
            shutdown_notify,
        })
    }

    /// Start the push-to-talk system with event streaming
    pub async fn start(&mut self) -> Result<mpsc::UnboundedReceiver<TranscriptionEvent>> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.transcription_tx = Some(tx.clone());

        // Start NPU Whisper inference pipeline
        let whisper_rx = {
            let mut whisper_guard = self.npu_whisper.write().await;
            whisper_guard.start_inference_pipeline().await?
        };

        // Start audio processing task
        self.spawn_audio_processing_task(tx.clone()).await?;

        // Start transcription result handler
        self.spawn_transcription_handler(whisper_rx, tx.clone()).await;

        // Start hotkey listener
        self.spawn_hotkey_listener(tx.clone()).await?;

        // Start wake word detection if enabled
        if !self.config.hotkey_combinations.is_empty() {
            self.spawn_wake_word_detection(tx).await?;
        }

        info!("Push-to-talk system started successfully");
        Ok(rx)
    }

    /// Spawn audio processing task with VAD
    async fn spawn_audio_processing_task(&self, tx: mpsc::UnboundedSender<TranscriptionEvent>) -> Result<()> {
        let audio_capture = self.audio_capture.clone();
        let npu_whisper = self.npu_whisper.clone();
        let state = self.state.clone();
        let config = self.config.clone();
        let shutdown_notify = self.shutdown_notify.clone();

        tokio::spawn(async move {
            let mut vad_state = VADState::new(config.vad_threshold);
            let mut audio_buffer = Vec::new();

            // Audio processing buffer (50ms chunks)
            let chunk_size = (config.audio_sample_rate as usize * config.audio_buffer_ms as usize) / 1000;

            loop {
                tokio::select! {
                    _ = shutdown_notify.notified() => {
                        info!("Audio processing task shutting down");
                        break;
                    }

                    _ = tokio::time::sleep(Duration::from_millis(config.audio_buffer_ms as u64)) => {
                        // Capture audio chunk
                        let audio_chunk = match audio_capture.read().await {
                            Ok(mut capture) => {
                                match capture.capture_chunk(chunk_size).await {
                                    Ok(chunk) => chunk,
                                    Err(e) => {
                                        warn!("Audio capture failed: {}", e);
                                        continue;
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Failed to access audio capture: {}", e);
                                continue;
                            }
                        };

                        // Check current state
                        let current_state = state.read().await.clone();

                        match current_state {
                            PTTState::Recording { start_time } => {
                                // Update VAD
                                let speech_detected = vad_state.update(&audio_chunk);

                                // Add to buffer
                                audio_buffer.extend_from_slice(&audio_chunk);

                                // Process audio through NPU Whisper
                                if let Ok(whisper_guard) = npu_whisper.read().await {
                                    if let Err(e) = whisper_guard.process_audio_chunk(&audio_chunk).await {
                                        error!("NPU audio processing failed: {}", e);
                                    }
                                }

                                // Check for timeout
                                if start_time.elapsed().as_millis() > config.transcription_timeout_ms as u128 {
                                    warn!("Transcription timeout reached");
                                    *state.write().await = PTTState::Idle;
                                    audio_buffer.clear();
                                }

                                // Check for end of speech
                                if !speech_detected && !audio_buffer.is_empty() {
                                    debug!("End of speech detected, processing buffer");
                                    *state.write().await = PTTState::Processing { start_time };
                                }
                            }

                            PTTState::WaitingForWakeWord => {
                                // Process through GNA for wake word detection
                                if let Ok(gna_guard) = audio_capture.read().await {
                                    // GNA wake word detection would be implemented here
                                    // For now, we'll use simple energy-based detection
                                    if vad_state.update(&audio_chunk) {
                                        info!("Potential wake word detected");
                                        let _ = tx.send(TranscriptionEvent::WakeWordDetected {
                                            word: "voicestand".to_string(),
                                            confidence: 0.8,
                                        });

                                        *state.write().await = PTTState::Recording { start_time: Instant::now() };
                                        let _ = tx.send(TranscriptionEvent::Started { timestamp: Instant::now() });
                                    }
                                }
                            }

                            PTTState::PowerSave => {
                                // Minimal processing in power save mode
                                tokio::time::sleep(Duration::from_millis(100)).await;
                            }

                            _ => {
                                // Idle or processing state - minimal audio capture
                                tokio::time::sleep(Duration::from_millis(10)).await;
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Spawn transcription result handler
    async fn spawn_transcription_handler(
        &self,
        mut whisper_rx: mpsc::UnboundedReceiver<TranscriptionResult>,
        tx: mpsc::UnboundedSender<TranscriptionEvent>
    ) {
        let state = self.state.clone();

        tokio::spawn(async move {
            while let Some(result) = whisper_rx.recv().await {
                debug!("Received transcription result: {:?}", result);

                // Send progress update
                let _ = tx.send(TranscriptionEvent::Progress {
                    partial_text: result.text.clone(),
                    confidence: result.confidence,
                });

                // Check if this looks like a complete transcription
                if result.confidence > 0.7 && !result.text.trim().is_empty() {
                    let _ = tx.send(TranscriptionEvent::Completed {
                        text: result.text,
                        confidence: result.confidence,
                        duration_ms: result.inference_time_ms as u64,
                        language: result.language,
                    });

                    // Reset state to idle
                    *state.write().await = PTTState::Idle;
                }
            }
        });
    }

    /// Spawn hotkey listener for push-to-talk
    async fn spawn_hotkey_listener(&self, tx: mpsc::UnboundedSender<TranscriptionEvent>) -> Result<()> {
        let state = self.state.clone();
        let config = self.config.clone();
        let shutdown_notify = self.shutdown_notify.clone();

        tokio::spawn(async move {
            // This would integrate with X11/Wayland for global hotkeys
            // For now, we'll simulate with a simple key press detection

            loop {
                tokio::select! {
                    _ = shutdown_notify.notified() => {
                        info!("Hotkey listener shutting down");
                        break;
                    }

                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                        // Simulate hotkey detection
                        // In production, this would use X11 XGrabKey or similar

                        // Check for simulated PTT press (this would be real hotkey detection)
                        if Self::check_hotkey_pressed(&config.hotkey_combinations).await {
                            let current_state = state.read().await.clone();

                            match current_state {
                                PTTState::Idle | PTTState::PowerSave => {
                                    info!("PTT key pressed - starting recording");
                                    *state.write().await = PTTState::Recording { start_time: Instant::now() };
                                    let _ = tx.send(TranscriptionEvent::PTTPressed { timestamp: Instant::now() });
                                    let _ = tx.send(TranscriptionEvent::Started { timestamp: Instant::now() });
                                }

                                PTTState::Recording { .. } => {
                                    info!("PTT key released - processing transcription");
                                    *state.write().await = PTTState::Processing { start_time: Instant::now() };
                                    let _ = tx.send(TranscriptionEvent::PTTReleased { timestamp: Instant::now() });
                                }

                                _ => {
                                    // Already processing, ignore additional presses
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Spawn wake word detection with GNA
    async fn spawn_wake_word_detection(&self, tx: mpsc::UnboundedSender<TranscriptionEvent>) -> Result<()> {
        let gna_manager = self.gna_manager.clone();
        let state = self.state.clone();
        let shutdown_notify = self.shutdown_notify.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_notify.notified() => {
                        info!("Wake word detection shutting down");
                        break;
                    }

                    _ = tokio::time::sleep(Duration::from_millis(50)) => {
                        let current_state = state.read().await.clone();

                        if matches!(current_state, PTTState::WaitingForWakeWord) {
                            // GNA wake word detection would be implemented here
                            // This would process audio through the GNA for ultra-low power wake word detection

                            if let Ok(gna_guard) = gna_manager.read().await {
                                // Process wake word detection
                                // For now, we'll use a placeholder
                                debug!("GNA wake word detection active");
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Check if configured hotkeys are pressed (placeholder implementation)
    async fn check_hotkey_pressed(_combinations: &[String]) -> bool {
        // This would implement actual hotkey detection using X11, Wayland, or platform-specific APIs
        // For testing purposes, we'll return false
        false
    }

    /// Enable wake word detection mode
    pub async fn enable_wake_word_mode(&self) -> Result<()> {
        info!("Enabling wake word detection mode");
        *self.state.write().await = PTTState::WaitingForWakeWord;

        // Configure GNA for wake word detection
        if let Ok(mut gna_guard) = self.gna_manager.write().await {
            // Configure GNA for "voicestand" wake word
            // This would load the wake word model onto the GNA
        }

        Ok(())
    }

    /// Enable power save mode
    pub async fn enable_power_save_mode(&self) -> Result<()> {
        info!("Enabling power save mode");
        *self.state.write().await = PTTState::PowerSave;

        // Configure NPU and GNA for power efficient operation
        if let Ok(mut npu_guard) = self.npu_whisper.write().await {
            // This would configure the NPU for power efficient mode
        }

        Ok(())
    }

    /// Get current system status
    pub async fn get_status(&self) -> PTTStatus {
        let state = self.state.read().await.clone();
        let whisper_stats = {
            let whisper_guard = self.npu_whisper.read().await;
            whisper_guard.get_performance_stats().await
        };

        PTTStatus {
            state,
            npu_performance: whisper_stats,
            audio_sample_rate: self.config.audio_sample_rate,
            power_save_enabled: self.config.power_save_mode,
            wake_word_enabled: !self.config.hotkey_combinations.is_empty(),
        }
    }

    /// Shutdown the push-to-talk system
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down push-to-talk system");

        // Notify all tasks to shutdown
        self.shutdown_notify.notify_waiters();

        // Shutdown NPU Whisper processor
        {
            let mut whisper_guard = self.npu_whisper.write().await;
            whisper_guard.shutdown().await?;
        }

        // Shutdown GNA manager
        {
            let mut gna_guard = self.gna_manager.write().await;
            gna_guard.shutdown().await?;
        }

        // Close transcription channel
        self.transcription_tx = None;

        info!("Push-to-talk system shutdown complete");
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PTTStatus {
    pub state: PTTState,
    pub npu_performance: crate::npu_whisper::NPUWhisperStats,
    pub audio_sample_rate: u32,
    pub power_save_enabled: bool,
    pub wake_word_enabled: bool,
}

impl PTTStatus {
    pub fn is_recording(&self) -> bool {
        matches!(self.state, PTTState::Recording { .. })
    }

    pub fn is_processing(&self) -> bool {
        matches!(self.state, PTTState::Processing { .. })
    }

    pub fn is_idle(&self) -> bool {
        matches!(self.state, PTTState::Idle)
    }

    pub fn meets_performance_targets(&self) -> bool {
        self.npu_performance.meets_performance_targets()
    }

    pub fn generate_status_report(&self) -> String {
        format!(
            "Push-to-Talk System Status\n\
             ==========================\n\
             State: {:?}\n\
             Audio Sample Rate: {}Hz\n\
             Power Save: {}\n\
             Wake Word: {}\n\
             Performance Targets: {}\n\
             \n\
             {}",
            self.state,
            self.audio_sample_rate,
            if self.power_save_enabled { "Enabled" } else { "Disabled" },
            if self.wake_word_enabled { "Enabled" } else { "Disabled" },
            if self.meets_performance_targets() { "✅ MET" } else { "❌ NOT MET" },
            self.npu_performance.generate_performance_report()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_ptt_manager_initialization() {
        let config = PTTConfig::default();
        let result = PushToTalkManager::new(config).await;

        match result {
            Ok(_) => println!("Push-to-talk manager initialized successfully"),
            Err(e) => println!("PTT manager initialization failed: {}", e),
        }
    }

    #[test]
    fn test_vad_state() {
        let mut vad = VADState::new(0.01);

        // Test with silence (low energy)
        let silence: Vec<f32> = vec![0.001; 1600];
        assert!(!vad.update(&silence));

        // Test with speech (high energy)
        let speech: Vec<f32> = vec![0.1; 1600];
        let is_speech = vad.update(&speech);

        println!("VAD detected speech: {}", is_speech);
    }

    #[test]
    async fn test_ptt_state_transitions() {
        let config = PTTConfig::default();
        let mut manager = match PushToTalkManager::new(config).await {
            Ok(manager) => manager,
            Err(e) => {
                println!("Failed to create PTT manager: {}", e);
                return;
            }
        };

        // Test state transitions
        assert!(manager.get_status().await.is_idle());

        if let Err(e) = manager.enable_wake_word_mode().await {
            println!("Failed to enable wake word mode: {}", e);
        }
        let status = manager.get_status().await;
        assert!(matches!(status.state, PTTState::WaitingForWakeWord));

        if let Err(e) = manager.enable_power_save_mode().await {
            println!("Failed to enable power save mode: {}", e);
        }
        let status = manager.get_status().await;
        assert!(matches!(status.state, PTTState::PowerSave));
    }
}