//! VoiceStand Main Application
//!
//! Memory-safe push-to-talk voice-to-text system with NPU/GNA acceleration.
//! Integrates all subsystems for production-grade operation.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::signal;
use tracing::{error, info, warn, Level};
use tracing_subscriber;

use voicestand_asr::{
    read_wav_16khz_mono, DecodeOptions, ModelSpec, SpeechBackend, WhisperCppBackend,
};
use voicestand_core::{
    IntegrationEvent, Result, VoiceStandConfig, VoiceStandError, VoiceStandIntegration,
};
use voicestand_text::{TextSink, XdotoolTextSink};

/// VoiceStand application
struct VoiceStandApp {
    integration: VoiceStandIntegration,
    config: VoiceStandConfig,
    text_sink: Option<Box<dyn TextSink>>,
}

impl VoiceStandApp {
    /// Create new VoiceStand application
    fn new(config: VoiceStandConfig) -> Result<Self> {
        let integration = VoiceStandIntegration::new(config.clone())?;

        Ok(Self {
            integration,
            config,
            text_sink: None,
        })
    }

    /// Initialize and start the application
    async fn run(&mut self) -> Result<()> {
        info!("🚀 Starting VoiceStand - Advanced Voice-to-Text System");
        info!("Target Performance: <2ms NPU inference, <100mW GNA power, <10ms latency");

        // Initialize all subsystems
        match self.integration.initialize().await {
            Ok(()) => {
                info!("✅ All subsystems initialized successfully");
            }
            Err(e) => {
                error!("❌ Initialization failed: {}", e);
                return Err(e);
            }
        }

        let sink = XdotoolTextSink::new().map_err(|error| {
            VoiceStandError::system(format!("Text sink initialization failed: {error}"))
        })?;
        info!(backend = sink.name(), "Focused-application text sink ready");
        self.text_sink = Some(Box::new(sink));

        // Start the integration system
        let mut events = match self.integration.start().await {
            Ok(events) => {
                info!("✅ VoiceStand system started - ready for voice commands");
                events
            }
            Err(e) => {
                error!("❌ Failed to start system: {}", e);
                return Err(e);
            }
        };

        // Print system status
        self.print_system_status().await;

        // Main application loop
        self.main_loop(&mut events).await?;

        Ok(())
    }

    /// Main application event loop
    async fn main_loop(
        &mut self,
        events: &mut tokio::sync::mpsc::Receiver<IntegrationEvent>,
    ) -> Result<()> {
        info!("🎤 VoiceStand is ready for voice commands");
        info!("Press Ctrl+Alt+Space to activate, or say 'voicestand' for wake word activation");

        // Setup graceful shutdown
        let mut shutdown_signal = Box::pin(signal::ctrl_c());

        loop {
            tokio::select! {
                // Handle integration events
                event = events.recv() => {
                    match event {
                        Some(event) => self.handle_integration_event(event).await?,
                        None => {
                            warn!("Integration event stream ended");
                            break;
                        }
                    }
                }

                // Handle shutdown signal
                _ = &mut shutdown_signal => {
                    info!("🛑 Received shutdown signal");
                    break;
                }

                // Periodic status updates
                _ = tokio::time::sleep(Duration::from_secs(30)) => {
                    self.print_periodic_status().await;
                }
            }
        }

        // Graceful shutdown
        self.shutdown().await?;

        Ok(())
    }

    /// Handle integration events
    async fn handle_integration_event(&mut self, event: IntegrationEvent) -> Result<()> {
        match event {
            IntegrationEvent::ComponentInitialized { component } => {
                info!("✅ Component initialized: {}", component);
            }

            IntegrationEvent::ComponentFailed { component, error } => {
                warn!("⚠️ Component failed: {} - {}", component, error);
            }

            IntegrationEvent::PTTActivated { timestamp } => {
                info!("🔴 Push-to-talk activated at {:?}", timestamp);
                self.integration.begin_ptt()?;
                if let Some(sink) = &mut self.text_sink {
                    sink.begin()
                        .map_err(|error| VoiceStandError::system(error.to_string()))?;
                }
                println!("🎤 Recording... (release key to stop)");
            }

            IntegrationEvent::PTTDeactivated { timestamp } => {
                info!("⚪ Push-to-talk deactivated at {:?}", timestamp);
                println!("⏹️ Recording stopped - processing...");
                self.integration.end_ptt().await?;
            }

            IntegrationEvent::WakeWordDetected { word, confidence } => {
                info!(
                    "🔊 Wake word detected: '{}' ({:.1}%)",
                    word,
                    confidence * 100.0
                );
                println!("🔊 Wake word '{}' detected! Listening...", word);
            }

            IntegrationEvent::VoiceActivityDetected { confidence } => {
                if confidence > 0.8 {
                    info!("🗣️ Voice activity detected ({:.1}%)", confidence * 100.0);
                }
            }

            IntegrationEvent::TranscriptionStarted { source } => match source {
                voicestand_core::TranscriptionSource::NPU => {
                    info!("🚀 NPU transcription started");
                    println!("🚀 Using NPU acceleration...");
                }
                voicestand_core::TranscriptionSource::CPU => {
                    info!("💻 CPU transcription started");
                    println!("💻 Using CPU fallback...");
                }
                voicestand_core::TranscriptionSource::Hybrid => {
                    info!("⚡ Hybrid transcription started");
                    println!("⚡ Using hybrid processing...");
                }
            },

            IntegrationEvent::TranscriptionCompleted { result } => {
                let performance_indicator = if result.meets_latency_target {
                    "🟢"
                } else {
                    "🟡"
                };

                info!(
                    "✅ Transcription completed: \"{}\" ({:.1}% confidence, {}ms)",
                    result.text,
                    result.confidence * 100.0,
                    result.duration_ms
                );

                println!("\n{} Transcription Result:", performance_indicator);
                println!("📝 Text: \"{}\"", result.text);
                println!("🎯 Confidence: {:.1}%", result.confidence * 100.0);
                println!("⏱️ Duration: {}ms", result.duration_ms);
                println!("🌍 Language: {}", result.language);

                if result.meets_latency_target {
                    println!("✅ Performance target met (<10ms)");
                } else {
                    println!("⚠️ Performance target exceeded (>10ms)");
                }
                println!();
                if let Some(sink) = &mut self.text_sink {
                    sink.commit(&result.text)
                        .map_err(|error| VoiceStandError::system(error.to_string()))?;
                }
            }

            IntegrationEvent::PartialTranscription {
                text,
                stable_prefix_bytes,
                confidence,
            } => {
                let (stable, changing) = text.split_at(stable_prefix_bytes.min(text.len()));
                println!("… {}[{}] ({:.0}%)", stable, changing, confidence * 100.0);
                if let Some(sink) = &mut self.text_sink {
                    sink.partial(&text)
                        .map_err(|error| VoiceStandError::system(error.to_string()))?;
                }
            }

            IntegrationEvent::TranscriptionFailed { error } => {
                error!("❌ Transcription failed: {}", error);
                println!("❌ Transcription failed: {}", error);
                if let Some(sink) = &mut self.text_sink {
                    sink.cancel()
                        .map_err(|error| VoiceStandError::system(error.to_string()))?;
                }
            }

            IntegrationEvent::AudioCaptured { frame_size, .. } => {
                // Only log periodically to avoid spam
                if frame_size > 0 {
                    // Audio capture is working
                }
            }

            IntegrationEvent::LiveAudioFrame { samples } => {
                self.integration.process_audio_frame(&samples).await?;
            }

            IntegrationEvent::SystemError { error } => {
                error!("🚨 System error: {}", error);
                println!("🚨 System error: {}", error);
            }

            IntegrationEvent::ShutdownInitiated => {
                info!("🛑 System shutdown initiated");
                println!("🛑 VoiceStand shutting down...");
            }

            _ => {
                // Handle other events as needed
            }
        }

        Ok(())
    }

    /// Print system status
    async fn print_system_status(&self) -> () {
        match self.integration.get_status().await {
            Ok(status) => {
                println!("\n=== VoiceStand System Status ===");
                println!("State: {:?}", status.state);
                println!("Uptime: {:.1}s", status.uptime.as_secs_f32());
                println!("Components Active: {}", status.components_active);
                println!("Components Failed: {}", status.components_failed);
                println!("Capabilities: {}", status.capabilities.join(", "));
                println!(
                    "Health: {}",
                    if self.integration.is_healthy() {
                        "✅ HEALTHY"
                    } else {
                        "⚠️ ISSUES"
                    }
                );
                println!("===============================\n");
            }
            Err(e) => {
                warn!("Failed to get system status: {}", e);
            }
        }
    }

    /// Print periodic status updates
    async fn print_periodic_status(&self) {
        if let Some(metrics) = self.integration.asr_metrics() {
            info!(
                started = metrics.started,
                completed = metrics.completed,
                cancelled = metrics.cancelled,
                failed = metrics.failed,
                average_latency_ms = metrics.average_latency_ms,
                "ASR runtime metrics"
            );
        }
        // Print status in development builds
        #[cfg(debug_assertions)]
        self.print_system_status().await;
    }

    /// Shutdown the application
    async fn shutdown(&mut self) -> Result<()> {
        info!("🛑 Shutting down VoiceStand application");

        // Shutdown integration system
        if let Err(e) = self.integration.shutdown().await {
            error!("Error during integration shutdown: {}", e);
        }

        info!("✅ VoiceStand application shutdown complete");
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let log_level = if std::env::var("RUST_LOG").is_ok() {
        Level::DEBUG
    } else {
        Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if matches!(
        arguments.first().map(String::as_str),
        Some("--version" | "-V")
    ) {
        println!("voicestand {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if matches!(arguments.first().map(String::as_str), Some("--check")) {
        let config = VoiceStandConfig::load()?;
        config.validate()?;
        let model = PathBuf::from(&config.speech.model_path);
        if !model.is_file() {
            return Err(VoiceStandError::config(format!(
                "model is missing: {}; run scripts/install-model.sh",
                model.display()
            )));
        }
        let sink = XdotoolTextSink::new()
            .map_err(|error| VoiceStandError::system(format!("Text sink check failed: {error}")))?;
        println!(
            "VoiceStand configuration, model, and {} text output are ready: {}",
            sink.name(),
            model.display()
        );
        return Ok(());
    }
    if matches!(arguments.first().map(String::as_str), Some("--smoke-test")) {
        let wav = arguments
            .get(1)
            .map(PathBuf::from)
            .ok_or_else(|| VoiceStandError::config("usage: voicestand --smoke-test WAV [MODEL]"))?;
        let mut config = VoiceStandConfig::load()?;
        if let Some(model) = arguments.get(2) {
            config.speech.model_path = model.clone();
        }
        return run_smoke_test(&config, &wav);
    }

    // Print welcome banner
    print_banner();

    // Load configuration
    let config = VoiceStandConfig::load().unwrap_or_else(|e| {
        warn!("Failed to load config: {} - using defaults", e);
        VoiceStandConfig::default()
    });

    // Create and run application
    let mut app = VoiceStandApp::new(config)?;

    match app.run().await {
        Ok(()) => {
            info!("VoiceStand application completed successfully");
        }
        Err(e) => {
            error!("VoiceStand application failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn run_smoke_test(config: &VoiceStandConfig, wav_path: &PathBuf) -> Result<()> {
    config.validate()?;
    let audio = read_wav_16khz_mono(wav_path)?;
    // The smoke test is deliberately CPU-only so it validates the universally
    // available production fallback independently of optional acceleration.
    let mut backend = WhisperCppBackend::cpu();
    let load_started = Instant::now();
    backend.load(&ModelSpec::new("smoke-test", &config.speech.model_path))?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
    let decode_started = Instant::now();
    let transcript = backend.transcribe(
        &audio,
        &DecodeOptions {
            language: match config.speech.language.as_str() {
                "" | "auto" => None,
                value => Some(value.to_string()),
            },
            thread_count: config.speech.num_threads.clamp(1, 4),
            ..DecodeOptions::default()
        },
    )?;
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let audio_seconds = audio.len() as f64 / 16_000.0;
    println!("{}", transcript.text);
    eprintln!("model_load_ms={load_ms:.1}");
    eprintln!("decode_ms={:.1}", decode_seconds * 1_000.0);
    eprintln!(
        "real_time_factor={:.3}",
        decode_seconds / audio_seconds.max(f64::EPSILON)
    );
    eprintln!("confidence={:.3}", transcript.confidence.unwrap_or(0.0));
    Ok(())
}

/// Print welcome banner
fn print_banner() {
    println!(
        r#"
╦  ╦┌─┐┬┌─┐┌─┐╔═╗┌┬┐┌─┐┌┐┌┌┬┐
╚╗╔╝│ ││ ├┤ └─┐╚═╗ │ ├─┤│││ ││
 ╚╝ └─┘┴└─┘└─┘╚═╝ ┴ ┴ ┴┘└┘─┴┘

Advanced Voice-to-Text System
Memory-Safe Rust Implementation

🚀 NPU Acceleration: <2ms inference
🔊 GNA Wake Words: <100mW power
🎤 Push-to-Talk: <10ms latency
🛡️ Memory Safety: Zero unwrap() calls
"#
    );
}

/// Signal handler for graceful shutdown
async fn handle_shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C");
        },
        _ = terminate => {
            info!("Received SIGTERM");
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_creation() {
        let config = VoiceStandConfig::default();
        let app = VoiceStandApp::new(config);
        assert!(app.is_ok());
    }

    #[tokio::test]
    async fn test_banner_display() {
        // Test that banner function doesn't panic
        print_banner();
    }

    #[tokio::test]
    async fn test_shutdown_signal() {
        // Test signal handling setup (quick test)
        tokio::select! {
            _ = handle_shutdown_signal() => {
                // Should not complete in test
                panic!("Unexpected signal");
            },
            _ = tokio::time::sleep(Duration::from_millis(10)) => {
                // Expected path - signal handler is set up correctly
            }
        }
    }
}
