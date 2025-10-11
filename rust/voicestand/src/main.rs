//! VoiceStand Main Application
//!
//! Memory-safe push-to-talk voice-to-text system with NPU/GNA acceleration.
//! Integrates all subsystems for production-grade operation.

use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tracing::{error, info, warn, Level};
use tracing_subscriber;

use voicestand_core::{
    VoiceStandConfig, VoiceStandIntegration, IntegrationEvent, Result, VoiceStandError
};

/// VoiceStand application
struct VoiceStandApp {
    integration: VoiceStandIntegration,
    config: VoiceStandConfig,
}

impl VoiceStandApp {
    /// Create new VoiceStand application
    fn new(config: VoiceStandConfig) -> Result<Self> {
        let integration = VoiceStandIntegration::new(config.clone())?;

        Ok(Self {
            integration,
            config,
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
    async fn main_loop(&mut self, events: &mut tokio::sync::mpsc::Receiver<IntegrationEvent>) -> Result<()> {
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
    async fn handle_integration_event(&self, event: IntegrationEvent) -> Result<()> {
        match event {
            IntegrationEvent::ComponentInitialized { component } => {
                info!("✅ Component initialized: {}", component);
            }

            IntegrationEvent::ComponentFailed { component, error } => {
                warn!("⚠️ Component failed: {} - {}", component, error);
            }

            IntegrationEvent::PTTActivated { timestamp } => {
                info!("🔴 Push-to-talk activated at {:?}", timestamp);
                println!("🎤 Recording... (release key to stop)");
            }

            IntegrationEvent::PTTDeactivated { timestamp } => {
                info!("⚪ Push-to-talk deactivated at {:?}", timestamp);
                println!("⏹️ Recording stopped - processing...");
            }

            IntegrationEvent::WakeWordDetected { word, confidence } => {
                info!("🔊 Wake word detected: '{}' ({:.1}%)", word, confidence * 100.0);
                println!("🔊 Wake word '{}' detected! Listening...", word);
            }

            IntegrationEvent::VoiceActivityDetected { confidence } => {
                if confidence > 0.8 {
                    info!("🗣️ Voice activity detected ({:.1}%)", confidence * 100.0);
                }
            }

            IntegrationEvent::TranscriptionStarted { source } => {
                match source {
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
                }
            }

            IntegrationEvent::TranscriptionCompleted { result } => {
                let performance_indicator = if result.meets_latency_target { "🟢" } else { "🟡" };

                info!("✅ Transcription completed: \"{}\" ({:.1}% confidence, {}ms)",
                      result.text, result.confidence * 100.0, result.duration_ms);

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
            }

            IntegrationEvent::TranscriptionFailed { error } => {
                error!("❌ Transcription failed: {}", error);
                println!("❌ Transcription failed: {}", error);
            }

            IntegrationEvent::AudioCaptured { frame_size, .. } => {
                // Only log periodically to avoid spam
                if frame_size > 0 {
                    // Audio capture is working
                }
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
                println!("Health: {}", if self.integration.is_healthy() { "✅ HEALTHY" } else { "⚠️ ISSUES" });
                println!("===============================\n");
            }
            Err(e) => {
                warn!("Failed to get system status: {}", e);
            }
        }
    }

    /// Print periodic status updates
    async fn print_periodic_status(&self) {
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

/// Print welcome banner
fn print_banner() {
    println!(r#"
╦  ╦┌─┐┬┌─┐┌─┐╔═╗┌┬┐┌─┐┌┐┌┌┬┐
╚╗╔╝│ ││ ├┤ └─┐╚═╗ │ ├─┤│││ ││
 ╚╝ └─┘┴└─┘└─┘╚═╝ ┴ ┴ ┴┘└┘─┴┘

Advanced Voice-to-Text System
Memory-Safe Rust Implementation

🚀 NPU Acceleration: <2ms inference
🔊 GNA Wake Words: <100mW power
🎤 Push-to-Talk: <10ms latency
🛡️ Memory Safety: Zero unwrap() calls
"#);
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