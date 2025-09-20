// GNA CLI Binary for VoiceStand
// Command-line interface for GNA wake word detection and NPU integration

use env_logger;
use log::{info, error};
use tokio;
use anyhow::Result;

use voicestand_intel::gna_cli::run_cli;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    info!("🚀 VoiceStand GNA CLI v1.0.0");
    info!("Intel Meteor Lake GNA Wake Word Detection System");

    // Run the CLI
    match run_cli().await {
        Ok(()) => {
            info!("✅ GNA CLI completed successfully");
            Ok(())
        }
        Err(e) => {
            error!("❌ GNA CLI failed: {}", e);
            std::process::exit(1);
        }
    }
}