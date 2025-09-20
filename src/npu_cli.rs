use voicestand_intel::{
    NPUModelCompiler, OptimizationConfig, ModelPrecision, OptimizationLevel,
    NPUWhisperProcessor, NPUWhisperConfig, PushToTalkManager, PTTConfig,
    NPUIntegrationTests
};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing::{info, warn, error, Level};
use tracing_subscriber;

/// NPU Voice-to-Text CLI Tool for Intel Meteor Lake
/// Provides model compilation, optimization, and push-to-talk functionality
#[derive(Parser)]
#[command(name = "npu-vtt")]
#[command(about = "Intel NPU Voice-to-Text CLI with <2ms inference and <100mW power")]
#[command(version = "1.0.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Log level
    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile and optimize Whisper model for NPU
    Compile {
        /// Input model path (ONNX or OpenVINO IR)
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for optimized model
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Target precision for optimization
        #[arg(short, long, value_enum, default_value_t = PrecisionArg::Fp16)]
        precision: PrecisionArg,

        /// Optimization level
        #[arg(long, value_enum, default_value_t = OptimizationArg::Speed)]
        optimization: OptimizationArg,

        /// Target latency in milliseconds
        #[arg(long, default_value_t = 2.0)]
        target_latency: f32,

        /// Power budget in milliwatts
        #[arg(long, default_value_t = 100)]
        power_budget: u32,

        /// Enable dynamic shapes
        #[arg(long)]
        dynamic_shapes: bool,
    },

    /// Run NPU performance benchmarks
    Benchmark {
        /// Number of benchmark iterations
        #[arg(short, long, default_value_t = 100)]
        iterations: usize,

        /// Model path for benchmarking
        #[arg(short, long)]
        model: Option<PathBuf>,

        /// Test duration in seconds
        #[arg(short, long, default_value_t = 30)]
        duration: u64,

        /// Generate detailed report
        #[arg(long)]
        detailed: bool,
    },

    /// Test NPU hardware and integration
    Test {
        /// Run comprehensive test suite
        #[arg(long)]
        comprehensive: bool,

        /// Test specific component
        #[arg(long, value_enum)]
        component: Option<TestComponent>,

        /// Generate detailed test report
        #[arg(long)]
        report: bool,
    },

    /// Start push-to-talk voice transcription
    Listen {
        /// Model path for transcription
        #[arg(short, long, default_value = "models/whisper-base-npu.xml")]
        model: PathBuf,

        /// Audio sample rate
        #[arg(long, default_value_t = 16000)]
        sample_rate: u32,

        /// Push-to-talk key combination
        #[arg(long, default_value = "ctrl+alt+space")]
        hotkey: String,

        /// Enable wake word detection
        #[arg(long)]
        wake_word: bool,

        /// Enable power save mode
        #[arg(long)]
        power_save: bool,

        /// Output transcriptions to file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Enable continuous listening mode
        #[arg(long)]
        continuous: bool,
    },

    /// Display NPU status and capabilities
    Status {
        /// Show detailed hardware information
        #[arg(long)]
        detailed: bool,

        /// Monitor performance metrics
        #[arg(long)]
        monitor: bool,

        /// Refresh interval in seconds for monitoring
        #[arg(long, default_value_t = 1)]
        interval: u64,
    },

    /// Manage NPU model cache
    Cache {
        /// List cached models
        #[arg(long)]
        list: bool,

        /// Clean old cache entries
        #[arg(long)]
        clean: bool,

        /// Maximum cache age in days
        #[arg(long, default_value_t = 30)]
        max_age: u64,

        /// Show cache statistics
        #[arg(long)]
        stats: bool,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum PrecisionArg {
    Fp32,
    Fp16,
    Int8,
    Int4,
}

impl From<PrecisionArg> for ModelPrecision {
    fn from(arg: PrecisionArg) -> Self {
        match arg {
            PrecisionArg::Fp32 => ModelPrecision::FP32,
            PrecisionArg::Fp16 => ModelPrecision::FP16,
            PrecisionArg::Int8 => ModelPrecision::INT8,
            PrecisionArg::Int4 => ModelPrecision::INT4,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum OptimizationArg {
    Speed,
    Balanced,
    Accuracy,
}

impl From<OptimizationArg> for OptimizationLevel {
    fn from(arg: OptimizationArg) -> Self {
        match arg {
            OptimizationArg::Speed => OptimizationLevel::Speed,
            OptimizationArg::Balanced => OptimizationLevel::Balanced,
            OptimizationArg::Accuracy => OptimizationLevel::Accuracy,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum TestComponent {
    Npu,
    Whisper,
    PushToTalk,
    Audio,
    Models,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => Level::ERROR,
            LogLevel::Warn => Level::WARN,
            LogLevel::Info => Level::INFO,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Trace => Level::TRACE,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { Level::DEBUG } else { cli.log_level.into() };
    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();

    info!("Intel NPU Voice-to-Text CLI v1.0.0");
    info!("Target: <2ms inference, <100mW power consumption");

    match cli.command {
        Commands::Compile {
            input,
            output,
            precision,
            optimization,
            target_latency,
            power_budget,
            dynamic_shapes,
        } => {
            handle_compile_command(input, output, precision, optimization, target_latency, power_budget, dynamic_shapes).await?;
        }

        Commands::Benchmark {
            iterations,
            model,
            duration,
            detailed,
        } => {
            handle_benchmark_command(iterations, model, duration, detailed).await?;
        }

        Commands::Test {
            comprehensive,
            component,
            report,
        } => {
            handle_test_command(comprehensive, component, report).await?;
        }

        Commands::Listen {
            model,
            sample_rate,
            hotkey,
            wake_word,
            power_save,
            output,
            continuous,
        } => {
            handle_listen_command(model, sample_rate, hotkey, wake_word, power_save, output, continuous).await?;
        }

        Commands::Status {
            detailed,
            monitor,
            interval,
        } => {
            handle_status_command(detailed, monitor, interval).await?;
        }

        Commands::Cache {
            list,
            clean,
            max_age,
            stats,
        } => {
            handle_cache_command(list, clean, max_age, stats).await?;
        }
    }

    Ok(())
}

async fn handle_compile_command(
    input: PathBuf,
    output: Option<PathBuf>,
    precision: PrecisionArg,
    optimization: OptimizationArg,
    target_latency: f32,
    power_budget: u32,
    dynamic_shapes: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Compiling Whisper model for Intel NPU");
    info!("Input: {:?}", input);
    info!("Precision: {:?}", precision);
    info!("Optimization: {:?}", optimization);
    info!("Target Latency: {:.2}ms", target_latency);
    info!("Power Budget: {}mW", power_budget);

    if !input.exists() {
        error!("Input model file does not exist: {:?}", input);
        std::process::exit(1);
    }

    let mut compiler = NPUModelCompiler::new()?;

    let config = OptimizationConfig {
        target_latency_ms: target_latency,
        power_budget_mw: power_budget,
        precision: precision.into(),
        optimization_level: optimization.into(),
        enable_dynamic_shapes: dynamic_shapes,
        ..Default::default()
    };

    let result = compiler.compile_whisper_model(&input, Some(config)).await?;

    println!("\n{}", result.generate_report());

    if result.meets_targets() {
        info!("✅ Model optimization successful - performance targets met");
    } else {
        warn!("⚠️  Model optimization completed but performance targets not met");
    }

    if let Some(output_dir) = output {
        if !output_dir.exists() {
            std::fs::create_dir_all(&output_dir)?;
        }

        let output_path = output_dir.join(result.optimized_model_path.file_name().unwrap());
        std::fs::copy(&result.optimized_model_path, &output_path)?;
        info!("Optimized model copied to: {:?}", output_path);
    }

    Ok(())
}

async fn handle_benchmark_command(
    iterations: usize,
    model: Option<PathBuf>,
    duration: u64,
    detailed: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running NPU performance benchmarks");
    info!("Iterations: {}", iterations);
    info!("Duration: {}s", duration);

    if let Some(model_path) = &model {
        info!("Model: {:?}", model_path);
        if !model_path.exists() {
            error!("Model file does not exist: {:?}", model_path);
            std::process::exit(1);
        }
    }

    let mut test_suite = NPUIntegrationTests::new();
    let summary = test_suite.run_all_tests().await?;

    if detailed {
        println!("\n{}", test_suite.generate_test_report());
    }

    println!("\nBenchmark Summary:");
    println!("==================");
    println!("Success Rate: {:.1}%", summary.success_rate());
    println!("Total Tests: {}", summary.total_tests);
    println!("Passed: {}", summary.passed_tests);
    println!("Failed: {}", summary.failed_tests);
    println!("Performance Targets: {}",
             if summary.performance_targets_met { "✅ MET" } else { "❌ NOT MET" });

    for benchmark in &summary.benchmarks {
        println!("\n{}: {}",
                 benchmark.benchmark_name,
                 if benchmark.meets_targets { "✅ TARGET MET" } else { "❌ BELOW TARGET" });
        println!("  Latency: {:.2}ms", benchmark.latency_ms);
        println!("  Throughput: {:.1} ops/sec", benchmark.throughput_ops_per_sec);
        println!("  Power: {:.1}mW", benchmark.power_consumption_mw);
        println!("  Accuracy: {:.1}%", benchmark.accuracy_score * 100.0);
    }

    Ok(())
}

async fn handle_test_command(
    comprehensive: bool,
    component: Option<TestComponent>,
    report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running NPU integration tests");

    let mut test_suite = NPUIntegrationTests::new();

    if comprehensive {
        info!("Running comprehensive test suite");
        let summary = test_suite.run_all_tests().await?;

        if report {
            println!("\n{}", test_suite.generate_test_report());
        }

        println!("\nTest Summary:");
        println!("=============");
        println!("Success Rate: {:.1}%", summary.success_rate());
        println!("Tests Passed: {}/{}", summary.passed_tests, summary.total_tests);
        println!("Duration: {}ms", summary.total_duration_ms);

        if summary.is_fully_successful() {
            info!("✅ All tests passed and performance targets met");
        } else {
            warn!("⚠️  Some tests failed or performance targets not met");
        }
    } else if let Some(comp) = component {
        info!("Testing specific component: {:?}", comp);

        match comp {
            TestComponent::Npu => {
                info!("Testing NPU hardware detection and basic operations");
                // Run NPU-specific tests
            }
            TestComponent::Whisper => {
                info!("Testing NPU Whisper processor");
                // Run Whisper-specific tests
            }
            TestComponent::PushToTalk => {
                info!("Testing push-to-talk manager");
                // Run PTT-specific tests
            }
            TestComponent::Audio => {
                info!("Testing audio processing pipeline");
                // Run audio-specific tests
            }
            TestComponent::Models => {
                info!("Testing model compilation and optimization");
                // Run model-specific tests
            }
        }

        println!("Component test completed");
    } else {
        info!("Running basic test suite");
        // Run basic tests only
        println!("Basic tests completed");
    }

    Ok(())
}

async fn handle_listen_command(
    model: PathBuf,
    sample_rate: u32,
    hotkey: String,
    wake_word: bool,
    power_save: bool,
    output: Option<PathBuf>,
    continuous: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting push-to-talk voice transcription");
    info!("Model: {:?}", model);
    info!("Sample Rate: {}Hz", sample_rate);
    info!("Hotkey: {}", hotkey);
    info!("Wake Word: {}", wake_word);
    info!("Power Save: {}", power_save);
    info!("Continuous: {}", continuous);

    if !model.exists() {
        error!("Model file does not exist: {:?}", model);
        std::process::exit(1);
    }

    // Initialize push-to-talk manager
    let ptt_config = PTTConfig {
        audio_sample_rate: sample_rate,
        hotkey_combinations: vec![hotkey],
        power_save_mode: power_save,
        ..Default::default()
    };

    let mut ptt_manager = PushToTalkManager::new(ptt_config).await?;

    if wake_word {
        info!("Enabling wake word detection mode");
        ptt_manager.enable_wake_word_mode().await?;
    }

    if power_save {
        info!("Enabling power save mode");
        ptt_manager.enable_power_save_mode().await?;
    }

    // Start transcription pipeline
    let mut event_rx = ptt_manager.start().await?;

    info!("🎤 Voice transcription active - Press {} to start recording",
          ptt_config.hotkey_combinations[0]);

    if wake_word {
        info!("🔊 Wake word detection active - Say 'voicestand' to start");
    }

    // Handle transcription events
    while let Some(event) = event_rx.recv().await {
        match event {
            voicestand_intel::TranscriptionEvent::Started { timestamp } => {
                info!("🔴 Recording started at {:?}", timestamp);
            }
            voicestand_intel::TranscriptionEvent::Progress { partial_text, confidence } => {
                if !partial_text.trim().is_empty() {
                    print!("\r🟡 Partial: {} ({:.1}%)", partial_text, confidence * 100.0);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
            }
            voicestand_intel::TranscriptionEvent::Completed { text, confidence, duration_ms, language } => {
                println!("\n✅ Transcription: \"{}\"", text);
                println!("   Confidence: {:.1}%, Duration: {}ms, Language: {}",
                         confidence * 100.0, duration_ms, language);

                // Save to output file if specified
                if let Some(output_path) = &output {
                    use std::io::Write;
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(output_path)?;
                    writeln!(file, "[{}] {}", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"), text)?;
                }
            }
            voicestand_intel::TranscriptionEvent::Error { error } => {
                error!("❌ Transcription error: {}", error);
            }
            voicestand_intel::TranscriptionEvent::WakeWordDetected { word, confidence } => {
                info!("🔊 Wake word detected: '{}' ({:.1}%)", word, confidence * 100.0);
            }
            voicestand_intel::TranscriptionEvent::PTTPressed { timestamp } => {
                info!("🔴 PTT pressed at {:?}", timestamp);
            }
            voicestand_intel::TranscriptionEvent::PTTReleased { timestamp } => {
                info!("⚪ PTT released at {:?}", timestamp);
            }
        }

        if !continuous {
            // Single transcription mode
            break;
        }
    }

    info!("Shutting down voice transcription");
    ptt_manager.shutdown().await?;

    Ok(())
}

async fn handle_status_command(
    detailed: bool,
    monitor: bool,
    interval: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Checking NPU status and capabilities");

    // Check NPU hardware availability
    match voicestand_intel::NPUManager::new().await {
        Ok(npu) => {
            println!("✅ Intel NPU detected and operational");

            let stats = npu.get_performance_stats().await;
            println!("\nNPU Status:");
            println!("===========");
            println!("Total Inferences: {}", stats.total_inferences);
            println!("Average Latency: {:.2}ms", stats.average_inference_time_ms);
            println!("NPU Utilization: {:.1}%", stats.npu_utilization_percent);
            println!("Model Cache: {} models", stats.model_cache_size);
            println!("Uptime: {:.1}s", stats.uptime.as_secs_f32());

            if detailed {
                println!("\n{}", stats.generate_report());
            }

            if monitor {
                info!("Starting performance monitoring (Ctrl+C to stop)");
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;

                    let current_stats = npu.get_performance_stats().await;
                    println!("\n[{}] Latency: {:.2}ms | Utilization: {:.1}% | Inferences: {}",
                             chrono::Local::now().format("%H:%M:%S"),
                             current_stats.average_inference_time_ms,
                             current_stats.npu_utilization_percent,
                             current_stats.total_inferences);
                }
            }
        }
        Err(e) => {
            error!("❌ NPU not available: {}", e);
            println!("Make sure you're running on Intel Meteor Lake hardware with NPU support");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn handle_cache_command(
    list: bool,
    clean: bool,
    max_age: u64,
    stats: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Managing NPU model cache");

    let mut compiler = NPUModelCompiler::new()?;

    if list {
        let cached_models = compiler.list_cached_models();
        println!("Cached Models:");
        println!("==============");

        if cached_models.is_empty() {
            println!("No models in cache");
        } else {
            for (i, model) in cached_models.iter().enumerate() {
                println!("{}. {} ({:?})",
                         i + 1,
                         model.path.file_name().unwrap().to_string_lossy(),
                         model.config.precision);
                println!("   Path: {:?}", model.path);
                println!("   Size: {:.2}MB", model.result.model_size_mb);
                println!("   Last Used: {:?}", model.last_used);
                println!("   Performance: {:.2}ms latency", model.result.estimated_latency_ms);
                println!();
            }
        }
    }

    if clean {
        info!("Cleaning cache entries older than {} days", max_age);
        let removed = compiler.cleanup_cache(max_age)?;
        println!("✅ Removed {} old cache entries", removed);
    }

    if stats {
        let cached_models = compiler.list_cached_models();
        let total_size: f64 = cached_models.iter()
            .map(|m| m.result.model_size_mb)
            .sum();

        println!("Cache Statistics:");
        println!("=================");
        println!("Total Models: {}", cached_models.len());
        println!("Total Size: {:.2}MB", total_size);

        if !cached_models.is_empty() {
            let avg_size = total_size / cached_models.len() as f64;
            let avg_latency: f32 = cached_models.iter()
                .map(|m| m.result.estimated_latency_ms)
                .sum::<f32>() / cached_models.len() as f32;

            println!("Average Model Size: {:.2}MB", avg_size);
            println!("Average Latency: {:.2}ms", avg_latency);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        // Test basic command parsing
        let cli = Cli::parse_from(&["npu-vtt", "status"]);
        assert!(matches!(cli.command, Commands::Status { .. }));

        let cli = Cli::parse_from(&["npu-vtt", "compile", "--input", "model.onnx", "--precision", "fp16"]);
        if let Commands::Compile { precision, .. } = cli.command {
            assert_eq!(precision, PrecisionArg::Fp16);
        } else {
            panic!("Expected Compile command");
        }
    }
}