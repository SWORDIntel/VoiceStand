use voicestand_intel::{
    IntelAcceleration,
    NPUWorkloadType,
    OptimizationProfile,
    WorkloadType,
    SIMDAudioProcessor,
    ThermalPolicy,
};
use voicestand_core::{Result, PerformanceMonitor};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{info, warn, error, Level};
use tracing_subscriber::FmtSubscriber;

/// Comprehensive demo of Intel Meteor Lake hardware acceleration for VoiceStand
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("Setting default subscriber failed");

    info!("🚀 VoiceStand Intel Acceleration Demo");
    info!("====================================");

    // Initialize Intel acceleration system
    let intel_accel = match IntelAcceleration::new().await {
        Ok(accel) => accel,
        Err(e) => {
            error!("Failed to initialize Intel acceleration: {}", e);
            return Err(e);
        }
    };

    let capabilities = intel_accel.capabilities();
    info!("Intel Hardware Detected:");
    info!("  NPU: {} ({:.1} TOPS)", capabilities.has_npu, capabilities.npu_tops);
    info!("  GNA: {} ({:.1}mW)", capabilities.has_gna, capabilities.gna_power_mw);
    info!("  AVX2: {}", capabilities.has_avx2);
    info!("  P-cores: {}, E-cores: {}", capabilities.p_cores, capabilities.e_cores);

    // Demo 1: Hardware Detection and Capability Assessment
    demo_hardware_detection(&intel_accel).await?;

    // Demo 2: NPU Acceleration for Voice Processing
    demo_npu_acceleration(&intel_accel).await?;

    // Demo 3: GNA Ultra-Low Power Wake Word Detection
    demo_gna_wake_word_detection(&intel_accel).await?;

    // Demo 4: CPU Optimization for Hybrid Architecture
    demo_cpu_optimization(&intel_accel).await?;

    // Demo 5: SIMD Audio Processing Acceleration
    demo_simd_audio_processing().await?;

    // Demo 6: Thermal Management and Adaptive Performance
    demo_thermal_management(&intel_accel).await?;

    // Demo 7: Real-time Performance Monitoring
    demo_performance_monitoring(&intel_accel).await?;

    info!("🏁 Intel Acceleration Demo Complete");

    // Graceful shutdown
    intel_accel.shutdown().await?;

    Ok(())
}

/// Demo 1: Hardware Detection and Capability Assessment
async fn demo_hardware_detection(intel_accel: &IntelAcceleration) -> Result<()> {
    info!("\n📊 Demo 1: Hardware Detection");
    info!("==============================");

    let capabilities = intel_accel.capabilities();

    // Generate comprehensive hardware report
    let hardware_report = capabilities.generate_report();
    info!("Hardware Report:\n{}", hardware_report);

    // Check VoiceStand compatibility
    if capabilities.meets_minimum_requirements() {
        info!("✅ System meets VoiceStand minimum requirements");
    } else {
        warn!("⚠️ System may not meet optimal performance requirements");
    }

    // Assess optimization opportunities
    info!("Optimization Opportunities:");
    if capabilities.has_npu {
        info!("  🧠 NPU available for AI workload acceleration");
    }
    if capabilities.has_gna {
        info!("  🔋 GNA available for ultra-low power voice detection");
    }
    if capabilities.has_avx2 {
        info!("  ⚡ AVX2 SIMD available for 8x parallel audio processing");
    }
    if capabilities.p_cores >= 6 {
        info!("  🏭 {} P-cores available for real-time audio processing", capabilities.p_cores);
    }

    Ok(())
}

/// Demo 2: NPU Acceleration for Voice Processing
async fn demo_npu_acceleration(intel_accel: &IntelAcceleration) -> Result<()> {
    info!("\n🧠 Demo 2: NPU Acceleration");
    info!("===========================");

    let npu_manager = intel_accel.npu_manager.read();

    // Simulate loading a Whisper model (would be real model in production)
    info!("Loading Whisper model onto NPU...");
    // npu_manager.load_whisper_model("models/whisper-base.onnx").await?;

    // Optimize NPU for real-time inference
    // npu_manager.optimize_for_workload(NPUWorkloadType::RealTime).await?;

    // Simulate audio inference workload
    info!("Simulating NPU inference workload...");
    for i in 0..5 {
        // Generate synthetic mel spectrogram data
        let mel_spectrogram: Vec<f32> = (0..240000).map(|x| (x as f32 * 0.001).sin()).collect();

        info!("  Inference {}: Processing {} samples", i + 1, mel_spectrogram.len());

        // Simulate inference timing
        let start = std::time::Instant::now();
        sleep(Duration::from_millis(50 + i * 10)).await; // Simulate varying inference times
        let inference_time = start.elapsed();

        info!("    Completed in {:.2}ms", inference_time.as_secs_f32() * 1000.0);
    }

    // Get NPU performance statistics
    // let npu_stats = npu_manager.get_performance_stats().await;
    // info!("NPU Performance Report:\n{}", npu_stats.generate_report());

    Ok(())
}

/// Demo 3: GNA Ultra-Low Power Wake Word Detection
async fn demo_gna_wake_word_detection(intel_accel: &IntelAcceleration) -> Result<()> {
    info!("\n🔋 Demo 3: GNA Wake Word Detection");
    info!("==================================");

    let mut gna_controller = intel_accel.gna_controller.write();

    // Load VAD model onto GNA
    info!("Loading VAD model onto GNA...");
    let synthetic_vad_model = vec![0u8; 4096]; // 4KB synthetic model
    // gna_controller.load_vad_model(&synthetic_vad_model).await?;

    // Load wake word model
    info!("Loading wake word detection model for 'Hey VoiceStand'...");
    let synthetic_wake_word_model = vec![0u8; 8192]; // 8KB synthetic model
    // gna_controller.load_wake_word_model(&synthetic_wake_word_model, "hey_voicestand").await?;

    // Start continuous processing
    info!("Starting GNA continuous processing...");
    // gna_controller.start_continuous_processing().await?;

    // Simulate continuous audio processing
    info!("Simulating audio stream processing...");
    for frame in 0..20 {
        // Generate synthetic audio frame (20ms @ 16kHz = 320 samples)
        let audio_frame: Vec<f32> = (0..320).map(|i| {
            if frame == 10 {
                // Simulate voice activity in frame 10
                0.1 * (i as f32 * 0.1).sin() + 0.05 * rand::random::<f32>()
            } else {
                // Background noise
                0.01 * rand::random::<f32>()
            }
        }).collect();

        // Process through GNA
        // let detection_result = gna_controller.process_audio_frame(&audio_frame).await?;

        if frame == 10 {
            info!("  Frame {}: 🗣️ Voice activity detected (simulated)", frame);
        } else if frame % 5 == 0 {
            info!("  Frame {}: 🔇 Silence detected", frame);
        }

        sleep(Duration::from_millis(20)).await; // 20ms frame rate
    }

    // Get GNA performance statistics
    // let gna_stats = gna_controller.get_performance_stats().await;
    // info!("GNA Performance Report:\n{}", gna_stats.generate_report());

    Ok(())
}

/// Demo 4: CPU Optimization for Hybrid Architecture
async fn demo_cpu_optimization(intel_accel: &IntelAcceleration) -> Result<()> {
    info!("\n🏭 Demo 4: CPU Optimization");
    info!("============================");

    let mut cpu_optimizer = intel_accel.cpu_optimizer.write();

    // Optimize thread placement for VoiceStand workload
    info!("Optimizing thread placement for hybrid P-core/E-core architecture...");
    cpu_optimizer.optimize_thread_placement()?;

    // Demonstrate workload-specific thread affinity
    let workload_types = [
        WorkloadType::RealTimeAudio,
        WorkloadType::AudioProcessing,
        WorkloadType::SpeechInference,
        WorkloadType::BackgroundTask,
        WorkloadType::UserInterface,
    ];

    for workload_type in &workload_types {
        info!("Setting thread affinity for {:?} workload", workload_type);
        cpu_optimizer.set_thread_affinity_for_workload(*workload_type)?;

        // Simulate workload execution
        sleep(Duration::from_millis(100)).await;

        // Get performance stats
        let cpu_stats = cpu_optimizer.update_performance_counters()?;
        info!("  CPU Utilization: P-cores {:.1}%, E-cores {:.1}%",
              cpu_stats.p_core_utilization, cpu_stats.e_core_utilization);
    }

    // Monitor thermal conditions
    let thermal_status = cpu_optimizer.update_thermal_monitoring().await?;
    info!("Thermal Status:");
    info!("  Temperature: {:.1}°C", thermal_status.current_temperature);
    info!("  Throttling: {}", if thermal_status.is_throttling { "Active" } else { "Inactive" });
    info!("  Thermal Headroom: {:.1}°C", thermal_status.thermal_headroom);

    Ok(())
}

/// Demo 5: SIMD Audio Processing Acceleration
async fn demo_simd_audio_processing() -> Result<()> {
    info!("\n⚡ Demo 5: SIMD Audio Processing");
    info!("=================================");

    let mut simd_processor = SIMDAudioProcessor::new()?;
    let simd_capability = simd_processor.get_simd_capability();
    info!("SIMD Capability: {:?}", simd_capability);

    // Generate test audio data
    let test_sizes = [1024, 4096, 16384, 65536];

    for &size in &test_sizes {
        info!("\nTesting with {} samples:", size);

        // Test noise gate with SIMD acceleration
        let mut test_samples: Vec<f32> = (0..size).map(|i| {
            0.1 * (i as f32 * 0.001).sin() + 0.02 * rand::random::<f32>()
        }).collect();

        let noise_gate_stats = simd_processor.simd_noise_gate(&mut test_samples, 0.01, 0.5)?;
        info!("  Noise Gate: {:.2} MSamples/sec ({:.1}% SIMD efficiency)",
              noise_gate_stats.throughput_msamples_per_sec,
              noise_gate_stats.simd_efficiency);

        // Test normalization with SIMD acceleration
        let normalize_stats = simd_processor.simd_normalize(&mut test_samples)?;
        info!("  Normalize: {:.2} MSamples/sec ({:.1}% SIMD efficiency)",
              normalize_stats.throughput_msamples_per_sec,
              normalize_stats.simd_efficiency);

        // Test FIR filter with SIMD acceleration
        let filter_coefficients = vec![0.1, 0.2, 0.4, 0.2, 0.1]; // Simple 5-tap filter
        let fir_stats = simd_processor.simd_fir_filter(&mut test_samples, &filter_coefficients)?;
        info!("  FIR Filter: {:.2} MSamples/sec ({:.1}% SIMD efficiency)",
              fir_stats.throughput_msamples_per_sec,
              fir_stats.simd_efficiency);
    }

    // Run comprehensive benchmark
    info!("\nRunning SIMD benchmark...");
    let benchmark_result = simd_processor.benchmark_simd_performance(65536)?;
    info!("SIMD Benchmark Report:\n{}", benchmark_result.generate_report());

    Ok(())
}

/// Demo 6: Thermal Management and Adaptive Performance
async fn demo_thermal_management(intel_accel: &IntelAcceleration) -> Result<()> {
    info!("\n🌡️ Demo 6: Thermal Management");
    info!("==============================");

    let mut thermal_manager = intel_accel.thermal_manager.write();

    // Set thermal policy
    thermal_manager.set_thermal_policy(ThermalPolicy::Adaptive);
    info!("Set thermal policy to Adaptive");

    // Monitor thermal conditions over time
    info!("Monitoring thermal conditions...");
    for cycle in 0..10 {
        let thermal_status = thermal_manager.update_thermal_state().await?;

        info!("Cycle {}: Temp {:.1}°C, Throttling: {}, Level: {}/10",
              cycle + 1,
              thermal_status.maximum_temperature,
              if thermal_status.is_throttling { "Active" } else { "Inactive" },
              thermal_status.throttle_level);

        // Simulate varying thermal load
        if cycle == 5 {
            info!("  Simulating high thermal load...");
        }

        sleep(Duration::from_millis(500)).await;
    }

    // Get thermal statistics
    let thermal_stats = thermal_manager.get_thermal_statistics();
    info!("Thermal Management Report:\n{}", thermal_stats.generate_report());

    Ok(())
}

/// Demo 7: Real-time Performance Monitoring
async fn demo_performance_monitoring(intel_accel: &IntelAcceleration) -> Result<()> {
    info!("\n📈 Demo 7: Performance Monitoring");
    info!("==================================");

    // Create performance monitor
    let perf_monitor = Arc::new(PerformanceMonitor::new());

    // Simulate real-time audio processing workload
    info!("Simulating real-time audio processing workload...");

    for iteration in 0..20 {
        let start_time = std::time::Instant::now();

        // Simulate audio processing operations
        let sample_count = 1024; // 1K samples per frame

        // Record sample processing
        perf_monitor.record_samples_processed(sample_count);
        perf_monitor.record_frame_processed();

        // Simulate memory allocation
        let _temp_buffer = vec![0.0f32; sample_count as usize];
        perf_monitor.record_allocation();
        perf_monitor.record_memory_usage(sample_count as usize * 4); // 4 bytes per f32

        // Simulate varying processing times
        let processing_delay = if iteration % 5 == 0 {
            Duration::from_millis(25) // Occasionally higher latency
        } else {
            Duration::from_millis(15) // Normal processing time
        };

        sleep(processing_delay).await;

        let frame_latency = start_time.elapsed();
        perf_monitor.record_audio_latency(frame_latency);

        // Occasional buffer overrun simulation
        if iteration == 10 {
            perf_monitor.record_buffer_overrun();
            info!("  Simulated buffer overrun at iteration {}", iteration);
        }

        if iteration % 5 == 0 {
            info!("  Processed frame {}: {:.1}ms latency", iteration + 1, frame_latency.as_secs_f32() * 1000.0);
        }
    }

    // Get performance statistics
    let perf_stats = perf_monitor.get_stats();
    info!("Performance Report:\n{}", perf_stats.generate_report());

    // Check performance targets
    let target_status = perf_monitor.check_performance_targets();
    info!("Performance Target Status:");
    info!("  Latency Target: {}", if target_status.latency_target_met { "✅ Met" } else { "❌ Missed" });
    info!("  Memory Target: {}", if target_status.memory_target_met { "✅ Met" } else { "❌ Exceeded" });
    info!("  Error Rate: {}", if target_status.error_rate_acceptable { "✅ Acceptable" } else { "❌ Too High" });
    info!("  CPU Usage: {}", if target_status.cpu_usage_acceptable { "✅ Acceptable" } else { "❌ Too High" });

    Ok(())
}

/// Generate some random data for testing
fn rand() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    static mut SEED: u64 = 1;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        let mut hasher = DefaultHasher::new();
        SEED.hash(&mut hasher);
        (hasher.finish() & 0xFFFFFF) as f32 / 16777216.0
    }
}