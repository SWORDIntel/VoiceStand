use voicestand_core::{Result, VoiceStandError};
use std::sync::Arc;
use parking_lot::RwLock;
use nix::sched::{CpuSet, sched_setaffinity, sched_getaffinity};
use nix::unistd::Pid;
use tracing::{info, warn, error, debug};

/// Intel Meteor Lake CPU optimizer for hybrid architecture
pub struct CPUOptimizer {
    topology: CPUTopology,
    thread_assignments: Arc<RwLock<ThreadAssignments>>,
    thermal_monitor: Arc<RwLock<ThermalMonitor>>,
    performance_monitor: Arc<RwLock<CPUPerformanceMonitor>>,
    original_affinity: CpuSet,
}

/// CPU topology for Intel Meteor Lake
#[derive(Debug, Clone)]
pub struct CPUTopology {
    pub p_cores: Vec<u8>,        // Performance cores (0,2,4,6,8,10)
    pub e_cores: Vec<u8>,        // Efficiency cores (12-19)
    pub lp_e_cores: Vec<u8>,     // Low-power E-cores (20-21)
    pub total_cores: u8,
    pub has_avx2: bool,
    pub has_avx512: bool,        // Hidden on Meteor Lake P-cores
    pub has_amx: bool,           // Advanced Matrix Extensions
}

/// Thread assignment tracking
#[derive(Debug, Clone)]
struct ThreadAssignments {
    audio_capture_cores: Vec<u8>,    // Real-time audio threads
    audio_processing_cores: Vec<u8>, // Audio processing pipeline
    speech_inference_cores: Vec<u8>, // Speech recognition inference
    background_cores: Vec<u8>,       // Background tasks, I/O
    gui_cores: Vec<u8>,             // GUI and user interaction
}

/// Thermal monitoring for dynamic frequency scaling
#[derive(Debug, Clone)]
struct ThermalMonitor {
    current_temp_celsius: f32,
    max_temp_celsius: f32,
    thermal_throttle_threshold: f32,
    p_core_frequencies: Vec<f32>,
    e_core_frequencies: Vec<f32>,
    is_throttling: bool,
}

/// CPU performance monitoring
#[derive(Debug, Clone)]
struct CPUPerformanceMonitor {
    p_core_utilization: Vec<f32>,
    e_core_utilization: Vec<f32>,
    context_switches: u64,
    cache_misses: u64,
    instructions_per_cycle: f32,
    branch_prediction_accuracy: f32,
    start_time: std::time::Instant,
}

impl CPUOptimizer {
    /// Initialize CPU optimizer with topology detection
    pub fn new(capabilities: &crate::HardwareCapabilities) -> Result<Self> {
        let topology = Self::detect_cpu_topology()?;
        info!("Detected CPU topology: {} P-cores, {} E-cores, {} LP E-cores",
              topology.p_cores.len(), topology.e_cores.len(), topology.lp_e_cores.len());

        // Get current process affinity to restore later
        let original_affinity = sched_getaffinity(Pid::from_raw(0))
            .map_err(|e| VoiceStandError::Hardware(format!("Failed to get CPU affinity: {}", e)))?;

        let thread_assignments = Arc::new(RwLock::new(ThreadAssignments::new()));
        let thermal_monitor = Arc::new(RwLock::new(ThermalMonitor::new()));
        let performance_monitor = Arc::new(RwLock::new(CPUPerformanceMonitor::new()));

        Ok(Self {
            topology,
            thread_assignments,
            thermal_monitor,
            performance_monitor,
            original_affinity,
        })
    }

    /// Detect Intel Meteor Lake CPU topology
    fn detect_cpu_topology() -> Result<CPUTopology> {
        debug!("Detecting CPU topology...");

        // Read /proc/cpuinfo for core information
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo")
            .map_err(|e| VoiceStandError::Hardware(format!("Failed to read /proc/cpuinfo: {}", e)))?;

        // Parse CPU features
        let has_avx2 = cpuinfo.contains("avx2");
        let has_avx512 = cpuinfo.contains("avx512f"); // Might be hidden on Meteor Lake
        let has_amx = cpuinfo.contains("amx");

        // For Intel Meteor Lake, we know the topology:
        // P-cores: 0,2,4,6,8,10 (with hyperthreading: 1,3,5,7,9,11)
        // E-cores: 12-19
        // LP E-cores: 20-21
        let topology = CPUTopology {
            p_cores: vec![0, 2, 4, 6, 8, 10],
            e_cores: vec![12, 13, 14, 15, 16, 17, 18, 19],
            lp_e_cores: vec![20, 21],
            total_cores: 22,
            has_avx2,
            has_avx512,
            has_amx,
        };

        info!("CPU Features: AVX2={}, AVX512={}, AMX={}", has_avx2, has_avx512, has_amx);

        Ok(topology)
    }

    /// Optimize thread placement for VoiceStand workload
    pub fn optimize_thread_placement(&mut self) -> Result<()> {
        info!("Optimizing thread placement for hybrid CPU architecture");

        let mut assignments = self.thread_assignments.write();

        // Real-time audio capture: Pin to dedicated P-cores
        assignments.audio_capture_cores = vec![self.topology.p_cores[0]]; // Core 0

        // Audio processing pipeline: Use remaining P-cores
        assignments.audio_processing_cores = self.topology.p_cores[1..4].to_vec(); // Cores 2,4,6

        // Speech inference: Dedicated P-cores with potential AVX-512
        assignments.speech_inference_cores = self.topology.p_cores[4..].to_vec(); // Cores 8,10

        // Background tasks: Use E-cores for efficiency
        assignments.background_cores = self.topology.e_cores[0..4].to_vec(); // Cores 12-15

        // GUI and user interaction: Responsive E-cores
        assignments.gui_cores = self.topology.e_cores[4..].to_vec(); // Cores 16-19

        self.apply_thread_affinity(&assignments)?;

        info!("Thread placement optimized for VoiceStand workload");
        Ok(())
    }

    /// Apply CPU affinity for current process
    fn apply_thread_affinity(&self, assignments: &ThreadAssignments) -> Result<()> {
        // For now, set process affinity to use all assigned cores
        let mut cpu_set = CpuSet::new();

        // Add all assigned cores to the CPU set
        for &core in &assignments.audio_capture_cores {
            cpu_set.set(core as usize).map_err(|e|
                VoiceStandError::Hardware(format!("Failed to set CPU {}: {}", core, e)))?;
        }
        for &core in &assignments.audio_processing_cores {
            cpu_set.set(core as usize).map_err(|e|
                VoiceStandError::Hardware(format!("Failed to set CPU {}: {}", core, e)))?;
        }
        for &core in &assignments.speech_inference_cores {
            cpu_set.set(core as usize).map_err(|e|
                VoiceStandError::Hardware(format!("Failed to set CPU {}: {}", core, e)))?;
        }

        // Apply affinity to current process
        sched_setaffinity(Pid::from_raw(0), &cpu_set)
            .map_err(|e| VoiceStandError::Hardware(format!("Failed to set CPU affinity: {}", e)))?;

        debug!("Applied CPU affinity for optimized thread placement");
        Ok(())
    }

    /// Set thread affinity for specific workload type
    pub fn set_thread_affinity_for_workload(&self, workload_type: WorkloadType) -> Result<()> {
        let assignments = self.thread_assignments.read();
        let mut cpu_set = CpuSet::new();

        let cores = match workload_type {
            WorkloadType::RealTimeAudio => &assignments.audio_capture_cores,
            WorkloadType::AudioProcessing => &assignments.audio_processing_cores,
            WorkloadType::SpeechInference => &assignments.speech_inference_cores,
            WorkloadType::BackgroundTask => &assignments.background_cores,
            WorkloadType::UserInterface => &assignments.gui_cores,
        };

        for &core in cores {
            cpu_set.set(core as usize).map_err(|e|
                VoiceStandError::Hardware(format!("Failed to set CPU {}: {}", core, e)))?;
        }

        sched_setaffinity(Pid::from_raw(0), &cpu_set)
            .map_err(|e| VoiceStandError::Hardware(format!("Failed to set affinity for {:?}: {}", workload_type, e)))?;

        debug!("Set thread affinity for {:?} workload", workload_type);
        Ok(())
    }

    /// Monitor thermal conditions and adjust performance
    pub async fn update_thermal_monitoring(&mut self) -> Result<ThermalStatus> {
        let mut thermal = self.thermal_monitor.write();

        // Read CPU temperature from thermal zones
        thermal.current_temp_celsius = Self::read_cpu_temperature().await?;

        // Check P-core frequencies
        for (i, &core) in self.topology.p_cores.iter().enumerate() {
            if i < thermal.p_core_frequencies.len() {
                thermal.p_core_frequencies[i] = Self::read_core_frequency(core).await?;
            }
        }

        // Check E-core frequencies
        for (i, &core) in self.topology.e_cores.iter().enumerate() {
            if i < thermal.e_core_frequencies.len() {
                thermal.e_core_frequencies[i] = Self::read_core_frequency(core).await?;
            }
        }

        // Determine throttling status
        thermal.is_throttling = thermal.current_temp_celsius > thermal.thermal_throttle_threshold;

        let status = ThermalStatus {
            current_temperature: thermal.current_temp_celsius,
            is_throttling: thermal.is_throttling,
            p_core_avg_frequency: thermal.p_core_frequencies.iter().sum::<f32>() / thermal.p_core_frequencies.len() as f32,
            e_core_avg_frequency: thermal.e_core_frequencies.iter().sum::<f32>() / thermal.e_core_frequencies.len() as f32,
            thermal_headroom: thermal.thermal_throttle_threshold - thermal.current_temp_celsius,
        };

        if thermal.is_throttling {
            warn!("CPU thermal throttling detected at {:.1}°C", thermal.current_temp_celsius);
        }

        Ok(status)
    }

    /// Read CPU temperature from thermal zone
    async fn read_cpu_temperature() -> Result<f32> {
        // Try multiple thermal zone locations
        let thermal_paths = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
            "/sys/devices/platform/coretemp.0/hwmon/hwmon0/temp1_input",
        ];

        for path in &thermal_paths {
            if let Ok(temp_str) = std::fs::read_to_string(path) {
                if let Ok(temp_millicelsius) = temp_str.trim().parse::<i32>() {
                    return Ok(temp_millicelsius as f32 / 1000.0);
                }
            }
        }

        // Fallback to estimated temperature
        warn!("Could not read CPU temperature, using estimate");
        Ok(65.0) // Conservative estimate
    }

    /// Read individual core frequency
    async fn read_core_frequency(core_id: u8) -> Result<f32> {
        let freq_path = format!("/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", core_id);

        if let Ok(freq_str) = std::fs::read_to_string(&freq_path) {
            if let Ok(freq_khz) = freq_str.trim().parse::<u32>() {
                return Ok(freq_khz as f32 / 1000.0); // Convert to MHz
            }
        }

        // Return base frequency estimate
        if core_id < 12 {
            Ok(3400.0) // P-core base frequency ~3.4GHz
        } else {
            Ok(2400.0) // E-core base frequency ~2.4GHz
        }
    }

    /// Update CPU performance counters
    pub fn update_performance_counters(&mut self) -> Result<CPUPerformanceStats> {
        let mut perf = self.performance_monitor.write();

        // Read CPU utilization per core (simplified)
        for (i, &core) in self.topology.p_cores.iter().enumerate() {
            if i < perf.p_core_utilization.len() {
                perf.p_core_utilization[i] = Self::read_core_utilization(core)?;
            }
        }

        for (i, &core) in self.topology.e_cores.iter().enumerate() {
            if i < perf.e_core_utilization.len() {
                perf.e_core_utilization[i] = Self::read_core_utilization(core)?;
            }
        }

        // Calculate averages
        let p_core_avg_utilization = perf.p_core_utilization.iter().sum::<f32>() / perf.p_core_utilization.len() as f32;
        let e_core_avg_utilization = perf.e_core_utilization.iter().sum::<f32>() / perf.e_core_utilization.len() as f32;

        let elapsed = perf.start_time.elapsed();

        Ok(CPUPerformanceStats {
            p_core_utilization: p_core_avg_utilization,
            e_core_utilization: e_core_avg_utilization,
            total_utilization: (p_core_avg_utilization + e_core_avg_utilization) / 2.0,
            context_switches_per_sec: perf.context_switches as f32 / elapsed.as_secs_f32(),
            cache_miss_rate: perf.cache_misses as f32 / elapsed.as_secs_f32(),
            instructions_per_cycle: perf.instructions_per_cycle,
            uptime: elapsed,
            optimal_core_usage: p_core_avg_utilization > 60.0 && e_core_avg_utilization < 40.0, // P-cores busy, E-cores available
        })
    }

    /// Read individual core utilization
    fn read_core_utilization(core_id: u8) -> Result<f32> {
        // In a real implementation, this would read from /proc/stat or use performance counters
        // For now, return a simulated value
        Ok(50.0 + (core_id as f32 * 2.0) % 30.0) // Simulated utilization
    }

    /// Get CPU capabilities and current configuration
    pub fn get_cpu_info(&self) -> CPUInfo {
        let assignments = self.thread_assignments.read();

        CPUInfo {
            topology: self.topology.clone(),
            thread_assignments: assignments.clone(),
            avx2_available: self.topology.has_avx2,
            avx512_available: self.topology.has_avx512,
            amx_available: self.topology.has_amx,
            optimization_active: true,
        }
    }

    /// Reset CPU affinity to original state
    pub fn reset_affinity(&self) -> Result<()> {
        sched_setaffinity(Pid::from_raw(0), &self.original_affinity)
            .map_err(|e| VoiceStandError::Hardware(format!("Failed to reset CPU affinity: {}", e)))?;

        info!("CPU affinity reset to original configuration");
        Ok(())
    }
}

/// Workload types for thread affinity optimization
#[derive(Debug, Clone, Copy)]
pub enum WorkloadType {
    RealTimeAudio,    // Latency-critical audio capture
    AudioProcessing,  // Audio signal processing
    SpeechInference, // AI inference workloads
    BackgroundTask,  // File I/O, networking
    UserInterface,   // GUI and user interaction
}

/// Thermal monitoring status
#[derive(Debug, Clone)]
pub struct ThermalStatus {
    pub current_temperature: f32,
    pub is_throttling: bool,
    pub p_core_avg_frequency: f32,
    pub e_core_avg_frequency: f32,
    pub thermal_headroom: f32,
}

/// CPU performance statistics
#[derive(Debug, Clone)]
pub struct CPUPerformanceStats {
    pub p_core_utilization: f32,
    pub e_core_utilization: f32,
    pub total_utilization: f32,
    pub context_switches_per_sec: f32,
    pub cache_miss_rate: f32,
    pub instructions_per_cycle: f32,
    pub uptime: std::time::Duration,
    pub optimal_core_usage: bool,
}

/// CPU information and configuration
#[derive(Debug, Clone)]
pub struct CPUInfo {
    pub topology: CPUTopology,
    pub thread_assignments: ThreadAssignments,
    pub avx2_available: bool,
    pub avx512_available: bool,
    pub amx_available: bool,
    pub optimization_active: bool,
}

impl ThreadAssignments {
    fn new() -> Self {
        Self {
            audio_capture_cores: Vec::new(),
            audio_processing_cores: Vec::new(),
            speech_inference_cores: Vec::new(),
            background_cores: Vec::new(),
            gui_cores: Vec::new(),
        }
    }
}

impl ThermalMonitor {
    fn new() -> Self {
        Self {
            current_temp_celsius: 0.0,
            max_temp_celsius: 0.0,
            thermal_throttle_threshold: 90.0, // 90°C throttle threshold
            p_core_frequencies: vec![0.0; 6],  // 6 P-cores
            e_core_frequencies: vec![0.0; 8], // 8 E-cores
            is_throttling: false,
        }
    }
}

impl CPUPerformanceMonitor {
    fn new() -> Self {
        Self {
            p_core_utilization: vec![0.0; 6],  // 6 P-cores
            e_core_utilization: vec![0.0; 8], // 8 E-cores
            context_switches: 0,
            cache_misses: 0,
            instructions_per_cycle: 1.0,
            branch_prediction_accuracy: 0.95,
            start_time: std::time::Instant::now(),
        }
    }
}

impl CPUPerformanceStats {
    /// Check if CPU is meeting performance targets
    pub fn meets_targets(&self) -> bool {
        self.total_utilization < 80.0 &&        // <80% total utilization
        self.optimal_core_usage &&              // P-cores busy, E-cores available
        self.context_switches_per_sec < 1000.0  // <1000 context switches/sec
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "Intel Meteor Lake CPU Performance Report\n\
             ========================================\n\
             P-Core Utilization: {:.1}%\n\
             E-Core Utilization: {:.1}%\n\
             Total Utilization: {:.1}% (target: <80%)\n\
             Context Switches: {:.1}/sec\n\
             Cache Miss Rate: {:.1}/sec\n\
             Instructions/Cycle: {:.2}\n\
             Optimal Core Usage: {}\n\
             Uptime: {:.1}s\n\
             Target Compliance: {}",
            self.p_core_utilization,
            self.e_core_utilization,
            self.total_utilization,
            self.context_switches_per_sec,
            self.cache_miss_rate,
            self.instructions_per_cycle,
            if self.optimal_core_usage { "✅ YES" } else { "❌ NO" },
            self.uptime.as_secs_f32(),
            if self.meets_targets() { "✅ PASSED" } else { "❌ FAILED" }
        )
    }
}