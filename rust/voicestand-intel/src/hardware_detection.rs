use voicestand_core::{Result, VoiceStandError};
use std::fs;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, debug};

/// Hardware capabilities detected on Intel Meteor Lake system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    // CPU Information
    pub cpu_model: String,
    pub cpu_vendor: String,
    pub cpu_family: u8,
    pub cpu_model_id: u8,
    pub cpu_stepping: u8,

    // Core Configuration
    pub p_cores: u8,
    pub e_cores: u8,
    pub lp_e_cores: u8,
    pub threads_per_core: u8,
    pub l3_cache_size_kb: u32,

    // SIMD Capabilities
    pub has_sse2: bool,
    pub has_sse4_1: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512dq: bool,
    pub has_fma: bool,
    pub has_amx: bool,

    // Intel-Specific Features
    pub has_npu: bool,
    pub has_gna: bool,
    pub npu_tops: f32,
    pub gna_power_mw: f32,

    // Memory Configuration
    pub total_memory_gb: u32,
    pub memory_type: String,
    pub memory_speed_mhz: u16,
    pub memory_channels: u8,

    // Power and Thermal
    pub max_turbo_frequency_mhz: u16,
    pub base_frequency_mhz: u16,
    pub thermal_design_power_w: u16,
    pub max_temperature_c: u8,

    // System Information
    pub is_laptop: bool,
    pub manufacturer: String,
    pub model: String,
    pub bios_version: String,
}

/// Runtime hardware adapter that optimizes based on current conditions
pub struct HardwareAdapter {
    capabilities: HardwareCapabilities,
    current_state: HardwareState,
    optimization_profile: OptimizationProfile,
    performance_history: Vec<PerformanceDataPoint>,
}

/// Current hardware state
#[derive(Debug, Clone)]
struct HardwareState {
    current_temperature: f32,
    current_frequency_p_cores: f32,
    current_frequency_e_cores: f32,
    current_power_consumption: f32,
    thermal_throttling_active: bool,
    battery_level: Option<f32>, // None for desktop systems
    is_plugged_in: bool,
}

/// Optimization profile for different use cases
#[derive(Debug, Clone)]
pub enum OptimizationProfile {
    MaxPerformance,    // Desktop, plugged in, maximum performance
    Balanced,          // Balanced performance and power
    PowerEfficient,    // Battery optimization, minimal power
    RealTime,          // Latency-optimized for real-time audio
    Background,        // Minimal CPU usage for background operation
}

/// Performance data point for adaptive optimization
#[derive(Debug, Clone)]
struct PerformanceDataPoint {
    timestamp: std::time::Instant,
    cpu_utilization: f32,
    temperature: f32,
    power_consumption: f32,
    audio_latency_ms: f32,
    user_satisfaction_score: Option<f32>,
}

impl HardwareCapabilities {
    /// Detect hardware capabilities on Intel Meteor Lake system
    pub fn detect() -> Result<Self> {
        info!("Starting hardware capability detection...");

        let cpu_info = Self::detect_cpu_info()?;
        let memory_info = Self::detect_memory_info()?;
        let system_info = Self::detect_system_info()?;
        let intel_features = Self::detect_intel_features()?;

        let capabilities = HardwareCapabilities {
            // CPU Information
            cpu_model: cpu_info.get("model_name").unwrap_or(&"Unknown".to_string()).clone(),
            cpu_vendor: cpu_info.get("vendor_id").unwrap_or(&"Unknown".to_string()).clone(),
            cpu_family: cpu_info.get("family").and_then(|s| s.parse().ok()).unwrap_or(0),
            cpu_model_id: cpu_info.get("model").and_then(|s| s.parse().ok()).unwrap_or(0),
            cpu_stepping: cpu_info.get("stepping").and_then(|s| s.parse().ok()).unwrap_or(0),

            // Core Configuration (Intel Meteor Lake specific)
            p_cores: 6,  // Fixed for Meteor Lake
            e_cores: 8,  // Fixed for Meteor Lake
            lp_e_cores: 2, // Fixed for Meteor Lake
            threads_per_core: if cpu_info.get("siblings").and_then(|s| s.parse::<u8>().ok()).unwrap_or(2) >
                                cpu_info.get("cores").and_then(|s| s.parse::<u8>().ok()).unwrap_or(1) { 2 } else { 1 },
            l3_cache_size_kb: Self::parse_cache_size(&cpu_info),

            // SIMD Capabilities
            has_sse2: cpu_info.get("flags").map_or(false, |f| f.contains("sse2")),
            has_sse4_1: cpu_info.get("flags").map_or(false, |f| f.contains("sse4_1")),
            has_avx: cpu_info.get("flags").map_or(false, |f| f.contains("avx")),
            has_avx2: cpu_info.get("flags").map_or(false, |f| f.contains("avx2")),
            has_avx512f: cpu_info.get("flags").map_or(false, |f| f.contains("avx512f")),
            has_avx512dq: cpu_info.get("flags").map_or(false, |f| f.contains("avx512dq")),
            has_fma: cpu_info.get("flags").map_or(false, |f| f.contains("fma")),
            has_amx: cpu_info.get("flags").map_or(false, |f| f.contains("amx")),

            // Intel-Specific Features
            has_npu: intel_features.has_npu,
            has_gna: intel_features.has_gna,
            npu_tops: intel_features.npu_tops,
            gna_power_mw: intel_features.gna_power_mw,

            // Memory Configuration
            total_memory_gb: memory_info.total_gb,
            memory_type: memory_info.memory_type,
            memory_speed_mhz: memory_info.speed_mhz,
            memory_channels: memory_info.channels,

            // Power and Thermal
            max_turbo_frequency_mhz: Self::detect_max_frequency().unwrap_or(4800), // Meteor Lake typical
            base_frequency_mhz: Self::detect_base_frequency().unwrap_or(3400),     // Meteor Lake typical
            thermal_design_power_w: Self::detect_tdp().unwrap_or(28),              // Meteor Lake H-series
            max_temperature_c: 100, // Intel typical

            // System Information
            is_laptop: system_info.is_laptop,
            manufacturer: system_info.manufacturer,
            model: system_info.model,
            bios_version: system_info.bios_version,
        };

        info!("Hardware detection completed: {} with NPU={}, GNA={}",
              capabilities.cpu_model, capabilities.has_npu, capabilities.has_gna);

        Ok(capabilities)
    }

    /// Parse CPU information from /proc/cpuinfo
    fn detect_cpu_info() -> Result<HashMap<String, String>> {
        let cpuinfo_content = fs::read_to_string("/proc/cpuinfo")
            .map_err(|e| VoiceStandError::Hardware(format!("Cannot read /proc/cpuinfo: {}", e)))?;

        let mut cpu_info = HashMap::new();

        for line in cpuinfo_content.lines() {
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().replace(' ', "_");
                let value = value.trim().to_string();

                // Only keep the first occurrence (from CPU 0)
                if !cpu_info.contains_key(&key) {
                    cpu_info.insert(key, value);
                }
            }
        }

        Ok(cpu_info)
    }

    /// Parse cache size information
    fn parse_cache_size(cpu_info: &HashMap<String, String>) -> u32 {
        if let Some(cache_size) = cpu_info.get("cache_size") {
            // Parse "12288 KB" format
            if let Some(size_str) = cache_size.split_whitespace().next() {
                return size_str.parse().unwrap_or(12288); // Meteor Lake typical L3
            }
        }
        12288 // Default for Meteor Lake
    }

    /// Detect memory information
    fn detect_memory_info() -> Result<MemoryInfo> {
        let meminfo = fs::read_to_string("/proc/meminfo")
            .map_err(|e| VoiceStandError::Hardware(format!("Cannot read /proc/meminfo: {}", e)))?;

        let mut total_kb = 0u32;
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(size_str) = line.split_whitespace().nth(1) {
                    total_kb = size_str.parse().unwrap_or(0);
                }
                break;
            }
        }

        // Detect memory type and speed (simplified)
        let (memory_type, speed_mhz) = Self::detect_memory_specs();

        Ok(MemoryInfo {
            total_gb: (total_kb / 1024 / 1024).max(1),
            memory_type,
            speed_mhz,
            channels: 2, // Typical dual-channel for laptops
        })
    }

    /// Detect memory specifications
    fn detect_memory_specs() -> (String, u16) {
        // Try to read from DMI if available
        if let Ok(dmi_content) = fs::read_to_string("/sys/devices/virtual/dmi/id/memory_type") {
            let memory_type = dmi_content.trim().to_string();
            // DDR5 typical speeds for Meteor Lake
            let speed = if memory_type.contains("DDR5") { 5600 } else { 3200 };
            return (memory_type, speed);
        }

        // Default for modern systems
        ("DDR5".to_string(), 5600)
    }

    /// Detect system information
    fn detect_system_info() -> Result<SystemInfo> {
        let mut system_info = SystemInfo::default();

        // Read DMI information if available
        if let Ok(vendor) = fs::read_to_string("/sys/devices/virtual/dmi/id/sys_vendor") {
            system_info.manufacturer = vendor.trim().to_string();
        }

        if let Ok(product) = fs::read_to_string("/sys/devices/virtual/dmi/id/product_name") {
            system_info.model = product.trim().to_string();
        }

        if let Ok(bios) = fs::read_to_string("/sys/devices/virtual/dmi/id/bios_version") {
            system_info.bios_version = bios.trim().to_string();
        }

        // Detect if laptop (check for battery or chassis type)
        system_info.is_laptop = std::path::Path::new("/sys/class/power_supply/BAT0").exists() ||
                               std::path::Path::new("/sys/class/power_supply/BAT1").exists();

        Ok(system_info)
    }

    /// Detect Intel-specific features (NPU, GNA)
    fn detect_intel_features() -> Result<IntelFeatures> {
        let mut features = IntelFeatures::default();

        // Check for Intel NPU device
        if std::path::Path::new("/sys/bus/pci/devices/0000:00:0b.0").exists() ||
           std::path::Path::new("/dev/intel_vpu").exists() {
            features.has_npu = true;
            features.npu_tops = 11.0; // Meteor Lake NPU specification
            info!("Intel NPU detected: 11 TOPS capability");
        }

        // Check for Intel GNA device
        if std::path::Path::new("/dev/intel_gna").exists() ||
           std::path::Path::new("/sys/class/gna").exists() {
            features.has_gna = true;
            features.gna_power_mw = 100.0; // Typical GNA power consumption
            info!("Intel GNA detected: ~100mW power budget");
        }

        Ok(features)
    }

    /// Detect maximum CPU frequency
    fn detect_max_frequency() -> Option<u16> {
        let freq_paths = [
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
            "/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq", // P-core
        ];

        for path in &freq_paths {
            if let Ok(freq_str) = fs::read_to_string(path) {
                if let Ok(freq_khz) = freq_str.trim().parse::<u32>() {
                    return Some((freq_khz / 1000) as u16); // Convert to MHz
                }
            }
        }

        None
    }

    /// Detect base CPU frequency
    fn detect_base_frequency() -> Option<u16> {
        if let Ok(freq_str) = fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency") {
            if let Ok(freq_khz) = freq_str.trim().parse::<u32>() {
                return Some((freq_khz / 1000) as u16);
            }
        }

        None
    }

    /// Detect thermal design power
    fn detect_tdp() -> Option<u16> {
        // Try to read from Intel RAPL if available
        if let Ok(power_str) = fs::read_to_string("/sys/class/powercap/intel-rapl/intel-rapl:0/constraint_0_power_limit_uw") {
            if let Ok(power_uw) = power_str.trim().parse::<u64>() {
                return Some((power_uw / 1_000_000) as u16); // Convert to watts
            }
        }

        None
    }

    /// Check if the system meets minimum requirements for VoiceStand
    pub fn meets_minimum_requirements(&self) -> bool {
        self.total_memory_gb >= 8 &&         // Minimum 8GB RAM
        self.has_avx2 &&                     // AVX2 required for SIMD
        (self.has_npu || self.has_gna) &&   // Intel AI accelerator required
        self.p_cores >= 4                   // Minimum 4 P-cores
    }

    /// Generate hardware capability report
    pub fn generate_report(&self) -> String {
        format!(
            "Intel Meteor Lake Hardware Report\n\
             ==================================\n\
             CPU: {}\n\
             Configuration: {} P-cores, {} E-cores, {} LP E-cores\n\
             \n\
             SIMD Capabilities:\n\
             - AVX2: {}\n\
             - AVX-512: {} (hidden on P-cores)\n\
             - FMA: {}\n\
             - AMX: {}\n\
             \n\
             Intel AI Accelerators:\n\
             - NPU: {} ({:.1} TOPS)\n\
             - GNA: {} ({:.1}mW)\n\
             \n\
             Memory: {} GB {} @ {} MHz\n\
             System: {} {} ({})\n\
             \n\
             VoiceStand Compatibility: {}",
            self.cpu_model,
            self.p_cores,
            self.e_cores,
            self.lp_e_cores,
            if self.has_avx2 { "✅" } else { "❌" },
            if self.has_avx512f { "✅" } else { "❌" },
            if self.has_fma { "✅" } else { "❌" },
            if self.has_amx { "✅" } else { "❌" },
            if self.has_npu { "✅" } else { "❌" },
            self.npu_tops,
            if self.has_gna { "✅" } else { "❌" },
            self.gna_power_mw,
            self.total_memory_gb,
            self.memory_type,
            self.memory_speed_mhz,
            self.manufacturer,
            self.model,
            if self.is_laptop { "Laptop" } else { "Desktop" },
            if self.meets_minimum_requirements() { "✅ SUPPORTED" } else { "❌ INSUFFICIENT" }
        )
    }
}

impl HardwareAdapter {
    /// Create hardware adapter with detected capabilities
    pub fn new() -> Result<Self> {
        let capabilities = HardwareCapabilities::detect()?;
        let current_state = HardwareState::detect(&capabilities)?;
        let optimization_profile = Self::determine_initial_profile(&capabilities, &current_state);

        Ok(Self {
            capabilities,
            current_state,
            optimization_profile,
            performance_history: Vec::with_capacity(1000), // Keep last 1000 data points
        })
    }

    /// Determine initial optimization profile
    fn determine_initial_profile(capabilities: &HardwareCapabilities, state: &HardwareState) -> OptimizationProfile {
        if !capabilities.is_laptop {
            OptimizationProfile::MaxPerformance
        } else if !state.is_plugged_in {
            OptimizationProfile::PowerEfficient
        } else {
            OptimizationProfile::Balanced
        }
    }

    /// Update hardware state and adapt optimization
    pub async fn update_and_adapt(&mut self) -> Result<AdaptationResult> {
        // Update current hardware state
        self.current_state = HardwareState::detect(&self.capabilities)?;

        // Record performance data point
        let data_point = PerformanceDataPoint {
            timestamp: std::time::Instant::now(),
            cpu_utilization: self.current_state.current_power_consumption / self.capabilities.thermal_design_power_w as f32 * 100.0,
            temperature: self.current_state.current_temperature,
            power_consumption: self.current_state.current_power_consumption,
            audio_latency_ms: 0.0, // Would be provided by audio system
            user_satisfaction_score: None, // Would be provided by user feedback
        };

        self.performance_history.push(data_point);

        // Keep only recent history
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }

        // Adaptive optimization based on current conditions
        let new_profile = self.determine_optimal_profile()?;
        let profile_changed = std::mem::discriminant(&new_profile) != std::mem::discriminant(&self.optimization_profile);

        if profile_changed {
            info!("Switching optimization profile: {:?} -> {:?}", self.optimization_profile, new_profile);
            self.optimization_profile = new_profile;
        }

        Ok(AdaptationResult {
            profile_changed,
            current_profile: self.optimization_profile.clone(),
            thermal_throttling: self.current_state.thermal_throttling_active,
            recommended_actions: self.generate_recommendations(),
        })
    }

    /// Determine optimal profile based on current conditions
    fn determine_optimal_profile(&self) -> Result<OptimizationProfile> {
        // Thermal throttling detected - switch to power efficient
        if self.current_state.thermal_throttling_active {
            return Ok(OptimizationProfile::PowerEfficient);
        }

        // Battery low - prioritize power efficiency
        if let Some(battery_level) = self.current_state.battery_level {
            if battery_level < 20.0 {
                return Ok(OptimizationProfile::PowerEfficient);
            }
        }

        // High temperature - reduce performance
        if self.current_state.current_temperature > 85.0 {
            return Ok(OptimizationProfile::Balanced);
        }

        // Desktop or plugged in with good thermal conditions
        if !self.capabilities.is_laptop || self.current_state.is_plugged_in {
            return Ok(OptimizationProfile::MaxPerformance);
        }

        // Default to balanced for laptops on battery
        Ok(OptimizationProfile::Balanced)
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.current_state.thermal_throttling_active {
            recommendations.push("Reduce CPU-intensive operations due to thermal throttling".to_string());
        }

        if self.current_state.current_temperature > 80.0 {
            recommendations.push("Monitor temperature closely - approaching thermal limits".to_string());
        }

        if let Some(battery_level) = self.current_state.battery_level {
            if battery_level < 30.0 {
                recommendations.push("Enable power saving mode - battery low".to_string());
            }
        }

        if self.capabilities.has_npu && matches!(self.optimization_profile, OptimizationProfile::MaxPerformance) {
            recommendations.push("Utilize Intel NPU for AI workloads to reduce CPU load".to_string());
        }

        if self.capabilities.has_gna {
            recommendations.push("Use Intel GNA for always-on voice detection to save power".to_string());
        }

        recommendations
    }

    /// Get current hardware capabilities
    pub fn get_capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    /// Get current optimization profile
    pub fn get_optimization_profile(&self) -> &OptimizationProfile {
        &self.optimization_profile
    }

    /// Set optimization profile manually
    pub fn set_optimization_profile(&mut self, profile: OptimizationProfile) {
        info!("Manually setting optimization profile: {:?}", profile);
        self.optimization_profile = profile;
    }
}

impl HardwareState {
    /// Detect current hardware state
    fn detect(capabilities: &HardwareCapabilities) -> Result<Self> {
        let current_temperature = Self::read_temperature()?;
        let (freq_p, freq_e) = Self::read_frequencies();
        let power_consumption = Self::read_power_consumption();
        let battery_info = Self::read_battery_info();

        Ok(Self {
            current_temperature,
            current_frequency_p_cores: freq_p,
            current_frequency_e_cores: freq_e,
            current_power_consumption: power_consumption,
            thermal_throttling_active: current_temperature > 90.0, // Simplified check
            battery_level: battery_info.0,
            is_plugged_in: battery_info.1,
        })
    }

    fn read_temperature() -> Result<f32> {
        let thermal_paths = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/devices/platform/coretemp.0/hwmon/hwmon0/temp1_input",
        ];

        for path in &thermal_paths {
            if let Ok(temp_str) = fs::read_to_string(path) {
                if let Ok(temp_millicelsius) = temp_str.trim().parse::<i32>() {
                    return Ok(temp_millicelsius as f32 / 1000.0);
                }
            }
        }

        Ok(65.0) // Default estimate
    }

    fn read_frequencies() -> (f32, f32) {
        let p_core_freq = fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            .and_then(|s| Ok(s.trim().parse::<u32>().unwrap_or(3400000) / 1000))
            .unwrap_or(3400) as f32;

        let e_core_freq = fs::read_to_string("/sys/devices/system/cpu/cpu12/cpufreq/scaling_cur_freq")
            .and_then(|s| Ok(s.trim().parse::<u32>().unwrap_or(2400000) / 1000))
            .unwrap_or(2400) as f32;

        (p_core_freq, e_core_freq)
    }

    fn read_power_consumption() -> f32 {
        // Try Intel RAPL
        if let Ok(power_str) = fs::read_to_string("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj") {
            if let Ok(_power_uj) = power_str.trim().parse::<u64>() {
                return 15.0; // Simplified - would need delta calculation
            }
        }

        15.0 // Default estimate in watts
    }

    fn read_battery_info() -> (Option<f32>, bool) {
        let battery_paths = ["/sys/class/power_supply/BAT0", "/sys/class/power_supply/BAT1"];

        for path in &battery_paths {
            if std::path::Path::new(path).exists() {
                let capacity = fs::read_to_string(format!("{}/capacity", path))
                    .and_then(|s| Ok(s.trim().parse::<f32>().unwrap_or(50.0)))
                    .unwrap_or(50.0);

                let status = fs::read_to_string(format!("{}/status", path))
                    .unwrap_or_else(|_| "Unknown".to_string());

                let is_plugged = status.contains("Charging") || status.contains("Full");

                return (Some(capacity), is_plugged);
            }
        }

        (None, true) // Assume desktop/plugged in if no battery found
    }
}

/// Supporting structures

#[derive(Debug, Default)]
struct MemoryInfo {
    total_gb: u32,
    memory_type: String,
    speed_mhz: u16,
    channels: u8,
}

#[derive(Debug, Default)]
struct SystemInfo {
    manufacturer: String,
    model: String,
    bios_version: String,
    is_laptop: bool,
}

#[derive(Debug, Default)]
struct IntelFeatures {
    has_npu: bool,
    has_gna: bool,
    npu_tops: f32,
    gna_power_mw: f32,
}

/// Adaptation result
#[derive(Debug)]
pub struct AdaptationResult {
    pub profile_changed: bool,
    pub current_profile: OptimizationProfile,
    pub thermal_throttling: bool,
    pub recommended_actions: Vec<String>,
}

/// Detect hardware capabilities (convenience function)
pub fn detect_hardware_capabilities() -> Result<HardwareCapabilities> {
    HardwareCapabilities::detect()
}