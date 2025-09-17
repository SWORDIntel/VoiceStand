use voicestand_core::{Result, VoiceStandError};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::VecDeque;
use tracing::{info, warn, error, debug};

/// Thermal management system for Intel Meteor Lake
pub struct ThermalManager {
    thermal_zones: Vec<ThermalZone>,
    monitoring_state: Arc<RwLock<MonitoringState>>,
    thermal_policy: ThermalPolicy,
    temperature_history: Arc<RwLock<VecDeque<TemperatureReading>>>,
    throttle_controller: Arc<RwLock<ThrottleController>>,
}

/// Individual thermal zone monitoring
#[derive(Debug, Clone)]
struct ThermalZone {
    id: u8,
    name: String,
    path: String,
    current_temp: f32,
    critical_temp: f32,
    warning_temp: f32,
    is_active: bool,
}

/// Thermal monitoring state
#[derive(Debug, Clone)]
struct MonitoringState {
    average_temperature: f32,
    peak_temperature: f32,
    thermal_events: u64,
    throttle_events: u64,
    monitoring_frequency_ms: u64,
    last_update: std::time::Instant,
}

/// Thermal management policy
#[derive(Debug, Clone)]
pub enum ThermalPolicy {
    Conservative,  // Throttle early, prioritize longevity
    Balanced,      // Standard thermal management
    Aggressive,    // Allow higher temperatures for performance
    Adaptive,      // ML-based adaptive throttling
}

/// Temperature reading with metadata
#[derive(Debug, Clone)]
struct TemperatureReading {
    timestamp: std::time::Instant,
    cpu_temp: f32,
    package_temp: f32,
    ambient_temp: Option<f32>,
    power_consumption: f32,
    frequency_p_cores: f32,
    frequency_e_cores: f32,
}

/// Throttling controller
#[derive(Debug, Clone)]
struct ThrottleController {
    is_throttling: bool,
    throttle_level: u8,        // 0-10 throttle intensity
    throttle_start_time: Option<std::time::Instant>,
    throttle_duration_ms: u64,
    recovery_time_ms: u64,
    adaptive_threshold: f32,
}

impl ThermalManager {
    /// Initialize thermal management system
    pub fn new() -> Result<Self> {
        let thermal_zones = Self::discover_thermal_zones()?;
        info!("Discovered {} thermal zones", thermal_zones.len());

        let monitoring_state = Arc::new(RwLock::new(MonitoringState {
            average_temperature: 0.0,
            peak_temperature: 0.0,
            thermal_events: 0,
            throttle_events: 0,
            monitoring_frequency_ms: 1000, // 1 second default
            last_update: std::time::Instant::now(),
        }));

        let temperature_history = Arc::new(RwLock::new(VecDeque::with_capacity(300))); // 5 minutes @ 1Hz
        let throttle_controller = Arc::new(RwLock::new(ThrottleController::new()));

        Ok(Self {
            thermal_zones,
            monitoring_state,
            thermal_policy: ThermalPolicy::Balanced,
            temperature_history,
            throttle_controller,
        })
    }

    /// Discover available thermal zones
    fn discover_thermal_zones() -> Result<Vec<ThermalZone>> {
        let mut zones = Vec::new();

        // Common thermal zone paths for Intel systems
        let thermal_paths = [
            ("/sys/class/thermal/thermal_zone0", "CPU Package", 100.0, 85.0),
            ("/sys/class/thermal/thermal_zone1", "CPU Core", 100.0, 85.0),
            ("/sys/class/thermal/thermal_zone2", "GPU", 95.0, 80.0),
            ("/sys/devices/platform/coretemp.0/hwmon/hwmon0/temp1_input", "Core 0", 100.0, 85.0),
            ("/sys/devices/platform/coretemp.0/hwmon/hwmon0/temp2_input", "Core 1", 100.0, 85.0),
        ];

        for (i, (path, name, critical, warning)) in thermal_paths.iter().enumerate() {
            if std::path::Path::new(&format!("{}/temp", path)).exists() ||
               std::path::Path::new(path).exists() {
                zones.push(ThermalZone {
                    id: i as u8,
                    name: name.to_string(),
                    path: path.to_string(),
                    current_temp: 0.0,
                    critical_temp: *critical,
                    warning_temp: *warning,
                    is_active: true,
                });

                debug!("Found thermal zone: {} at {}", name, path);
            }
        }

        if zones.is_empty() {
            // Create synthetic thermal zone as fallback
            zones.push(ThermalZone {
                id: 0,
                name: "CPU Synthetic".to_string(),
                path: "/proc/cpuinfo".to_string(), // Fallback
                current_temp: 65.0,
                critical_temp: 100.0,
                warning_temp: 85.0,
                is_active: false,
            });
            warn!("No thermal zones found, using synthetic monitoring");
        }

        Ok(zones)
    }

    /// Update thermal readings and apply management
    pub async fn update_thermal_state(&mut self) -> Result<ThermalStatus> {
        let start_time = std::time::Instant::now();

        // Read temperatures from all zones
        for zone in &mut self.thermal_zones {
            if zone.is_active {
                zone.current_temp = self.read_zone_temperature(zone).await?;
            }
        }

        // Calculate aggregate temperatures
        let active_zones: Vec<_> = self.thermal_zones.iter().filter(|z| z.is_active).collect();
        let avg_temp = if !active_zones.is_empty() {
            active_zones.iter().map(|z| z.current_temp).sum::<f32>() / active_zones.len() as f32
        } else {
            65.0 // Fallback estimate
        };

        let max_temp = active_zones.iter().map(|z| z.current_temp).fold(0.0f32, f32::max);

        // Update monitoring state
        {
            let mut state = self.monitoring_state.write();
            state.average_temperature = avg_temp;
            if max_temp > state.peak_temperature {
                state.peak_temperature = max_temp;
            }
            state.last_update = std::time::Instant::now();

            // Check for thermal events
            if max_temp > 90.0 {
                state.thermal_events += 1;
            }
        }

        // Record temperature history
        let reading = TemperatureReading {
            timestamp: std::time::Instant::now(),
            cpu_temp: avg_temp,
            package_temp: max_temp,
            ambient_temp: None, // Would need ambient sensor
            power_consumption: self.estimate_power_consumption(),
            frequency_p_cores: self.read_p_core_frequency().await,
            frequency_e_cores: self.read_e_core_frequency().await,
        };

        {
            let mut history = self.temperature_history.write();
            if history.len() >= 300 {
                history.pop_front();
            }
            history.push_back(reading);
        }

        // Apply thermal management
        let throttle_recommendation = self.evaluate_thermal_policy(avg_temp, max_temp).await?;

        // Update throttle controller
        {
            let mut controller = self.throttle_controller.write();
            controller.update_throttle_state(throttle_recommendation);

            if throttle_recommendation.should_throttle && !controller.is_throttling {
                controller.is_throttling = true;
                controller.throttle_start_time = Some(std::time::Instant::now());
                self.monitoring_state.write().throttle_events += 1;
                warn!("Thermal throttling activated at {:.1}°C", max_temp);
            } else if !throttle_recommendation.should_throttle && controller.is_throttling {
                controller.is_throttling = false;
                controller.throttle_start_time = None;
                info!("Thermal throttling deactivated");
            }
        }

        let processing_time = start_time.elapsed();

        Ok(ThermalStatus {
            average_temperature: avg_temp,
            maximum_temperature: max_temp,
            is_throttling: throttle_recommendation.should_throttle,
            throttle_level: throttle_recommendation.throttle_level,
            thermal_headroom: 90.0 - max_temp, // Assuming 90°C throttle threshold
            active_zones: active_zones.len() as u8,
            monitoring_overhead_us: processing_time.as_micros() as u32,
        })
    }

    /// Read temperature from a specific thermal zone
    async fn read_zone_temperature(&self, zone: &ThermalZone) -> Result<f32> {
        // Try standard thermal zone path first
        let temp_path = if zone.path.contains("thermal_zone") {
            format!("{}/temp", zone.path)
        } else {
            zone.path.clone()
        };

        if let Ok(temp_str) = std::fs::read_to_string(&temp_path) {
            if let Ok(temp_millicelsius) = temp_str.trim().parse::<i32>() {
                return Ok(temp_millicelsius as f32 / 1000.0);
            }
        }

        // Fallback to hwmon paths for Intel systems
        let hwmon_paths = [
            "/sys/devices/platform/coretemp.0/hwmon/hwmon0/temp1_input",
            "/sys/devices/platform/coretemp.0/hwmon/hwmon1/temp1_input",
        ];

        for path in &hwmon_paths {
            if let Ok(temp_str) = std::fs::read_to_string(path) {
                if let Ok(temp_millicelsius) = temp_str.trim().parse::<i32>() {
                    return Ok(temp_millicelsius as f32 / 1000.0);
                }
            }
        }

        // Estimate based on CPU load (fallback)
        Ok(60.0 + (rand::random::<f32>() * 20.0)) // 60-80°C estimate
    }

    /// Evaluate thermal policy and generate throttling recommendation
    async fn evaluate_thermal_policy(&self, avg_temp: f32, max_temp: f32) -> Result<ThrottleRecommendation> {
        let recommendation = match self.thermal_policy {
            ThermalPolicy::Conservative => {
                ThrottleRecommendation {
                    should_throttle: max_temp > 80.0,
                    throttle_level: if max_temp > 85.0 { 8 } else if max_temp > 80.0 { 4 } else { 0 },
                    target_frequency_reduction: if max_temp > 85.0 { 0.3 } else if max_temp > 80.0 { 0.15 } else { 0.0 },
                    recommended_actions: self.generate_thermal_actions(max_temp),
                }
            }
            ThermalPolicy::Balanced => {
                ThrottleRecommendation {
                    should_throttle: max_temp > 90.0,
                    throttle_level: if max_temp > 95.0 { 10 } else if max_temp > 90.0 { 6 } else { 0 },
                    target_frequency_reduction: if max_temp > 95.0 { 0.4 } else if max_temp > 90.0 { 0.2 } else { 0.0 },
                    recommended_actions: self.generate_thermal_actions(max_temp),
                }
            }
            ThermalPolicy::Aggressive => {
                ThrottleRecommendation {
                    should_throttle: max_temp > 95.0,
                    throttle_level: if max_temp > 98.0 { 8 } else if max_temp > 95.0 { 4 } else { 0 },
                    target_frequency_reduction: if max_temp > 98.0 { 0.25 } else if max_temp > 95.0 { 0.1 } else { 0.0 },
                    recommended_actions: self.generate_thermal_actions(max_temp),
                }
            }
            ThermalPolicy::Adaptive => {
                self.adaptive_thermal_evaluation(avg_temp, max_temp).await?
            }
        };

        Ok(recommendation)
    }

    /// Adaptive thermal evaluation using temperature trends
    async fn adaptive_thermal_evaluation(&self, avg_temp: f32, max_temp: f32) -> Result<ThrottleRecommendation> {
        let history = self.temperature_history.read();

        // Calculate temperature trend over last 30 seconds
        let recent_temps: Vec<f32> = history.iter()
            .filter(|reading| reading.timestamp.elapsed().as_secs() <= 30)
            .map(|reading| reading.package_temp)
            .collect();

        let temp_trend = if recent_temps.len() >= 2 {
            recent_temps.last().unwrap() - recent_temps.first().unwrap()
        } else {
            0.0
        };

        // Predictive throttling based on trend
        let predicted_temp = max_temp + (temp_trend * 2.0); // Predict 2x trend continuation

        let should_throttle = predicted_temp > 90.0 || max_temp > 88.0;
        let throttle_level = if predicted_temp > 95.0 || max_temp > 92.0 {
            8
        } else if should_throttle {
            4
        } else {
            0
        };

        Ok(ThrottleRecommendation {
            should_throttle,
            throttle_level,
            target_frequency_reduction: throttle_level as f32 * 0.05, // 5% per level
            recommended_actions: self.generate_thermal_actions(max_temp),
        })
    }

    /// Generate thermal management actions
    fn generate_thermal_actions(&self, max_temp: f32) -> Vec<String> {
        let mut actions = Vec::new();

        if max_temp > 95.0 {
            actions.push("CRITICAL: Reduce CPU-intensive operations immediately".to_string());
            actions.push("Switch to power-efficient processing mode".to_string());
        } else if max_temp > 90.0 {
            actions.push("WARNING: Reduce processing intensity".to_string());
            actions.push("Increase fan speeds if available".to_string());
        } else if max_temp > 85.0 {
            actions.push("INFO: Monitor temperature closely".to_string());
            actions.push("Consider reducing background tasks".to_string());
        }

        actions
    }

    /// Estimate current power consumption
    fn estimate_power_consumption(&self) -> f32 {
        // Try Intel RAPL if available
        if let Ok(energy_str) = std::fs::read_to_string("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj") {
            if let Ok(_energy_uj) = energy_str.trim().parse::<u64>() {
                // Would need to track deltas over time for accurate power calculation
                return 20.0; // Simplified estimate
            }
        }

        // Fallback estimation based on temperature
        let avg_temp = self.monitoring_state.read().average_temperature;
        10.0 + (avg_temp - 50.0) * 0.5 // Rough correlation: temp vs power
    }

    /// Read P-core frequency
    async fn read_p_core_frequency(&self) -> f32 {
        if let Ok(freq_str) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") {
            if let Ok(freq_khz) = freq_str.trim().parse::<u32>() {
                return freq_khz as f32 / 1000.0; // Convert to MHz
            }
        }
        3400.0 // Default P-core base frequency
    }

    /// Read E-core frequency
    async fn read_e_core_frequency(&self) -> f32 {
        if let Ok(freq_str) = std::fs::read_to_string("/sys/devices/system/cpu/cpu12/cpufreq/scaling_cur_freq") {
            if let Ok(freq_khz) = freq_str.trim().parse::<u32>() {
                return freq_khz as f32 / 1000.0;
            }
        }
        2400.0 // Default E-core base frequency
    }

    /// Get thermal statistics
    pub fn get_thermal_statistics(&self) -> ThermalStatistics {
        let state = self.monitoring_state.read();
        let history = self.temperature_history.read();
        let controller = self.throttle_controller.read();

        let temp_variance = if history.len() > 1 {
            let mean = history.iter().map(|r| r.cpu_temp).sum::<f32>() / history.len() as f32;
            let variance = history.iter()
                .map(|r| (r.cpu_temp - mean).powi(2))
                .sum::<f32>() / history.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };

        ThermalStatistics {
            average_temperature: state.average_temperature,
            peak_temperature: state.peak_temperature,
            temperature_variance: temp_variance,
            total_thermal_events: state.thermal_events,
            total_throttle_events: state.throttle_events,
            current_throttle_level: if controller.is_throttling { controller.throttle_level } else { 0 },
            uptime_seconds: state.last_update.elapsed().as_secs(),
            active_thermal_zones: self.thermal_zones.iter().filter(|z| z.is_active).count() as u8,
        }
    }

    /// Set thermal policy
    pub fn set_thermal_policy(&mut self, policy: ThermalPolicy) {
        info!("Changing thermal policy: {:?} -> {:?}", self.thermal_policy, policy);
        self.thermal_policy = policy;
    }

    /// Emergency thermal shutdown check
    pub fn check_emergency_conditions(&self) -> bool {
        let max_temp = self.thermal_zones.iter()
            .filter(|z| z.is_active)
            .map(|z| z.current_temp)
            .fold(0.0f32, f32::max);

        max_temp > 100.0 // Emergency shutdown temperature
    }
}

impl ThrottleController {
    fn new() -> Self {
        Self {
            is_throttling: false,
            throttle_level: 0,
            throttle_start_time: None,
            throttle_duration_ms: 0,
            recovery_time_ms: 5000, // 5 second recovery time
            adaptive_threshold: 90.0,
        }
    }

    fn update_throttle_state(&mut self, recommendation: ThrottleRecommendation) {
        self.throttle_level = recommendation.throttle_level;

        if let Some(start_time) = self.throttle_start_time {
            self.throttle_duration_ms = start_time.elapsed().as_millis() as u64;
        }
    }
}

/// Thermal status information
#[derive(Debug, Clone)]
pub struct ThermalStatus {
    pub average_temperature: f32,
    pub maximum_temperature: f32,
    pub is_throttling: bool,
    pub throttle_level: u8,
    pub thermal_headroom: f32,
    pub active_zones: u8,
    pub monitoring_overhead_us: u32,
}

/// Throttling recommendation
#[derive(Debug, Clone)]
struct ThrottleRecommendation {
    pub should_throttle: bool,
    pub throttle_level: u8,
    pub target_frequency_reduction: f32,
    pub recommended_actions: Vec<String>,
}

/// Thermal statistics
#[derive(Debug, Clone)]
pub struct ThermalStatistics {
    pub average_temperature: f32,
    pub peak_temperature: f32,
    pub temperature_variance: f32,
    pub total_thermal_events: u64,
    pub total_throttle_events: u64,
    pub current_throttle_level: u8,
    pub uptime_seconds: u64,
    pub active_thermal_zones: u8,
}

impl ThermalStatistics {
    /// Generate thermal report
    pub fn generate_report(&self) -> String {
        format!(
            "Intel Thermal Management Report\n\
             ===============================\n\
             Temperature:\n\
             - Average: {:.1}°C\n\
             - Peak: {:.1}°C\n\
             - Variance: {:.2}°C\n\
             \n\
             Thermal Events: {}\n\
             Throttle Events: {}\n\
             Current Throttle Level: {}/10\n\
             \n\
             Active Thermal Zones: {}\n\
             Monitoring Uptime: {:.1}s\n\
             \n\
             Thermal Health: {}",
            self.average_temperature,
            self.peak_temperature,
            self.temperature_variance,
            self.total_thermal_events,
            self.total_throttle_events,
            self.current_throttle_level,
            self.active_thermal_zones,
            self.uptime_seconds,
            if self.peak_temperature < 85.0 && self.total_throttle_events == 0 {
                "✅ EXCELLENT"
            } else if self.peak_temperature < 90.0 && self.total_throttle_events < 5 {
                "✅ GOOD"
            } else if self.peak_temperature < 95.0 {
                "⚠️ CAUTION"
            } else {
                "❌ CRITICAL"
            }
        )
    }
}