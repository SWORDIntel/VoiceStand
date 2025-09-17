use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use voicestand_core::{Result, VoiceStandError};
use voicestand_intel::{ThermalManager, ThermalPolicy, OptimizationProfile};
use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use tokio::time::interval;

/// Integrated thermal and power management system
pub struct ThermalPowerManager {
    thermal_manager: Arc<RwLock<ThermalManager>>,
    power_monitor: PowerMonitor,
    adaptive_controller: AdaptiveController,
    config: ThermalPowerConfig,
    performance_monitor: Arc<voicestand_core::PerformanceMonitor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPowerConfig {
    // Thermal limits
    pub throttle_temperature_celsius: u8,
    pub critical_temperature_celsius: u8,
    pub target_temperature_celsius: u8,

    // Power limits
    pub max_power_watts: f32,
    pub battery_power_limit_watts: f32,
    pub thermal_power_limit_watts: f32,

    // Control parameters
    pub monitoring_interval_ms: u64,
    pub thermal_time_constant_sec: f32,
    pub power_averaging_window_sec: u32,
    pub adaptive_learning_enabled: bool,

    // Performance trade-offs
    pub allow_performance_scaling: bool,
    pub min_performance_percent: u8,
    pub emergency_throttle_enabled: bool,
}

impl Default for ThermalPowerConfig {
    fn default() -> Self {
        Self {
            throttle_temperature_celsius: 85,
            critical_temperature_celsius: 100,
            target_temperature_celsius: 75,
            max_power_watts: 28.0,      // Meteor Lake TDP
            battery_power_limit_watts: 15.0,
            thermal_power_limit_watts: 20.0,
            monitoring_interval_ms: 250, // 4Hz monitoring
            thermal_time_constant_sec: 2.0,
            power_averaging_window_sec: 5,
            adaptive_learning_enabled: true,
            allow_performance_scaling: true,
            min_performance_percent: 50,
            emergency_throttle_enabled: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PowerMonitor {
    power_history: Arc<RwLock<VecDeque<PowerMeasurement>>>,
    current_power_watts: Arc<parking_lot::Mutex<f32>>,
    power_source: Arc<parking_lot::Mutex<PowerSource>>,
    battery_level: Arc<parking_lot::Mutex<Option<u8>>>,
    history_size: usize,
}

#[derive(Debug, Clone)]
pub struct PowerMeasurement {
    pub timestamp: Instant,
    pub total_power_watts: f32,
    pub cpu_power_watts: f32,
    pub gpu_power_watts: f32,
    pub npu_power_watts: f32,
    pub temperature_celsius: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PowerSource {
    AC,
    Battery,
    Unknown,
}

/// Adaptive thermal and power controller with ML-based optimization
pub struct AdaptiveController {
    thermal_history: VecDeque<ThermalMeasurement>,
    workload_patterns: VecDeque<WorkloadPattern>,
    prediction_model: ThermalPredictionModel,
    learning_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct ThermalMeasurement {
    pub timestamp: Instant,
    pub cpu_temp: u8,
    pub package_temp: u8,
    pub gpu_temp: u8,
    pub ambient_temp: u8,
    pub fan_speed_rpm: u16,
    pub thermal_throttling: bool,
}

#[derive(Debug, Clone)]
pub struct WorkloadPattern {
    pub timestamp: Instant,
    pub cpu_utilization: f32,
    pub npu_utilization: f32,
    pub audio_processing_rate: f32,
    pub inference_frequency: f32,
}

/// Simple thermal prediction model
pub struct ThermalPredictionModel {
    thermal_coefficients: [f32; 4], // CPU, GPU, NPU, ambient contributions
    time_constants: [f32; 3],       // Fast, medium, slow thermal responses
    prediction_horizon_sec: f32,
}

impl ThermalPowerManager {
    pub async fn new() -> Result<Self> {
        let config = ThermalPowerConfig::default();

        let thermal_manager = Arc::new(RwLock::new(ThermalManager::new()?));
        let power_monitor = PowerMonitor::new();
        let adaptive_controller = AdaptiveController::new();
        let performance_monitor = Arc::new(voicestand_core::PerformanceMonitor::new());

        let mut manager = Self {
            thermal_manager,
            power_monitor,
            adaptive_controller,
            config,
            performance_monitor,
        };

        // Start background monitoring
        manager.start_background_monitoring().await?;

        Ok(manager)
    }

    pub async fn start_background_monitoring(&mut self) -> Result<()> {
        let thermal_manager = self.thermal_manager.clone();
        let power_monitor = self.power_monitor.clone();
        let performance_monitor = self.performance_monitor.clone();
        let config = self.config.clone();

        // Spawn monitoring task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(config.monitoring_interval_ms));

            loop {
                interval.tick().await;

                // Update thermal state
                if let Ok(thermal_status) = thermal_manager.write().update_thermal_state().await {
                    // Update power measurements
                    let power_measurement = PowerMeasurement {
                        timestamp: Instant::now(),
                        total_power_watts: Self::estimate_total_power(&thermal_status),
                        cpu_power_watts: Self::estimate_cpu_power(&thermal_status),
                        gpu_power_watts: 0.5, // Minimal iGPU usage
                        npu_power_watts: 2.0,  // NPU when active
                        temperature_celsius: thermal_status.average_temperature as u8,
                    };

                    power_monitor.record_power_measurement(power_measurement);

                    // Record performance impact
                    if thermal_status.is_throttling {
                        performance_monitor.record_processing_error(); // Count as performance impact
                    }
                }
            }
        });

        Ok(())
    }

    /// Main thermal and power control loop
    pub async fn adaptive_control_loop(&mut self) -> Result<ThermalPowerAction> {
        // 1. Gather current state
        let thermal_status = self.thermal_manager.write().update_thermal_state().await?;
        let power_status = self.power_monitor.get_current_status();
        let performance_stats = self.performance_monitor.get_stats();

        // 2. Update adaptive controller
        self.adaptive_controller.update_measurements(
            &thermal_status,
            &power_status,
            &performance_stats
        );

        // 3. Predict future thermal state
        let thermal_prediction = if self.config.adaptive_learning_enabled {
            self.adaptive_controller.predict_thermal_future(5.0) // 5 second prediction
        } else {
            None
        };

        // 4. Determine necessary actions
        let action = self.determine_control_action(
            &thermal_status,
            &power_status,
            &performance_stats,
            thermal_prediction.as_ref()
        );

        // 5. Execute action
        self.execute_control_action(&action).await?;

        Ok(action)
    }

    fn determine_control_action(
        &self,
        thermal_status: &voicestand_intel::ThermalStatus,
        power_status: &PowerStatus,
        performance_stats: &voicestand_core::PerformanceStats,
        thermal_prediction: Option<&ThermalPrediction>,
    ) -> ThermalPowerAction {
        let current_temp = thermal_status.average_temperature as u8;

        // Emergency thermal protection
        if current_temp >= self.config.critical_temperature_celsius {
            return ThermalPowerAction::EmergencyThrottle {
                reason: "Critical temperature exceeded".to_string(),
                target_performance_percent: 25,
            };
        }

        // Predictive thermal management
        if let Some(prediction) = thermal_prediction {
            if prediction.predicted_max_temp > self.config.throttle_temperature_celsius {
                return ThermalPowerAction::PreventiveThrottle {
                    reason: format!("Predicted temperature {:.1}°C exceeds limit", prediction.predicted_max_temp),
                    target_performance_percent: 70,
                    prediction_confidence: prediction.confidence,
                };
            }
        }

        // Current thermal throttling
        if current_temp >= self.config.throttle_temperature_celsius {
            let throttle_intensity = ((current_temp - self.config.throttle_temperature_celsius) as f32 /
                                    (self.config.critical_temperature_celsius - self.config.throttle_temperature_celsius) as f32)
                                    .min(1.0);

            let target_performance = (100.0 - throttle_intensity * 50.0).max(50.0) as u8;

            return ThermalPowerAction::ThermalThrottle {
                reason: format!("Temperature {}°C exceeds throttle limit", current_temp),
                target_performance_percent: target_performance,
                throttle_intensity,
            };
        }

        // Power limit management
        if power_status.current_power_watts > power_status.power_limit_watts {
            let power_ratio = power_status.current_power_watts / power_status.power_limit_watts;
            let target_performance = (100.0 / power_ratio).max(60.0) as u8;

            return ThermalPowerAction::PowerThrottle {
                reason: format!("Power {:.1}W exceeds limit {:.1}W",
                              power_status.current_power_watts,
                              power_status.power_limit_watts),
                target_performance_percent: target_performance,
            };
        }

        // Performance optimization when thermal/power headroom available
        if current_temp < self.config.target_temperature_celsius &&
           power_status.current_power_watts < (power_status.power_limit_watts * 0.8) &&
           !performance_stats.meets_targets() {

            return ThermalPowerAction::PerformanceBoost {
                reason: "Thermal and power headroom available".to_string(),
                target_performance_percent: 110,
            };
        }

        // No action needed
        ThermalPowerAction::NoAction {
            reason: format!("Stable operation: {}°C, {:.1}W", current_temp, power_status.current_power_watts),
        }
    }

    async fn execute_control_action(&self, action: &ThermalPowerAction) -> Result<()> {
        match action {
            ThermalPowerAction::EmergencyThrottle { target_performance_percent, .. } => {
                // Maximum throttling
                self.thermal_manager.write().set_thermal_policy(ThermalPolicy::Conservative);
                self.apply_performance_scaling(*target_performance_percent).await?;
            },
            ThermalPowerAction::ThermalThrottle { target_performance_percent, .. } => {
                // Adaptive thermal throttling
                self.thermal_manager.write().set_thermal_policy(ThermalPolicy::Adaptive);
                self.apply_performance_scaling(*target_performance_percent).await?;
            },
            ThermalPowerAction::PreventiveThrottle { target_performance_percent, .. } => {
                // Predictive throttling
                self.thermal_manager.write().set_thermal_policy(ThermalPolicy::Adaptive);
                self.apply_performance_scaling(*target_performance_percent).await?;
            },
            ThermalPowerAction::PowerThrottle { target_performance_percent, .. } => {
                // Power-limited throttling
                self.apply_performance_scaling(*target_performance_percent).await?;
            },
            ThermalPowerAction::PerformanceBoost { target_performance_percent, .. } => {
                // Increase performance when headroom available
                self.thermal_manager.write().set_thermal_policy(ThermalPolicy::Aggressive);
                self.apply_performance_scaling(*target_performance_percent).await?;
            },
            ThermalPowerAction::NoAction { .. } => {
                // Maintain current state
                self.thermal_manager.write().set_thermal_policy(ThermalPolicy::Balanced);
            }
        }

        Ok(())
    }

    async fn apply_performance_scaling(&self, target_percent: u8) -> Result<()> {
        if !self.config.allow_performance_scaling {
            return Ok(());
        }

        let clamped_percent = target_percent.max(self.config.min_performance_percent).min(120);

        let optimization_profile = match clamped_percent {
            0..=50 => OptimizationProfile::PowerEfficient,
            51..=80 => OptimizationProfile::Balanced,
            81..=100 => OptimizationProfile::MaxPerformance,
            _ => OptimizationProfile::MaxPerformance, // Boost mode
        };

        // Apply CPU optimization changes (this would integrate with actual CPU optimizer)
        // For now, we just update the thermal manager
        self.thermal_manager.write().set_thermal_policy(
            match optimization_profile {
                OptimizationProfile::PowerEfficient => ThermalPolicy::Conservative,
                OptimizationProfile::Balanced => ThermalPolicy::Balanced,
                OptimizationProfile::MaxPerformance => ThermalPolicy::Aggressive,
            }
        );

        Ok(())
    }

    pub fn get_thermal_power_status(&self) -> ThermalPowerStatus {
        let power_status = self.power_monitor.get_current_status();

        ThermalPowerStatus {
            current_temperature_celsius: 75, // Would get from thermal manager
            power_consumption_watts: power_status.current_power_watts,
            power_source: power_status.power_source,
            battery_level_percent: power_status.battery_level,
            thermal_throttling_active: false, // Would get from thermal manager
            power_throttling_active: power_status.current_power_watts > power_status.power_limit_watts,
            thermal_headroom_celsius: self.config.throttle_temperature_celsius as i8 - 75,
            power_headroom_watts: power_status.power_limit_watts - power_status.current_power_watts,
            performance_scaling_percent: 100, // Current performance level
            last_update: Instant::now(),
        }
    }

    // Helper methods for power estimation
    fn estimate_total_power(thermal_status: &voicestand_intel::ThermalStatus) -> f32 {
        // Simple power estimation based on temperature
        let base_power = 8.0; // Idle power
        let thermal_power = (thermal_status.average_temperature - 40.0) * 0.3;
        base_power + thermal_power.max(0.0)
    }

    fn estimate_cpu_power(thermal_status: &voicestand_intel::ThermalStatus) -> f32 {
        // CPU power estimation
        let base_cpu_power = 5.0;
        let thermal_cpu_power = (thermal_status.average_temperature - 40.0) * 0.2;
        base_cpu_power + thermal_cpu_power.max(0.0)
    }
}

impl PowerMonitor {
    pub fn new() -> Self {
        Self {
            power_history: Arc::new(RwLock::new(VecDeque::new())),
            current_power_watts: Arc::new(parking_lot::Mutex::new(0.0)),
            power_source: Arc::new(parking_lot::Mutex::new(PowerSource::Unknown)),
            battery_level: Arc::new(parking_lot::Mutex::new(None)),
            history_size: 1000,
        }
    }

    pub fn record_power_measurement(&self, measurement: PowerMeasurement) {
        let mut history = self.power_history.write();
        if history.len() >= self.history_size {
            history.pop_front();
        }
        history.push_back(measurement.clone());

        *self.current_power_watts.lock() = measurement.total_power_watts;
    }

    pub fn get_current_status(&self) -> PowerStatus {
        let current_power = *self.current_power_watts.lock();
        let power_source = self.power_source.lock().clone();
        let battery_level = *self.battery_level.lock();

        let power_limit = match power_source {
            PowerSource::AC => 28.0,      // Full TDP on AC
            PowerSource::Battery => 15.0, // Reduced on battery
            PowerSource::Unknown => 20.0, // Conservative default
        };

        PowerStatus {
            current_power_watts: current_power,
            power_limit_watts: power_limit,
            power_source,
            battery_level,
            power_efficiency: current_power / power_limit, // 0.0 to 1.0+
        }
    }
}

impl AdaptiveController {
    pub fn new() -> Self {
        Self {
            thermal_history: VecDeque::new(),
            workload_patterns: VecDeque::new(),
            prediction_model: ThermalPredictionModel::new(),
            learning_enabled: true,
        }
    }

    pub fn update_measurements(
        &mut self,
        thermal_status: &voicestand_intel::ThermalStatus,
        power_status: &PowerStatus,
        performance_stats: &voicestand_core::PerformanceStats,
    ) {
        // Record thermal measurement
        let thermal_measurement = ThermalMeasurement {
            timestamp: Instant::now(),
            cpu_temp: thermal_status.average_temperature as u8,
            package_temp: thermal_status.average_temperature as u8,
            gpu_temp: 60, // Estimated
            ambient_temp: 25, // Estimated
            fan_speed_rpm: 0, // Not available
            thermal_throttling: thermal_status.is_throttling,
        };

        // Record workload pattern
        let workload_pattern = WorkloadPattern {
            timestamp: Instant::now(),
            cpu_utilization: performance_stats.cpu_usage_avg,
            npu_utilization: 0.5, // Estimated from performance stats
            audio_processing_rate: performance_stats.frames_per_second as f32,
            inference_frequency: 10.0, // Estimated
        };

        // Maintain history
        if self.thermal_history.len() >= 500 {
            self.thermal_history.pop_front();
        }
        self.thermal_history.push_back(thermal_measurement);

        if self.workload_patterns.len() >= 500 {
            self.workload_patterns.pop_front();
        }
        self.workload_patterns.push_back(workload_pattern);

        // Update prediction model if learning enabled
        if self.learning_enabled && self.thermal_history.len() > 10 {
            self.prediction_model.update_model(&self.thermal_history, &self.workload_patterns);
        }
    }

    pub fn predict_thermal_future(&self, horizon_sec: f32) -> Option<ThermalPrediction> {
        if self.thermal_history.len() < 5 {
            return None;
        }

        let prediction = self.prediction_model.predict_temperature(horizon_sec, &self.thermal_history);
        Some(prediction)
    }
}

impl ThermalPredictionModel {
    pub fn new() -> Self {
        Self {
            thermal_coefficients: [1.0, 0.5, 0.3, 0.1], // CPU dominant
            time_constants: [1.0, 5.0, 30.0],          // Fast, medium, slow responses
            prediction_horizon_sec: 10.0,
        }
    }

    pub fn update_model(
        &mut self,
        thermal_history: &VecDeque<ThermalMeasurement>,
        _workload_patterns: &VecDeque<WorkloadPattern>,
    ) {
        // Simple learning: adjust coefficients based on recent thermal behavior
        if thermal_history.len() >= 10 {
            let recent_temps: Vec<f32> = thermal_history
                .iter()
                .rev()
                .take(10)
                .map(|m| m.cpu_temp as f32)
                .collect();

            let temp_trend = recent_temps.first().unwrap() - recent_temps.last().unwrap();

            // Adjust coefficients based on trend
            if temp_trend.abs() > 2.0 {
                self.thermal_coefficients[0] *= 1.0 + (temp_trend * 0.01);
            }
        }
    }

    pub fn predict_temperature(
        &self,
        horizon_sec: f32,
        thermal_history: &VecDeque<ThermalMeasurement>,
    ) -> ThermalPrediction {
        if thermal_history.is_empty() {
            return ThermalPrediction {
                predicted_max_temp: 85,
                predicted_avg_temp: 75.0,
                confidence: 0.0,
                horizon_sec,
            };
        }

        let current_temp = thermal_history.back().unwrap().cpu_temp as f32;

        // Simple exponential prediction based on recent trend
        let recent_samples = thermal_history.iter().rev().take(5).collect::<Vec<_>>();
        let temp_trend = if recent_samples.len() >= 2 {
            (recent_samples[0].cpu_temp as f32 - recent_samples.last().unwrap().cpu_temp as f32) / recent_samples.len() as f32
        } else {
            0.0
        };

        let predicted_temp = current_temp + (temp_trend * horizon_sec);
        let confidence = (1.0 - (temp_trend.abs() * 0.1)).max(0.1).min(1.0);

        ThermalPrediction {
            predicted_max_temp: predicted_temp as u8,
            predicted_avg_temp: predicted_temp,
            confidence,
            horizon_sec,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PowerStatus {
    pub current_power_watts: f32,
    pub power_limit_watts: f32,
    pub power_source: PowerSource,
    pub battery_level: Option<u8>,
    pub power_efficiency: f32,
}

#[derive(Debug, Clone)]
pub struct ThermalPowerStatus {
    pub current_temperature_celsius: u8,
    pub power_consumption_watts: f32,
    pub power_source: PowerSource,
    pub battery_level_percent: Option<u8>,
    pub thermal_throttling_active: bool,
    pub power_throttling_active: bool,
    pub thermal_headroom_celsius: i8,
    pub power_headroom_watts: f32,
    pub performance_scaling_percent: u8,
    pub last_update: Instant,
}

#[derive(Debug, Clone)]
pub struct ThermalPrediction {
    pub predicted_max_temp: u8,
    pub predicted_avg_temp: f32,
    pub confidence: f32,
    pub horizon_sec: f32,
}

#[derive(Debug, Clone)]
pub enum ThermalPowerAction {
    NoAction {
        reason: String,
    },
    ThermalThrottle {
        reason: String,
        target_performance_percent: u8,
        throttle_intensity: f32,
    },
    PowerThrottle {
        reason: String,
        target_performance_percent: u8,
    },
    PreventiveThrottle {
        reason: String,
        target_performance_percent: u8,
        prediction_confidence: f32,
    },
    EmergencyThrottle {
        reason: String,
        target_performance_percent: u8,
    },
    PerformanceBoost {
        reason: String,
        target_performance_percent: u8,
    },
}

impl ThermalPowerAction {
    pub fn requires_immediate_action(&self) -> bool {
        matches!(self, ThermalPowerAction::EmergencyThrottle { .. })
    }

    pub fn affects_performance(&self) -> bool {
        !matches!(self, ThermalPowerAction::NoAction { .. })
    }

    pub fn get_description(&self) -> String {
        match self {
            ThermalPowerAction::NoAction { reason } => format!("Stable: {}", reason),
            ThermalPowerAction::ThermalThrottle { reason, target_performance_percent, .. } =>
                format!("Thermal throttle to {}%: {}", target_performance_percent, reason),
            ThermalPowerAction::PowerThrottle { reason, target_performance_percent } =>
                format!("Power throttle to {}%: {}", target_performance_percent, reason),
            ThermalPowerAction::PreventiveThrottle { reason, target_performance_percent, confidence } =>
                format!("Predictive throttle to {}% ({:.1}% confidence): {}", target_performance_percent, confidence * 100.0, reason),
            ThermalPowerAction::EmergencyThrottle { reason, target_performance_percent } =>
                format!("EMERGENCY throttle to {}%: {}", target_performance_percent, reason),
            ThermalPowerAction::PerformanceBoost { reason, target_performance_percent } =>
                format!("Performance boost to {}%: {}", target_performance_percent, reason),
        }
    }
}