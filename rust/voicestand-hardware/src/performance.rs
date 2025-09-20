//! Performance Monitoring and Metrics
//!
//! Comprehensive performance tracking for NPU and GNA operations with memory safety.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::error::{HardwareError, HardwareResult};

/// Hardware performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    /// Component name (NPU, GNA, etc.)
    pub component: String,
    /// Total operations performed
    pub total_operations: u64,
    /// Average operation latency in milliseconds
    pub average_latency_ms: f32,
    /// Peak latency in milliseconds
    pub peak_latency_ms: f32,
    /// Minimum latency in milliseconds
    pub min_latency_ms: f32,
    /// Operations per second
    pub ops_per_second: f32,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Memory usage in megabytes
    pub memory_usage_mb: f32,
    /// Power consumption in milliwatts
    pub power_consumption_mw: f32,
    /// Error count
    pub error_count: u64,
    /// Success rate (0.0 - 1.0)
    pub success_rate: f32,
    /// Performance targets met
    pub targets_met: bool,
    /// Additional component-specific metrics
    pub additional_metrics: HashMap<String, MetricValue>,
    /// Timestamp of last update
    pub last_updated: SystemTime,
}

/// Generic metric value type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
    Duration(Duration),
}

impl MetricValue {
    /// Convert to float if possible
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Convert to integer if possible
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Convert to boolean if possible
    pub fn as_bool(&self) -> Option<bool> {
        if let Self::Boolean(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Convert to string
    pub fn as_string(&self) -> String {
        match self {
            Self::Float(v) => v.to_string(),
            Self::Integer(v) => v.to_string(),
            Self::Boolean(v) => v.to_string(),
            Self::String(v) => v.clone(),
            Self::Duration(v) => format!("{:.3}ms", v.as_secs_f64() * 1000.0),
        }
    }
}

impl Default for HardwareMetrics {
    fn default() -> Self {
        Self {
            component: "unknown".to_string(),
            total_operations: 0,
            average_latency_ms: 0.0,
            peak_latency_ms: 0.0,
            min_latency_ms: f32::INFINITY,
            ops_per_second: 0.0,
            total_processing_time: Duration::ZERO,
            memory_usage_mb: 0.0,
            power_consumption_mw: 0.0,
            error_count: 0,
            success_rate: 1.0,
            targets_met: false,
            additional_metrics: HashMap::new(),
            last_updated: SystemTime::now(),
        }
    }
}

impl HardwareMetrics {
    /// Create new metrics for component
    pub fn new(component: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            min_latency_ms: f32::INFINITY,
            success_rate: 1.0,
            last_updated: SystemTime::now(),
            ..Default::default()
        }
    }

    /// Update latency statistics
    pub fn update_latency(&mut self, latency: Duration) {
        let latency_ms = latency.as_secs_f32() * 1000.0;

        // Update running average
        if self.total_operations == 0 {
            self.average_latency_ms = latency_ms;
        } else {
            // Exponential moving average with alpha = 0.1
            self.average_latency_ms = 0.9 * self.average_latency_ms + 0.1 * latency_ms;
        }

        // Update peak and min
        if latency_ms > self.peak_latency_ms {
            self.peak_latency_ms = latency_ms;
        }
        if latency_ms < self.min_latency_ms {
            self.min_latency_ms = latency_ms;
        }

        self.total_processing_time += latency;
        self.last_updated = SystemTime::now();
    }

    /// Update throughput statistics
    pub fn update_throughput(&mut self, operations: u64, duration: Duration) {
        self.total_operations += operations;

        if duration.as_secs_f32() > 0.0 {
            let new_ops_per_sec = operations as f32 / duration.as_secs_f32();

            // Update running average of ops/sec
            if self.ops_per_second == 0.0 {
                self.ops_per_second = new_ops_per_sec;
            } else {
                self.ops_per_second = 0.9 * self.ops_per_second + 0.1 * new_ops_per_sec;
            }
        }

        self.last_updated = SystemTime::now();
    }

    /// Record successful operation
    pub fn record_success(&mut self, latency: Duration) {
        self.update_latency(latency);
        self.update_throughput(1, latency);
        self.update_success_rate();
    }

    /// Record failed operation
    pub fn record_error(&mut self) {
        self.error_count += 1;
        self.total_operations += 1;
        self.update_success_rate();
        self.last_updated = SystemTime::now();
    }

    /// Update success rate calculation
    fn update_success_rate(&mut self) {
        if self.total_operations > 0 {
            let successful_ops = self.total_operations - self.error_count;
            self.success_rate = successful_ops as f32 / self.total_operations as f32;
        }
    }

    /// Add memory usage information
    pub fn add_memory_usage(&mut self, memory_mb: f32) {
        self.memory_usage_mb = memory_mb;
        self.last_updated = SystemTime::now();
    }

    /// Add power consumption information
    pub fn add_power_consumption(&mut self, component: &str, power_mw: f32) {
        self.power_consumption_mw = power_mw;
        self.additional_metrics.insert(
            format!("{}_power_mw", component),
            MetricValue::Float(power_mw as f64),
        );
        self.last_updated = SystemTime::now();
    }

    /// Add custom metric
    pub fn add_metric(&mut self, key: impl Into<String>, value: MetricValue) {
        self.additional_metrics.insert(key.into(), value);
        self.last_updated = SystemTime::now();
    }

    /// Merge metrics from another component
    pub fn merge_npu_metrics(&mut self, npu_metrics: HardwareMetrics) {
        // Add NPU-specific metrics
        self.additional_metrics.insert(
            "npu_total_operations".to_string(),
            MetricValue::Integer(npu_metrics.total_operations as i64),
        );
        self.additional_metrics.insert(
            "npu_average_latency_ms".to_string(),
            MetricValue::Float(npu_metrics.average_latency_ms as f64),
        );
        self.additional_metrics.insert(
            "npu_ops_per_second".to_string(),
            MetricValue::Float(npu_metrics.ops_per_second as f64),
        );
        self.additional_metrics.insert(
            "npu_success_rate".to_string(),
            MetricValue::Float(npu_metrics.success_rate as f64),
        );

        // Merge all additional metrics from NPU
        for (key, value) in npu_metrics.additional_metrics {
            self.additional_metrics.insert(format!("npu_{}", key), value);
        }

        self.last_updated = SystemTime::now();
    }

    /// Merge metrics from GNA component
    pub fn merge_gna_metrics(&mut self, gna_metrics: HardwareMetrics) {
        // Add GNA-specific metrics
        self.additional_metrics.insert(
            "gna_total_operations".to_string(),
            MetricValue::Integer(gna_metrics.total_operations as i64),
        );
        self.additional_metrics.insert(
            "gna_average_latency_ms".to_string(),
            MetricValue::Float(gna_metrics.average_latency_ms as f64),
        );
        self.additional_metrics.insert(
            "gna_power_consumption_mw".to_string(),
            MetricValue::Float(gna_metrics.power_consumption_mw as f64),
        );
        self.additional_metrics.insert(
            "gna_success_rate".to_string(),
            MetricValue::Float(gna_metrics.success_rate as f64),
        );

        // Merge all additional metrics from GNA
        for (key, value) in gna_metrics.additional_metrics {
            self.additional_metrics.insert(format!("gna_{}", key), value);
        }

        self.last_updated = SystemTime::now();
    }

    /// Check if performance targets are met
    pub fn check_targets(&mut self, target_latency_ms: f32, target_ops_per_sec: f32) {
        self.targets_met = self.average_latency_ms <= target_latency_ms
            && self.ops_per_second >= target_ops_per_sec
            && self.success_rate >= 0.95; // 95% success rate requirement
    }

    /// Generate human-readable performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("=== {} Performance Report ===\n", self.component));
        report.push_str(&format!("Total Operations: {}\n", self.total_operations));
        report.push_str(&format!("Average Latency: {:.2}ms\n", self.average_latency_ms));
        report.push_str(&format!("Peak Latency: {:.2}ms\n", self.peak_latency_ms));
        report.push_str(&format!("Min Latency: {:.2}ms\n", self.min_latency_ms));
        report.push_str(&format!("Throughput: {:.1} ops/sec\n", self.ops_per_second));
        report.push_str(&format!("Success Rate: {:.1}%\n", self.success_rate * 100.0));
        report.push_str(&format!("Error Count: {}\n", self.error_count));
        report.push_str(&format!("Memory Usage: {:.1} MB\n", self.memory_usage_mb));
        report.push_str(&format!("Power Consumption: {:.1} mW\n", self.power_consumption_mw));
        report.push_str(&format!("Targets Met: {}\n", if self.targets_met { "✅" } else { "❌" }));

        if !self.additional_metrics.is_empty() {
            report.push_str("\nAdditional Metrics:\n");
            for (key, value) in &self.additional_metrics {
                report.push_str(&format!("  {}: {}\n", key, value.as_string()));
            }
        }

        report
    }
}

/// Performance tracker for specific hardware component
pub struct PerformanceTracker {
    component: String,
    metrics: Arc<RwLock<HardwareMetrics>>,
    start_time: Instant,
    operation_history: Arc<RwLock<Vec<OperationRecord>>>,
}

/// Individual operation record for detailed analysis
#[derive(Debug, Clone)]
struct OperationRecord {
    timestamp: Instant,
    operation_type: String,
    latency: Duration,
    success: bool,
    additional_data: HashMap<String, MetricValue>,
}

impl PerformanceTracker {
    /// Create new performance tracker
    pub fn new(component: impl Into<String>) -> Self {
        let component = component.into();
        let metrics = Arc::new(RwLock::new(HardwareMetrics::new(component.clone())));

        Self {
            component,
            metrics,
            start_time: Instant::now(),
            operation_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record NPU inference operation
    pub fn record_inference(
        &self,
        total_time: Duration,
        inference_time: Duration,
        audio_samples: usize,
        confidence: f32,
    ) {
        let mut metrics = self.metrics.write();

        metrics.record_success(total_time);
        metrics.add_metric("inference_time_ms", MetricValue::Duration(inference_time));
        metrics.add_metric("audio_samples", MetricValue::Integer(audio_samples as i64));
        metrics.add_metric("confidence", MetricValue::Float(confidence as f64));
        metrics.add_metric("samples_per_ms", MetricValue::Float(
            audio_samples as f64 / total_time.as_secs_f64() / 1000.0
        ));

        // Check NPU performance targets (<2ms inference)
        metrics.check_targets(2.0, 50.0); // 2ms latency, 50 inferences/sec

        // Record in operation history
        let mut history = self.operation_history.write();
        history.push(OperationRecord {
            timestamp: Instant::now(),
            operation_type: "npu_inference".to_string(),
            latency: total_time,
            success: true,
            additional_data: {
                let mut data = HashMap::new();
                data.insert("inference_time_ms".to_string(), MetricValue::Duration(inference_time));
                data.insert("confidence".to_string(), MetricValue::Float(confidence as f64));
                data
            },
        });

        // Keep only last 1000 operations
        if history.len() > 1000 {
            history.remove(0);
        }
    }

    /// Record GNA wake word detection
    pub fn record_wake_word_detection(
        &self,
        total_time: Duration,
        detection_time: Duration,
        confidence: f32,
        power_mw: f32,
    ) {
        let mut metrics = self.metrics.write();

        metrics.record_success(total_time);
        metrics.add_metric("detection_time_ms", MetricValue::Duration(detection_time));
        metrics.add_metric("confidence", MetricValue::Float(confidence as f64));
        metrics.add_power_consumption("GNA", power_mw);

        // Check GNA performance targets (<100mW power)
        let power_target_met = power_mw <= 100.0;
        metrics.add_metric("power_target_met", MetricValue::Boolean(power_target_met));
        metrics.check_targets(50.0, 20.0); // 50ms latency, 20 detections/sec

        // Record in operation history
        let mut history = self.operation_history.write();
        history.push(OperationRecord {
            timestamp: Instant::now(),
            operation_type: "gna_wake_word_detection".to_string(),
            latency: total_time,
            success: true,
            additional_data: {
                let mut data = HashMap::new();
                data.insert("detection_time_ms".to_string(), MetricValue::Duration(detection_time));
                data.insert("confidence".to_string(), MetricValue::Float(confidence as f64));
                data.insert("power_mw".to_string(), MetricValue::Float(power_mw as f64));
                data
            },
        });

        // Keep only last 1000 operations
        if history.len() > 1000 {
            history.remove(0);
        }
    }

    /// Record error operation
    pub fn record_error(&self, operation_type: &str, error: &str) {
        let mut metrics = self.metrics.write();
        metrics.record_error();

        // Record in operation history
        let mut history = self.operation_history.write();
        history.push(OperationRecord {
            timestamp: Instant::now(),
            operation_type: operation_type.to_string(),
            latency: Duration::ZERO,
            success: false,
            additional_data: {
                let mut data = HashMap::new();
                data.insert("error".to_string(), MetricValue::String(error.to_string()));
                data
            },
        });

        // Keep only last 1000 operations
        if history.len() > 1000 {
            history.remove(0);
        }

        warn!("Performance tracker recorded error: {} - {}", operation_type, error);
    }

    /// Get current metrics (thread-safe)
    pub fn get_current_metrics(&self) -> HardwareMetrics {
        self.metrics.read().clone()
    }

    /// Get performance summary over time window
    pub fn get_summary(&self, time_window: Duration) -> PerformanceSummary {
        let history = self.operation_history.read();
        let cutoff_time = Instant::now() - time_window;

        let recent_operations: Vec<_> = history
            .iter()
            .filter(|op| op.timestamp >= cutoff_time)
            .collect();

        let total_operations = recent_operations.len();
        let successful_operations = recent_operations.iter().filter(|op| op.success).count();
        let failed_operations = total_operations - successful_operations;

        let average_latency = if successful_operations > 0 {
            let total_latency: Duration = recent_operations
                .iter()
                .filter(|op| op.success)
                .map(|op| op.latency)
                .sum();
            total_latency / successful_operations as u32
        } else {
            Duration::ZERO
        };

        let ops_per_second = if time_window.as_secs_f32() > 0.0 {
            total_operations as f32 / time_window.as_secs_f32()
        } else {
            0.0
        };

        PerformanceSummary {
            component: self.component.clone(),
            time_window,
            total_operations,
            successful_operations,
            failed_operations,
            average_latency,
            ops_per_second,
            success_rate: if total_operations > 0 {
                successful_operations as f32 / total_operations as f32
            } else {
                0.0
            },
        }
    }

    /// Reset all metrics and history
    pub fn reset(&self) {
        let mut metrics = self.metrics.write();
        *metrics = HardwareMetrics::new(self.component.clone());

        let mut history = self.operation_history.write();
        history.clear();

        info!("Performance tracker reset for component: {}", self.component);
    }
}

/// Performance summary over a time window
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub component: String,
    pub time_window: Duration,
    pub total_operations: usize,
    pub successful_operations: usize,
    pub failed_operations: usize,
    pub average_latency: Duration,
    pub ops_per_second: f32,
    pub success_rate: f32,
}

impl PerformanceSummary {
    /// Generate human-readable summary report
    pub fn generate_report(&self) -> String {
        format!(
            "=== {} Performance Summary ({:.1}s window) ===\n\
             Total Operations: {}\n\
             Successful: {} ({:.1}%)\n\
             Failed: {} ({:.1}%)\n\
             Average Latency: {:.2}ms\n\
             Throughput: {:.1} ops/sec\n",
            self.component,
            self.time_window.as_secs_f32(),
            self.total_operations,
            self.successful_operations,
            self.success_rate * 100.0,
            self.failed_operations,
            (1.0 - self.success_rate) * 100.0,
            self.average_latency.as_secs_f32() * 1000.0,
            self.ops_per_second
        )
    }
}

/// Performance monitor for aggregating metrics across components
pub struct PerformanceMonitor {
    trackers: Arc<RwLock<HashMap<String, Arc<PerformanceTracker>>>>,
    monitor_tx: Option<mpsc::Sender<MonitorCommand>>,
    monitor_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Commands for performance monitor
#[derive(Debug)]
enum MonitorCommand {
    Stop,
    GetMetrics { response_tx: mpsc::Sender<HashMap<String, HardwareMetrics>> },
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            trackers: Arc::new(RwLock::new(HashMap::new())),
            monitor_tx: None,
            monitor_handle: None,
        }
    }

    /// Start performance monitoring background task
    pub async fn start(&mut self) -> HardwareResult<()> {
        if self.monitor_tx.is_some() {
            return Ok(()); // Already started
        }

        let (tx, mut rx) = mpsc::channel(32);
        let trackers = Arc::clone(&self.trackers);

        let monitor_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Periodic monitoring tasks
                        let trackers_guard = trackers.read();
                        for (component, tracker) in trackers_guard.iter() {
                            let metrics = tracker.get_current_metrics();

                            // Log warnings for performance issues
                            if metrics.success_rate < 0.95 {
                                warn!("{} success rate below 95%: {:.1}%",
                                      component, metrics.success_rate * 100.0);
                            }

                            if metrics.average_latency_ms > 100.0 {
                                warn!("{} average latency high: {:.2}ms",
                                      component, metrics.average_latency_ms);
                            }
                        }
                    }

                    cmd = rx.recv() => {
                        match cmd {
                            Some(MonitorCommand::Stop) => {
                                info!("Performance monitor stopping");
                                break;
                            }
                            Some(MonitorCommand::GetMetrics { response_tx }) => {
                                let metrics = {
                                    let trackers_guard = trackers.read();
                                    let mut metrics = HashMap::new();

                                    for (component, tracker) in trackers_guard.iter() {
                                        metrics.insert(component.clone(), tracker.get_current_metrics());
                                    }
                                    metrics
                                }; // trackers_guard dropped here

                                let _ = response_tx.send(metrics).await;
                            }
                            None => break,
                        }
                    }
                }
            }

            info!("Performance monitor stopped");
        });

        self.monitor_tx = Some(tx);
        self.monitor_handle = Some(monitor_handle);

        info!("✅ Performance monitor started");
        Ok(())
    }

    /// Stop performance monitoring
    pub async fn stop(&mut self) -> HardwareResult<()> {
        if let Some(tx) = self.monitor_tx.take() {
            let _ = tx.send(MonitorCommand::Stop).await;
        }

        if let Some(handle) = self.monitor_handle.take() {
            let _ = handle.await;
        }

        info!("✅ Performance monitor stopped");
        Ok(())
    }

    /// Add performance tracker for component
    pub fn add_tracker(&self, component: impl Into<String>, tracker: Arc<PerformanceTracker>) {
        let component = component.into();
        let mut trackers = self.trackers.write();
        trackers.insert(component.clone(), tracker);
        debug!("Added performance tracker for component: {}", component);
    }

    /// Get current metrics for all components
    pub async fn get_current_metrics(&self) -> HardwareResult<HardwareMetrics> {
        if let Some(tx) = &self.monitor_tx {
            let (response_tx, mut response_rx) = mpsc::channel(1);

            tx.send(MonitorCommand::GetMetrics { response_tx })
                .await
                .map_err(|_| HardwareError::concurrency_error(
                    "monitor_command",
                    "Failed to send metrics request",
                ))?;

            if let Some(metrics_map) = response_rx.recv().await {
                // Aggregate metrics from all components
                let mut aggregated = HardwareMetrics::new("system");

                for (component, metrics) in metrics_map {
                    if component == "NPU" {
                        aggregated.merge_npu_metrics(metrics);
                    } else if component == "GNA" {
                        aggregated.merge_gna_metrics(metrics);
                    }
                }

                Ok(aggregated)
            } else {
                Err(HardwareError::concurrency_error(
                    "metrics_aggregation",
                    "Failed to receive metrics response",
                ))
            }
        } else {
            // Monitor not started, return basic metrics
            let trackers = self.trackers.read();
            let mut aggregated = HardwareMetrics::new("system");

            for (component, tracker) in trackers.iter() {
                let metrics = tracker.get_current_metrics();
                if component == "NPU" {
                    aggregated.merge_npu_metrics(metrics);
                } else if component == "GNA" {
                    aggregated.merge_gna_metrics(metrics);
                }
            }

            Ok(aggregated)
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_metrics_creation() {
        let metrics = HardwareMetrics::new("NPU");
        assert_eq!(metrics.component, "NPU");
        assert_eq!(metrics.total_operations, 0);
        assert_eq!(metrics.success_rate, 1.0);
        assert_eq!(metrics.min_latency_ms, f32::INFINITY);
    }

    #[test]
    fn test_metrics_latency_update() {
        let mut metrics = HardwareMetrics::new("test");
        let latency = Duration::from_millis(5);

        metrics.update_latency(latency);
        assert_eq!(metrics.average_latency_ms, 5.0);
        assert_eq!(metrics.peak_latency_ms, 5.0);
        assert_eq!(metrics.min_latency_ms, 5.0);

        // Test second update
        let latency2 = Duration::from_millis(10);
        metrics.update_latency(latency2);
        assert!(metrics.average_latency_ms > 5.0 && metrics.average_latency_ms < 10.0);
        assert_eq!(metrics.peak_latency_ms, 10.0);
        assert_eq!(metrics.min_latency_ms, 5.0);
    }

    #[test]
    fn test_metric_value_conversions() {
        let float_val = MetricValue::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(float_val.as_integer(), Some(3));

        let int_val = MetricValue::Integer(42);
        assert_eq!(int_val.as_integer(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));

        let bool_val = MetricValue::Boolean(true);
        assert_eq!(bool_val.as_bool(), Some(true));
        assert_eq!(bool_val.as_string(), "true");
    }

    #[tokio::test]
    async fn test_performance_tracker() {
        let tracker = PerformanceTracker::new("test");

        // Record successful operation
        tracker.record_inference(
            Duration::from_millis(2),
            Duration::from_millis(1),
            1600,
            0.95,
        );

        let metrics = tracker.get_current_metrics();
        assert_eq!(metrics.total_operations, 1);
        assert_eq!(metrics.error_count, 0);
        assert_eq!(metrics.success_rate, 1.0);
        assert_eq!(metrics.average_latency_ms, 2.0);
    }

    #[tokio::test]
    async fn test_performance_summary() {
        let tracker = PerformanceTracker::new("test");

        // Record multiple operations
        for _ in 0..10 {
            tracker.record_inference(
                Duration::from_millis(1),
                Duration::from_millis(1),
                1600,
                0.9,
            );
        }

        let summary = tracker.get_summary(Duration::from_secs(60));
        assert_eq!(summary.total_operations, 10);
        assert_eq!(summary.successful_operations, 10);
        assert_eq!(summary.failed_operations, 0);
        assert_eq!(summary.success_rate, 1.0);
    }
}