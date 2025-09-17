use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Performance monitoring system for VoiceStand
pub struct PerformanceMonitor {
    // Latency tracking
    audio_latencies: Arc<RwLock<VecDeque<Duration>>>,
    processing_latencies: Arc<RwLock<VecDeque<Duration>>>,

    // Throughput tracking
    samples_processed: AtomicU64,
    frames_processed: AtomicU64,

    // Memory tracking
    current_memory_usage: AtomicUsize,
    peak_memory_usage: AtomicUsize,
    allocation_count: AtomicU64,

    // CPU tracking
    cpu_usage_history: Arc<RwLock<VecDeque<f32>>>,

    // Error tracking
    buffer_overruns: AtomicU64,
    processing_errors: AtomicU64,

    // Configuration
    history_size: usize,
    start_time: Instant,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            audio_latencies: Arc::new(RwLock::new(VecDeque::new())),
            processing_latencies: Arc::new(RwLock::new(VecDeque::new())),
            samples_processed: AtomicU64::new(0),
            frames_processed: AtomicU64::new(0),
            current_memory_usage: AtomicUsize::new(0),
            peak_memory_usage: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
            cpu_usage_history: Arc::new(RwLock::new(VecDeque::new())),
            buffer_overruns: AtomicU64::new(0),
            processing_errors: AtomicU64::new(0),
            history_size: 1000,
            start_time: Instant::now(),
        }
    }

    /// Record audio processing latency
    pub fn record_audio_latency(&self, latency: Duration) {
        let mut latencies = self.audio_latencies.write();
        if latencies.len() >= self.history_size {
            latencies.pop_front();
        }
        latencies.push_back(latency);
    }

    /// Record processing stage latency
    pub fn record_processing_latency(&self, latency: Duration) {
        let mut latencies = self.processing_latencies.write();
        if latencies.len() >= self.history_size {
            latencies.pop_front();
        }
        latencies.push_back(latency);
    }

    /// Update throughput counters
    pub fn record_samples_processed(&self, count: u64) {
        self.samples_processed.fetch_add(count, Ordering::Relaxed);
    }

    pub fn record_frame_processed(&self) {
        self.frames_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Update memory usage
    pub fn record_memory_usage(&self, bytes: usize) {
        self.current_memory_usage.store(bytes, Ordering::Relaxed);

        // Update peak if necessary
        let current_peak = self.peak_memory_usage.load(Ordering::Relaxed);
        if bytes > current_peak {
            self.peak_memory_usage.store(bytes, Ordering::Relaxed);
        }
    }

    pub fn record_allocation(&self) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record CPU usage
    pub fn record_cpu_usage(&self, usage_percent: f32) {
        let mut history = self.cpu_usage_history.write();
        if history.len() >= self.history_size {
            history.pop_front();
        }
        history.push_back(usage_percent);
    }

    /// Record error events
    pub fn record_buffer_overrun(&self) {
        self.buffer_overruns.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_processing_error(&self) {
        self.processing_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get comprehensive performance statistics
    pub fn get_stats(&self) -> PerformanceStats {
        let audio_latencies = self.audio_latencies.read();
        let processing_latencies = self.processing_latencies.read();
        let cpu_history = self.cpu_usage_history.read();

        PerformanceStats {
            // Latency stats
            audio_latency_avg: calculate_average_duration(&*audio_latencies),
            audio_latency_p95: calculate_percentile_duration(&*audio_latencies, 95.0),
            audio_latency_p99: calculate_percentile_duration(&*audio_latencies, 99.0),

            processing_latency_avg: calculate_average_duration(&*processing_latencies),
            processing_latency_p95: calculate_percentile_duration(&*processing_latencies, 95.0),
            processing_latency_p99: calculate_percentile_duration(&*processing_latencies, 99.0),

            // Throughput stats
            samples_per_second: self.calculate_samples_per_second(),
            frames_per_second: self.calculate_frames_per_second(),

            // Memory stats
            current_memory_mb: self.current_memory_usage.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0,
            peak_memory_mb: self.peak_memory_usage.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0,
            total_allocations: self.allocation_count.load(Ordering::Relaxed),

            // CPU stats
            cpu_usage_avg: calculate_average(&*cpu_history),
            cpu_usage_max: cpu_history.iter().cloned().fold(0.0f32, f32::max),

            // Error stats
            buffer_overruns: self.buffer_overruns.load(Ordering::Relaxed),
            processing_errors: self.processing_errors.load(Ordering::Relaxed),

            // Runtime stats
            uptime: self.start_time.elapsed(),
        }
    }

    /// Check if performance targets are being met
    pub fn check_performance_targets(&self) -> PerformanceTargetStatus {
        let stats = self.get_stats();

        PerformanceTargetStatus {
            latency_target_met: stats.audio_latency_p95.as_millis() < 50, // <50ms target
            memory_target_met: stats.peak_memory_mb < 100.0, // <100MB target
            error_rate_acceptable: {
                let total_frames = stats.frames_per_second * stats.uptime.as_secs_f64();
                let error_rate = (stats.buffer_overruns + stats.processing_errors) as f64 / total_frames.max(1.0);
                error_rate < 0.001 // <0.1% error rate
            },
            cpu_usage_acceptable: stats.cpu_usage_avg < 80.0, // <80% average CPU
        }
    }

    fn calculate_samples_per_second(&self) -> f64 {
        let samples = self.samples_processed.load(Ordering::Relaxed) as f64;
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 { samples / elapsed } else { 0.0 }
    }

    fn calculate_frames_per_second(&self) -> f64 {
        let frames = self.frames_processed.load(Ordering::Relaxed) as f64;
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 { frames / elapsed } else { 0.0 }
    }

    /// Reset all counters (for testing)
    pub fn reset(&self) {
        self.audio_latencies.write().clear();
        self.processing_latencies.write().clear();
        self.samples_processed.store(0, Ordering::Relaxed);
        self.frames_processed.store(0, Ordering::Relaxed);
        self.current_memory_usage.store(0, Ordering::Relaxed);
        self.peak_memory_usage.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.cpu_usage_history.write().clear();
        self.buffer_overruns.store(0, Ordering::Relaxed);
        self.processing_errors.store(0, Ordering::Relaxed);
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    // Latency metrics (in milliseconds for display)
    pub audio_latency_avg: Duration,
    pub audio_latency_p95: Duration,
    pub audio_latency_p99: Duration,
    pub processing_latency_avg: Duration,
    pub processing_latency_p95: Duration,
    pub processing_latency_p99: Duration,

    // Throughput metrics
    pub samples_per_second: f64,
    pub frames_per_second: f64,

    // Memory metrics (in MB)
    pub current_memory_mb: f64,
    pub peak_memory_mb: f64,
    pub total_allocations: u64,

    // CPU metrics (percentage)
    pub cpu_usage_avg: f32,
    pub cpu_usage_max: f32,

    // Error metrics
    pub buffer_overruns: u64,
    pub processing_errors: u64,

    // Runtime metrics
    pub uptime: Duration,
}

impl PerformanceStats {
    /// Check if all performance targets are met
    pub fn meets_targets(&self) -> bool {
        self.audio_latency_p95.as_millis() < 50 &&
        self.peak_memory_mb < 100.0 &&
        self.cpu_usage_avg < 80.0
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        format!(
            "VoiceStand Performance Report\n\
             =============================\n\
             Latency:\n\
             - Audio P95: {:.2}ms (target: <50ms)\n\
             - Audio P99: {:.2}ms\n\
             - Processing P95: {:.2}ms\n\
             \n\
             Throughput:\n\
             - Samples/sec: {:.0}\n\
             - Frames/sec: {:.1}\n\
             \n\
             Memory:\n\
             - Current: {:.1}MB\n\
             - Peak: {:.1}MB (target: <100MB)\n\
             - Allocations: {}\n\
             \n\
             CPU:\n\
             - Average: {:.1}% (target: <80%)\n\
             - Maximum: {:.1}%\n\
             \n\
             Errors:\n\
             - Buffer overruns: {}\n\
             - Processing errors: {}\n\
             \n\
             Runtime: {:.1}s\n\
             Target Compliance: {}",
            self.audio_latency_p95.as_millis(),
            self.audio_latency_p99.as_millis(),
            self.processing_latency_p95.as_millis(),
            self.samples_per_second,
            self.frames_per_second,
            self.current_memory_mb,
            self.peak_memory_mb,
            self.total_allocations,
            self.cpu_usage_avg,
            self.cpu_usage_max,
            self.buffer_overruns,
            self.processing_errors,
            self.uptime.as_secs_f64(),
            if self.meets_targets() { "✅ PASSED" } else { "❌ FAILED" }
        )
    }
}

/// Performance target status
#[derive(Debug, Clone)]
pub struct PerformanceTargetStatus {
    pub latency_target_met: bool,
    pub memory_target_met: bool,
    pub error_rate_acceptable: bool,
    pub cpu_usage_acceptable: bool,
}

impl PerformanceTargetStatus {
    pub fn all_targets_met(&self) -> bool {
        self.latency_target_met &&
        self.memory_target_met &&
        self.error_rate_acceptable &&
        self.cpu_usage_acceptable
    }
}

/// Timer utility for measuring performance
pub struct PerformanceTimer {
    start: Instant,
    monitor: Arc<PerformanceMonitor>,
    timer_type: TimerType,
}

#[derive(Debug, Clone, Copy)]
pub enum TimerType {
    AudioProcessing,
    GeneralProcessing,
}

impl PerformanceTimer {
    pub fn start(monitor: Arc<PerformanceMonitor>, timer_type: TimerType) -> Self {
        Self {
            start: Instant::now(),
            monitor,
            timer_type,
        }
    }

    pub fn finish(self) {
        let elapsed = self.start.elapsed();
        match self.timer_type {
            TimerType::AudioProcessing => self.monitor.record_audio_latency(elapsed),
            TimerType::GeneralProcessing => self.monitor.record_processing_latency(elapsed),
        }
    }
}

// Helper functions for statistical calculations
fn calculate_average_duration(durations: &VecDeque<Duration>) -> Duration {
    if durations.is_empty() {
        return Duration::from_secs(0);
    }

    let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
    Duration::from_nanos(total_nanos / durations.len() as u64)
}

fn calculate_percentile_duration(durations: &VecDeque<Duration>, percentile: f64) -> Duration {
    if durations.is_empty() {
        return Duration::from_secs(0);
    }

    let mut sorted: Vec<Duration> = durations.iter().cloned().collect();
    sorted.sort();

    let index = ((percentile / 100.0) * (sorted.len() - 1) as f64) as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn calculate_average(values: &VecDeque<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = PerformanceMonitor::new();
        let stats = monitor.get_stats();
        assert_eq!(stats.samples_per_second, 0.0);
        assert_eq!(stats.buffer_overruns, 0);
    }

    #[test]
    fn test_latency_recording() {
        let monitor = PerformanceMonitor::new();

        monitor.record_audio_latency(Duration::from_millis(10));
        monitor.record_audio_latency(Duration::from_millis(20));
        monitor.record_audio_latency(Duration::from_millis(30));

        let stats = monitor.get_stats();
        assert!(stats.audio_latency_avg.as_millis() > 0);
        assert!(stats.audio_latency_p95.as_millis() >= 20);
    }

    #[test]
    fn test_throughput_tracking() {
        let monitor = PerformanceMonitor::new();

        // Simulate some processing
        thread::sleep(Duration::from_millis(100));
        monitor.record_samples_processed(1600); // 0.1 second at 16kHz
        monitor.record_frame_processed();

        let stats = monitor.get_stats();
        assert!(stats.samples_per_second > 0.0);
        assert!(stats.frames_per_second > 0.0);
    }

    #[test]
    fn test_memory_tracking() {
        let monitor = PerformanceMonitor::new();

        monitor.record_memory_usage(1024 * 1024); // 1MB
        monitor.record_allocation();

        let stats = monitor.get_stats();
        assert_eq!(stats.current_memory_mb, 1.0);
        assert_eq!(stats.total_allocations, 1);
    }

    #[test]
    fn test_performance_timer() {
        let monitor = Arc::new(PerformanceMonitor::new());

        {
            let timer = PerformanceTimer::start(monitor.clone(), TimerType::AudioProcessing);
            thread::sleep(Duration::from_millis(10));
            timer.finish();
        }

        let stats = monitor.get_stats();
        assert!(stats.audio_latency_avg.as_millis() >= 10);
    }

    #[test]
    fn test_target_compliance() {
        let monitor = PerformanceMonitor::new();

        // Record good performance
        monitor.record_audio_latency(Duration::from_millis(5)); // Well under 50ms
        monitor.record_memory_usage(50 * 1024 * 1024); // 50MB, under 100MB
        monitor.record_cpu_usage(60.0); // Under 80%

        let targets = monitor.check_performance_targets();
        assert!(targets.latency_target_met);
        assert!(targets.memory_target_met);
        assert!(targets.cpu_usage_acceptable);
    }
}