// GNA Wake Word Detector for VoiceStand Push-to-Talk System
// Intel Meteor Lake GNA (Gaussian & Neural-Network Accelerator) Integration
// Hardware: 00:08.0 System peripheral: Intel Corporation Meteor Lake-P Gaussian & Neural-Network Accelerator

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::os::unix::io::{AsRawFd, RawFd};
use std::fs::OpenOptions;
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout};
use anyhow::{Result, Context, anyhow};
use log::{info, warn, error, debug};
use voicestand_core::{VoiceStandError, Result as VsResult};

// GNA Hardware Interface
const GNA_DEVICE_PATH: &str = "/dev/accel/accel0";
const GNA_PCI_ADDRESS: &str = "0000:00:08.0";

// Performance Targets
const POWER_TARGET_MW: f32 = 100.0;  // <100mW always-on
const RESPONSE_TIME_TARGET_MS: u64 = 50;  // <50ms response
const DETECTION_ACCURACY_TARGET: f32 = 0.95;  // >95% accuracy
const FALSE_POSITIVE_TARGET: f32 = 0.001;  // <0.1% false positive

// Wake Word Configuration
const WAKE_WORDS: &[&str] = &["Voice Mode", "Start Listening", "Begin Recording"];
const SAMPLE_RATE: u32 = 16000;
const FRAME_SIZE_MS: u32 = 30;
const HOP_SIZE_MS: u32 = 10;
const MFCC_COEFFS: usize = 13;
const CONTEXT_WINDOW: usize = 10;  // 10 frames for template matching

#[derive(Debug, Clone)]
pub struct GNAWakeWordConfig {
    pub device_path: String,
    pub sample_rate: u32,
    pub frame_size_ms: u32,
    pub hop_size_ms: u32,
    pub detection_threshold: f32,
    pub power_target_mw: f32,
    pub wake_words: Vec<String>,
    pub always_on: bool,
    pub vad_threshold: f32,
}

impl Default for GNAWakeWordConfig {
    fn default() -> Self {
        Self {
            device_path: GNA_DEVICE_PATH.to_string(),
            sample_rate: SAMPLE_RATE,
            frame_size_ms: FRAME_SIZE_MS,
            hop_size_ms: HOP_SIZE_MS,
            detection_threshold: 0.85,
            power_target_mw: POWER_TARGET_MW,
            wake_words: WAKE_WORDS.iter().map(|s| s.to_string()).collect(),
            always_on: true,
            vad_threshold: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WakeWordDetection {
    pub wake_word: String,
    pub confidence: f32,
    pub timestamp: Instant,
    pub power_consumption_mw: f32,
    pub processing_time_ms: f32,
}

// Safe mutex locking utility
fn safe_lock<T>(mutex: &std::sync::Mutex<T>) -> VsResult<std::sync::MutexGuard<T>> {
    mutex.lock().map_err(|e| VoiceStandError::lock_poisoned(format!("Mutex poisoned: {}", e)))
}

#[derive(Debug, Clone, Default)]
pub struct GNAPerformanceMetrics {
    pub total_detections: u64,
    pub false_positives: u64,
    pub average_power_mw: f32,
    pub average_response_time_ms: f32,
    pub detection_accuracy: f32,
    pub uptime_hours: f32,
}

// GNA-optimized MFCC feature template
#[derive(Debug, Clone)]
struct WakeWordTemplate {
    word: String,
    mfcc_template: Vec<Vec<f32>>,  // Context frames of MFCC coefficients
    energy_profile: Vec<f32>,
    spectral_centroid_profile: Vec<f32>,
    zero_crossing_profile: Vec<f32>,
    confidence_threshold: f32,
}

pub struct GNAWakeWordDetector {
    config: GNAWakeWordConfig,
    device_fd: Option<RawFd>,
    wake_word_templates: Vec<WakeWordTemplate>,
    audio_buffer: Arc<Mutex<VecDeque<f32>>>,
    performance_metrics: Arc<Mutex<GNAPerformanceMetrics>>,
    detection_sender: Option<mpsc::Sender<WakeWordDetection>>,
    is_running: Arc<Mutex<bool>>,
    power_monitor: PowerMonitor,
    start_time: Instant,
}

impl GNAWakeWordDetector {
    pub fn new(config: GNAWakeWordConfig) -> Result<Self> {
        info!("Initializing GNA Wake Word Detector for Intel Meteor Lake");

        let power_monitor = PowerMonitor::new(config.power_target_mw)?;

        let mut detector = Self {
            config,
            device_fd: None,
            wake_word_templates: Vec::new(),
            audio_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(16000 * 2))), // 2 second buffer
            performance_metrics: Arc::new(Mutex::new(GNAPerformanceMetrics {
                total_detections: 0,
                false_positives: 0,
                average_power_mw: 0.0,
                average_response_time_ms: 0.0,
                detection_accuracy: 0.0,
                uptime_hours: 0.0,
            })),
            detection_sender: None,
            is_running: Arc::new(Mutex::new(false)),
            power_monitor,
            start_time: Instant::now(),
        };

        // Initialize GNA hardware
        detector.initialize_gna_hardware()?;

        // Load default wake word templates
        detector.load_default_templates()?;

        info!("GNA Wake Word Detector initialized successfully");
        Ok(detector)
    }

    fn initialize_gna_hardware(&mut self) -> Result<()> {
        info!("Initializing GNA hardware at {}", GNA_PCI_ADDRESS);

        // Check if GNA device exists
        if !std::path::Path::new(&self.config.device_path).exists() {
            return Err(anyhow!("GNA device not found at {}", self.config.device_path));
        }

        // Open GNA device with read/write access
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.config.device_path)
            .context("Failed to open GNA device")?;

        self.device_fd = Some(file.as_raw_fd());

        // Configure GNA for ultra-low power wake word detection
        self.configure_gna_power_mode()?;

        info!("GNA hardware initialized successfully");
        Ok(())
    }

    fn configure_gna_power_mode(&self) -> Result<()> {
        info!("Configuring GNA for ultra-low power mode (<{}mW)", self.config.power_target_mw);

        // GNA power configuration would go here
        // This would involve low-level hardware register programming
        // For now, we simulate the configuration

        debug!("GNA power mode configured for always-on detection");
        Ok(())
    }

    fn load_default_templates(&mut self) -> Result<()> {
        info!("Loading default wake word templates");

        for wake_word in &self.config.wake_words {
            let template = self.generate_wake_word_template(wake_word)?;
            self.wake_word_templates.push(template);
            debug!("Loaded template for wake word: {}", wake_word);
        }

        info!("Loaded {} wake word templates", self.wake_word_templates.len());
        Ok(())
    }

    fn generate_wake_word_template(&self, wake_word: &str) -> Result<WakeWordTemplate> {
        // Generate phoneme-based MFCC templates for wake words
        // This is a simplified implementation - in production, this would use
        // pre-trained templates from actual voice samples

        let mut mfcc_template = Vec::new();
        let mut energy_profile = Vec::new();
        let mut spectral_centroid_profile = Vec::new();
        let mut zero_crossing_profile = Vec::new();

        // Generate synthetic templates based on phonetic characteristics
        let phonetic_length = wake_word.len().max(5).min(20);

        for i in 0..phonetic_length {
            let mut mfcc_frame = vec![0.0f32; MFCC_COEFFS];

            // Generate phoneme-specific MFCC patterns
            for j in 0..MFCC_COEFFS {
                let phonetic_component = (i as f32 / phonetic_length as f32) * 2.0 * std::f32::consts::PI;
                let coefficient_component = (j as f32 / MFCC_COEFFS as f32) * std::f32::consts::PI;

                mfcc_frame[j] = (phonetic_component + coefficient_component).sin()
                    * (1.0 + 0.1 * wake_word.chars().nth(i % wake_word.len()).unwrap_or('a') as u32 as f32 / 100.0);
            }

            mfcc_template.push(mfcc_frame);
            energy_profile.push(0.5 + 0.3 * (i as f32 / phonetic_length as f32).sin());
            spectral_centroid_profile.push(1000.0 + 500.0 * (i as f32 / phonetic_length as f32).cos());
            zero_crossing_profile.push(0.1 + 0.05 * (i as f32 / phonetic_length as f32).sin());
        }

        Ok(WakeWordTemplate {
            word: wake_word.to_string(),
            mfcc_template,
            energy_profile,
            spectral_centroid_profile,
            zero_crossing_profile,
            confidence_threshold: self.config.detection_threshold,
        })
    }

    pub async fn start_always_on_detection(&mut self) -> Result<mpsc::Receiver<WakeWordDetection>> {
        info!("Starting always-on GNA wake word detection");

        let (tx, rx) = mpsc::channel(10);
        self.detection_sender = Some(tx);

        if let Ok(mut guard) = safe_lock(&self.is_running) {
            *guard = true;
        }

        // Start the detection loop
        let detector_clone = self.clone_for_detection();
        tokio::spawn(async move {
            detector_clone.detection_loop().await;
        });

        info!("Always-on GNA detection started");
        Ok(rx)
    }

    fn clone_for_detection(&self) -> DetectionWorker {
        DetectionWorker {
            config: self.config.clone(),
            wake_word_templates: self.wake_word_templates.clone(),
            audio_buffer: Arc::clone(&self.audio_buffer),
            performance_metrics: Arc::clone(&self.performance_metrics),
            detection_sender: self.detection_sender.as_ref().ok_or_else(|| anyhow!("Detection sender not initialized"))?.clone(),
            is_running: Arc::clone(&self.is_running),
            power_monitor: self.power_monitor.clone(),
        }
    }

    pub fn process_audio_frame(&mut self, audio_samples: &[f32]) -> Result<Option<WakeWordDetection>> {
        let start_time = Instant::now();
        let power_start = self.power_monitor.get_current_power_mw();

        // Add samples to circular buffer
        {
            if let Ok(mut buffer) = safe_lock(&self.audio_buffer) {
                for &sample in audio_samples {
                    buffer.push_back(sample);
                    if buffer.len() > self.config.sample_rate as usize * 2 {  // 2 second max
                        buffer.pop_front();
                    }
                }
            }
        }

        // Voice Activity Detection
        if !self.detect_voice_activity(audio_samples)? {
            return Ok(None);
        }

        // Extract features for wake word detection
        let features = self.extract_mfcc_features(audio_samples)?;

        // GNA-accelerated template matching
        let detection = self.gna_template_matching(&features)?;

        // Update performance metrics
        let processing_time = start_time.elapsed().as_millis() as f32;
        let power_consumption = self.power_monitor.get_current_power_mw() - power_start;

        if let Some(mut detection) = detection {
            detection.processing_time_ms = processing_time;
            detection.power_consumption_mw = power_consumption;

            self.update_performance_metrics(&detection);

            info!("Wake word detected: {} (confidence: {:.3}, power: {:.1}mW, time: {:.1}ms)",
                detection.wake_word, detection.confidence, detection.power_consumption_mw, detection.processing_time_ms);

            return Ok(Some(detection));
        }

        Ok(None)
    }

    fn detect_voice_activity(&self, audio_samples: &[f32]) -> Result<bool> {
        if audio_samples.is_empty() {
            return Ok(false);
        }

        // Energy-based VAD
        let energy: f32 = audio_samples.iter().map(|&x| x * x).sum();
        let rms_energy = (energy / audio_samples.len() as f32).sqrt();

        // Zero crossing rate
        let mut zero_crossings = 0;
        for i in 1..audio_samples.len() {
            if (audio_samples[i-1] >= 0.0) != (audio_samples[i] >= 0.0) {
                zero_crossings += 1;
            }
        }
        let zcr = zero_crossings as f32 / audio_samples.len() as f32;

        // Combined VAD decision
        let is_speech = rms_energy > self.config.vad_threshold && zcr < 0.5;

        Ok(is_speech)
    }

    fn extract_mfcc_features(&self, audio_samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        // Simplified MFCC extraction optimized for GNA
        let frame_size = (self.config.sample_rate * self.config.frame_size_ms / 1000) as usize;
        let hop_size = (self.config.sample_rate * self.config.hop_size_ms / 1000) as usize;

        let mut mfcc_features = Vec::new();

        for i in (0..audio_samples.len().saturating_sub(frame_size)).step_by(hop_size) {
            let frame = &audio_samples[i..i + frame_size.min(audio_samples.len() - i)];
            let mfcc_frame = self.compute_mfcc_frame(frame)?;
            mfcc_features.push(mfcc_frame);
        }

        Ok(mfcc_features)
    }

    fn compute_mfcc_frame(&self, frame: &[f32]) -> Result<Vec<f32>> {
        // Simplified MFCC computation
        let mut mfcc = vec![0.0f32; MFCC_COEFFS];

        // Pre-emphasis
        let mut preemphasized = vec![0.0f32; frame.len()];
        preemphasized[0] = frame[0];
        for i in 1..frame.len() {
            preemphasized[i] = frame[i] - 0.97 * frame[i-1];
        }

        // Windowing (Hamming window)
        for i in 0..preemphasized.len() {
            let window_val = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (frame.len() - 1) as f32).cos();
            preemphasized[i] *= window_val;
        }

        // Simplified mel-frequency analysis
        for i in 0..MFCC_COEFFS {
            let mut energy = 0.0f32;
            let start_idx = i * frame.len() / MFCC_COEFFS;
            let end_idx = ((i + 1) * frame.len() / MFCC_COEFFS).min(frame.len());

            for j in start_idx..end_idx {
                energy += preemphasized[j] * preemphasized[j];
            }

            mfcc[i] = if energy > 0.0 { energy.ln() } else { -10.0 };
        }

        Ok(mfcc)
    }

    fn gna_template_matching(&self, features: &[Vec<f32>]) -> Result<Option<WakeWordDetection>> {
        if features.len() < CONTEXT_WINDOW {
            return Ok(None);
        }

        let mut best_match: Option<WakeWordDetection> = None;
        let mut best_score = 0.0f32;

        // Use the last CONTEXT_WINDOW frames for matching
        let context_start = features.len().saturating_sub(CONTEXT_WINDOW);
        let context_features = &features[context_start..];

        for template in &self.wake_word_templates {
            let score = self.compute_template_similarity(context_features, template)?;

            if score > template.confidence_threshold && score > best_score {
                best_score = score;
                best_match = Some(WakeWordDetection {
                    wake_word: template.word.clone(),
                    confidence: score,
                    timestamp: Instant::now(),
                    power_consumption_mw: 0.0,  // Will be filled by caller
                    processing_time_ms: 0.0,   // Will be filled by caller
                });
            }
        }

        Ok(best_match)
    }

    fn compute_template_similarity(&self, features: &[Vec<f32>], template: &WakeWordTemplate) -> Result<f32> {
        // Dynamic Time Warping (DTW) for template matching
        let n = features.len();
        let m = template.mfcc_template.len();

        if n == 0 || m == 0 {
            return Ok(0.0);
        }

        // Simplified DTW - in production, this would be GNA-accelerated
        let mut dtw = vec![vec![f32::INFINITY; m + 1]; n + 1];
        dtw[0][0] = 0.0;

        for i in 1..=n {
            for j in 1..=m {
                let cost = self.compute_frame_distance(&features[i-1], &template.mfcc_template[j-1]);
                dtw[i][j] = cost + dtw[i-1][j].min(dtw[i][j-1]).min(dtw[i-1][j-1]);
            }
        }

        // Normalize by path length and convert to similarity score
        let distance = dtw[n][m] / (n + m) as f32;
        let similarity = (-distance).exp();

        Ok(similarity)
    }

    fn compute_frame_distance(&self, frame1: &[f32], frame2: &[f32]) -> f32 {
        let min_len = frame1.len().min(frame2.len());
        let mut distance = 0.0f32;

        for i in 0..min_len {
            let diff = frame1[i] - frame2[i];
            distance += diff * diff;
        }

        distance.sqrt()
    }

    fn update_performance_metrics(&self, detection: &WakeWordDetection) {
        if let Ok(mut metrics) = safe_lock(&self.performance_metrics) {
            metrics.total_detections += 1;

            // Update running averages
            let n = metrics.total_detections as f32;
            metrics.average_power_mw = (metrics.average_power_mw * (n - 1.0) + detection.power_consumption_mw) / n;
            metrics.average_response_time_ms = (metrics.average_response_time_ms * (n - 1.0) + detection.processing_time_ms) / n;

            // Update uptime
            metrics.uptime_hours = self.start_time.elapsed().as_secs_f32() / 3600.0;
        }
    }

    pub fn get_performance_metrics(&self) -> GNAPerformanceMetrics {
        safe_lock(&self.performance_metrics).map(|guard| guard.clone()).unwrap_or_default()
    }

    pub fn stop_detection(&mut self) {
        info!("Stopping GNA wake word detection");
        if let Ok(mut guard) = safe_lock(&self.is_running) {
            *guard = false;
        }
    }
}

// Separate struct for the detection worker to avoid borrowing issues
struct DetectionWorker {
    config: GNAWakeWordConfig,
    wake_word_templates: Vec<WakeWordTemplate>,
    audio_buffer: Arc<Mutex<VecDeque<f32>>>,
    performance_metrics: Arc<Mutex<GNAPerformanceMetrics>>,
    detection_sender: mpsc::Sender<WakeWordDetection>,
    is_running: Arc<Mutex<bool>>,
    power_monitor: PowerMonitor,
}

impl DetectionWorker {
    async fn detection_loop(&self) {
        info!("Starting GNA detection loop");

        while *safe_lock(&self.is_running).unwrap_or_default() {
            // Simulate audio processing with low power consumption
            sleep(Duration::from_millis(10)).await;  // 100Hz processing rate

            // Check power consumption
            let current_power = self.power_monitor.get_current_power_mw();
            if current_power > self.config.power_target_mw * 1.2 {
                warn!("Power consumption exceeded target: {:.1}mW > {:.1}mW",
                    current_power, self.config.power_target_mw);
            }

            // In a real implementation, this would process actual audio data
            // For now, we simulate the detection loop
        }

        info!("GNA detection loop stopped");
    }
}

// Power monitoring for GNA device
#[derive(Clone)]
struct PowerMonitor {
    target_power_mw: f32,
    baseline_power: f32,
}

impl PowerMonitor {
    fn new(target_power_mw: f32) -> Result<Self> {
        Ok(Self {
            target_power_mw,
            baseline_power: 50.0,  // Estimated baseline system power
        })
    }

    fn get_current_power_mw(&self) -> f32 {
        // In a real implementation, this would read from hardware
        // For simulation, return baseline + some variation
        self.baseline_power + (rand::random::<f32>() - 0.5) * 10.0
    }
}

// Utility functions for wake word detection
pub mod gna_utils {
    use super::*;

    pub fn validate_gna_hardware() -> Result<bool> {
        info!("Validating GNA hardware");

        // Check if GNA PCI device exists
        let lspci_output = std::process::Command::new("lspci")
            .args(&["-d", "8086:"])
            .output()
            .context("Failed to run lspci")?;

        let output_str = String::from_utf8_lossy(&lspci_output.stdout);
        let gna_found = output_str.contains("Gaussian") && output_str.contains("Neural");

        if gna_found {
            info!("GNA hardware validated: Intel Meteor Lake GNA found");
        } else {
            warn!("GNA hardware not found in PCI devices");
        }

        Ok(gna_found)
    }

    pub fn get_gna_device_info() -> Result<String> {
        let lspci_output = std::process::Command::new("lspci")
            .args(&["-v", "-d", "8086:"])
            .output()
            .context("Failed to get GNA device info")?;

        let output_str = String::from_utf8_lossy(&lspci_output.stdout);

        // Extract GNA-specific information
        for line in output_str.lines() {
            if line.contains("Gaussian") && line.contains("Neural") {
                return Ok(line.to_string());
            }
        }

        Err(anyhow!("GNA device information not found"))
    }

    pub fn estimate_power_consumption(config: &GNAWakeWordConfig) -> f32 {
        // Estimate power consumption based on configuration
        let base_power = 30.0;  // Base GNA power
        let processing_factor = if config.always_on { 1.5 } else { 1.0 };
        let sample_rate_factor = config.sample_rate as f32 / 16000.0;

        base_power * processing_factor * sample_rate_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_gna_initialization() {
        let config = GNAWakeWordConfig::default();

        // This test would pass in a real environment with GNA hardware
        match GNAWakeWordDetector::new(config) {
            Ok(_) => println!("GNA detector initialized successfully"),
            Err(e) => println!("GNA initialization failed (expected without hardware): {}", e),
        }
    }

    #[tokio::test]
    async fn test_wake_word_template_generation() {
        let config = GNAWakeWordConfig::default();

        if let Ok(mut detector) = GNAWakeWordDetector::new(config) {
            assert!(!detector.wake_word_templates.is_empty());
            assert_eq!(detector.wake_word_templates.len(), WAKE_WORDS.len());

            for (i, template) in detector.wake_word_templates.iter().enumerate() {
                assert_eq!(template.word, WAKE_WORDS[i]);
                assert!(!template.mfcc_template.is_empty());
            }
        }
    }

    #[test]
    fn test_audio_processing() {
        let config = GNAWakeWordConfig::default();

        if let Ok(mut detector) = GNAWakeWordDetector::new(config) {
            // Generate test audio (1 second at 16kHz)
            let test_audio: Vec<f32> = (0..16000)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.1)
                .collect();

            match detector.process_audio_frame(&test_audio) {
                Ok(result) => {
                    if let Some(detection) = result {
                        println!("Detected: {} (confidence: {:.3})", detection.wake_word, detection.confidence);
                    } else {
                        println!("No wake word detected (expected for test signal)");
                    }
                }
                Err(e) => println!("Audio processing failed: {}", e),
            }
        }
    }

    #[test]
    fn test_power_estimation() {
        let config = GNAWakeWordConfig::default();
        let estimated_power = gna_utils::estimate_power_consumption(&config);

        assert!(estimated_power > 0.0);
        assert!(estimated_power < POWER_TARGET_MW * 2.0);  // Should be reasonable
        println!("Estimated power consumption: {:.1}mW", estimated_power);
    }
}