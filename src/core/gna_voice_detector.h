#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <cmath>
#include <limits>

// OpenVINO includes for GNA backend
#ifdef ENABLE_OPENVINO
#include <openvino/openvino.hpp>
#endif

namespace vtt {

/**
 * @brief GNA Voice Detector - Ultra-low power voice activity detection
 *
 * Implements always-on voice detection using Intel GNA (Gaussian Neural Accelerator)
 * for continuous operation at <0.05W power consumption with >90% accuracy.
 *
 * Features:
 * - Hardware-accelerated VAD on Intel GNA
 * - Wake word detection with DTW template matching
 * - Power-optimized audio preprocessing
 * - 50ms response time to NPU handoff
 * - Thermal-aware operation
 */

struct GNAConfig {
    // Audio parameters
    uint32_t sample_rate = 16000;           // 16kHz audio input
    uint32_t frame_size = 320;              // 20ms frames (320 samples @ 16kHz)
    uint32_t hop_size = 160;                // 10ms hop (50% overlap)

    // VAD parameters
    float vad_threshold = 0.35f;            // Voice activity threshold
    float energy_threshold = 0.01f;         // Minimum energy threshold
    uint32_t min_speech_frames = 3;         // Minimum frames for speech detection
    uint32_t max_silence_frames = 10;       // Max silence before speech end

    // Wake word parameters
    std::vector<std::string> wake_words = {"voicestand", "hey voice"};
    float wake_word_threshold = 0.8f;       // Wake word confidence threshold
    uint32_t wake_word_timeout_ms = 5000;   // Wake word validity timeout

    // Power management
    float max_power_watts = 0.05f;          // Maximum GNA power consumption
    uint32_t thermal_throttle_temp = 75;    // Thermal throttling temperature (°C)
    bool enable_power_gating = true;        // Enable aggressive power gating

    // GNA device configuration
    std::string gna_device = "/dev/accel/accel0";
    std::string gna_model_path = "models/gna_vad.xml";
    uint32_t gna_precision = 16;            // 16-bit fixed point precision
};

struct GNADetectionResult {
    bool voice_detected = false;            // Voice activity detected
    bool wake_word_detected = false;        // Wake word detected
    float confidence = 0.0f;                // Detection confidence [0.0-1.0]
    std::string wake_word = "";             // Detected wake word (if any)
    uint64_t timestamp_us = 0;              // Detection timestamp (microseconds)
    float power_consumption_mw = 0.0f;      // Current power consumption (milliwatts)
    uint32_t temperature_celsius = 0;       // GNA temperature
};

struct AudioFeatures {
    std::vector<float> mfcc;                // 13 MFCC coefficients
    std::vector<float> energy;              // Frame energy
    std::vector<float> spectral_centroid;   // Spectral features
    float zcr = 0.0f;                       // Zero crossing rate
    float spectral_rolloff = 0.0f;          // Spectral rolloff point
};

using GNADetectionCallback = std::function<void(const GNADetectionResult&)>;
using GNAPowerCallback = std::function<void(float power_mw, uint32_t temp_c)>;

/**
 * @brief Power-optimized feature extractor for GNA processing
 */
class GNAFeatureExtractor {
public:
    explicit GNAFeatureExtractor(const GNAConfig& config);

    // Extract features optimized for GNA inference
    AudioFeatures extract_features(const float* samples, size_t num_samples);

    // Pre-emphasis filter for speech enhancement
    void apply_preemphasis(float* samples, size_t num_samples, float alpha = 0.97f);

    // Efficient windowing with power optimization
    void apply_hamming_window(float* samples, size_t num_samples);

    // Fast MFCC computation optimized for GNA
    std::vector<float> compute_mfcc(const float* samples, size_t num_samples);

private:
    GNAConfig config_;
    std::vector<float> hamming_window_;
    std::vector<float> mel_filterbank_;
    std::vector<float> dct_matrix_;

    // Pre-computed tables for efficiency
    void init_mel_filterbank();
    void init_dct_matrix();
};

/**
 * @brief GNA Power Manager - Monitor and optimize power consumption
 */
class GNAPowerManager {
public:
    explicit GNAPowerManager(const GNAConfig& config);
    ~GNAPowerManager();

    // Start power monitoring
    bool start_monitoring();
    void stop_monitoring();

    // Power state management
    void set_power_gating(bool enabled);
    void set_thermal_throttling(bool enabled);

    // Power consumption monitoring
    float get_current_power_mw() const;
    uint32_t get_temperature_celsius() const;
    bool is_thermal_throttled() const;

    // Power optimization
    void optimize_for_idle();
    void optimize_for_detection();
    void emergency_power_reduction();

    void set_power_callback(GNAPowerCallback callback);

private:
    GNAConfig config_;
    std::atomic<bool> monitoring_active_;
    std::atomic<float> current_power_mw_;
    std::atomic<uint32_t> current_temp_c_;
    std::atomic<bool> thermal_throttled_;

    std::thread monitoring_thread_;
    GNAPowerCallback power_callback_;
    mutable std::mutex callback_mutex_;

    void monitoring_loop();
    void read_power_consumption();
    void read_temperature();
    void apply_thermal_throttling();
};

/**
 * @brief Main GNA Voice Detector class
 */
class GNAVoiceDetector {
public:
    explicit GNAVoiceDetector(const GNAConfig& config = {});
    ~GNAVoiceDetector();

    // Lifecycle management
    bool initialize();
    bool start_detection();
    void stop_detection();
    void shutdown();

    // Audio input processing
    bool process_audio(const float* samples, size_t num_samples);

    // Callback registration
    void set_detection_callback(GNADetectionCallback callback);
    void set_power_callback(GNAPowerCallback callback);

    // Status and control
    bool is_running() const { return detection_active_; }
    GNADetectionResult get_last_result() const;

    // Wake word management
    bool add_wake_word(const std::string& word, const std::vector<float>& template_features);
    bool remove_wake_word(const std::string& word);
    void clear_wake_words();

    // Power optimization
    void set_power_mode(const std::string& mode); // "ultra_low", "balanced", "performance"
    float get_power_consumption_mw() const;

    // Model management
    bool load_gna_model(const std::string& model_path);
    bool is_model_loaded() const { return model_loaded_; }

private:
    GNAConfig config_;
    std::atomic<bool> initialized_;
    std::atomic<bool> detection_active_;
    std::atomic<bool> model_loaded_;

    // OpenVINO GNA components
#ifdef ENABLE_OPENVINO
    std::unique_ptr<ov::Core> ov_core_;
    std::unique_ptr<ov::CompiledModel> compiled_model_;
    std::unique_ptr<ov::InferRequest> infer_request_;
#endif

    // Feature extraction and power management
    std::unique_ptr<GNAFeatureExtractor> feature_extractor_;
    std::unique_ptr<GNAPowerManager> power_manager_;

    // Audio processing buffers
    std::vector<float> audio_buffer_;
    std::vector<float> feature_buffer_;

    // Wake word templates
    std::map<std::string, std::vector<float>> wake_word_templates_;

    // Detection state
    GNADetectionResult last_result_;
    mutable std::mutex result_mutex_;

    // Callbacks
    GNADetectionCallback detection_callback_;
    mutable std::mutex callback_mutex_;

    // Processing thread
    std::thread processing_thread_;
    std::atomic<bool> processing_active_;
    std::condition_variable processing_cv_;
    std::mutex processing_mutex_;

    // Internal methods
    bool initialize_openvino();
    bool load_model_to_gna();
    void processing_loop();

    // VAD and wake word detection
    bool detect_voice_activity(const AudioFeatures& features);
    bool detect_wake_word(const AudioFeatures& features, std::string& detected_word);
    float compute_wake_word_similarity(const std::vector<float>& features,
                                     const std::vector<float>& template_features);

    // Power optimization methods
    void optimize_gna_power();
    void handle_thermal_event();

    // Utility methods
    uint64_t get_timestamp_us() const;
    void notify_detection_result(const GNADetectionResult& result);
};

} // namespace vtt