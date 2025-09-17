#include "gna_voice_detector.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>

// System includes for power management
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace vtt {

namespace {
    // Constants for audio processing
    constexpr float PI = 3.14159265359f;
    constexpr float TWO_PI = 2.0f * PI;
    constexpr int NUM_MFCC_COEFFS = 13;
    constexpr int NUM_MEL_FILTERS = 26;
    constexpr float MEL_LOW_FREQ = 80.0f;
    constexpr float MEL_HIGH_FREQ = 8000.0f;

    // Power management constants
    constexpr const char* GNA_POWER_SYSFS = "/sys/class/drm/card0/device/power_state";
    constexpr const char* GNA_TEMP_SYSFS = "/sys/class/thermal/thermal_zone0/temp";
    constexpr uint32_t POWER_MONITORING_INTERVAL_MS = 100;

    // Utility functions
    inline float hz_to_mel(float hz) {
        return 1127.0f * std::log(1.0f + hz / 700.0f);
    }

    inline float mel_to_hz(float mel) {
        return 700.0f * (std::exp(mel / 1127.0f) - 1.0f);
    }

    inline uint64_t get_timestamp_microseconds() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    }
}

// GNAFeatureExtractor Implementation
GNAFeatureExtractor::GNAFeatureExtractor(const GNAConfig& config) : config_(config) {
    // Initialize Hamming window
    hamming_window_.resize(config_.frame_size);
    for (size_t i = 0; i < config_.frame_size; ++i) {
        hamming_window_[i] = 0.54f - 0.46f * std::cos(TWO_PI * i / (config_.frame_size - 1));
    }

    init_mel_filterbank();
    init_dct_matrix();
}

AudioFeatures GNAFeatureExtractor::extract_features(const float* samples, size_t num_samples) {
    AudioFeatures features;

    if (num_samples != config_.frame_size) {
        // Handle frame size mismatch
        return features;
    }

    // Create working copy of samples
    std::vector<float> work_samples(samples, samples + num_samples);

    // Apply pre-emphasis
    apply_preemphasis(work_samples.data(), num_samples);

    // Apply windowing
    apply_hamming_window(work_samples.data(), num_samples);

    // Compute MFCC features
    features.mfcc = compute_mfcc(work_samples.data(), num_samples);

    // Compute energy
    float energy = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += work_samples[i] * work_samples[i];
    }
    features.energy = {energy / num_samples};

    // Compute zero crossing rate
    int zero_crossings = 0;
    for (size_t i = 1; i < num_samples; ++i) {
        if ((work_samples[i] >= 0.0f) != (work_samples[i-1] >= 0.0f)) {
            zero_crossings++;
        }
    }
    features.zcr = static_cast<float>(zero_crossings) / (num_samples - 1);

    return features;
}

void GNAFeatureExtractor::apply_preemphasis(float* samples, size_t num_samples, float alpha) {
    for (size_t i = num_samples - 1; i > 0; --i) {
        samples[i] -= alpha * samples[i-1];
    }
}

void GNAFeatureExtractor::apply_hamming_window(float* samples, size_t num_samples) {
    for (size_t i = 0; i < num_samples && i < hamming_window_.size(); ++i) {
        samples[i] *= hamming_window_[i];
    }
}

std::vector<float> GNAFeatureExtractor::compute_mfcc(const float* samples, size_t num_samples) {
    // Simplified MFCC computation optimized for GNA
    std::vector<float> mfcc(NUM_MFCC_COEFFS, 0.0f);

    // FFT and Mel filterbank application would go here
    // For now, use simplified spectral features
    for (int i = 0; i < NUM_MFCC_COEFFS; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < num_samples; ++j) {
            sum += samples[j] * std::cos(PI * i * j / num_samples);
        }
        mfcc[i] = sum / num_samples;
    }

    return mfcc;
}

void GNAFeatureExtractor::init_mel_filterbank() {
    // Initialize mel filterbank for MFCC computation
    mel_filterbank_.resize(NUM_MEL_FILTERS * config_.frame_size / 2);

    float mel_low = hz_to_mel(MEL_LOW_FREQ);
    float mel_high = hz_to_mel(MEL_HIGH_FREQ);
    float mel_step = (mel_high - mel_low) / (NUM_MEL_FILTERS + 1);

    // Create triangular filters
    for (int i = 0; i < NUM_MEL_FILTERS; ++i) {
        float mel_center = mel_low + (i + 1) * mel_step;
        float hz_center = mel_to_hz(mel_center);

        // Simplified triangular filter implementation
        for (size_t j = 0; j < config_.frame_size / 2; ++j) {
            float hz = static_cast<float>(j) * config_.sample_rate / config_.frame_size;
            float response = std::max(0.0f, 1.0f - std::abs(hz - hz_center) / (config_.sample_rate / 20.0f));
            mel_filterbank_[i * config_.frame_size / 2 + j] = response;
        }
    }
}

void GNAFeatureExtractor::init_dct_matrix() {
    // Initialize DCT matrix for MFCC computation
    dct_matrix_.resize(NUM_MFCC_COEFFS * NUM_MEL_FILTERS);

    for (int i = 0; i < NUM_MFCC_COEFFS; ++i) {
        for (int j = 0; j < NUM_MEL_FILTERS; ++j) {
            dct_matrix_[i * NUM_MEL_FILTERS + j] =
                std::cos(PI * i * (j + 0.5f) / NUM_MEL_FILTERS) *
                std::sqrt(2.0f / NUM_MEL_FILTERS);
        }
    }
}

// GNAPowerManager Implementation
GNAPowerManager::GNAPowerManager(const GNAConfig& config)
    : config_(config), monitoring_active_(false), current_power_mw_(0.0f),
      current_temp_c_(0), thermal_throttled_(false) {
}

GNAPowerManager::~GNAPowerManager() {
    stop_monitoring();
}

bool GNAPowerManager::start_monitoring() {
    if (monitoring_active_.load()) {
        return true;
    }

    monitoring_active_ = true;
    monitoring_thread_ = std::thread(&GNAPowerManager::monitoring_loop, this);
    return true;
}

void GNAPowerManager::stop_monitoring() {
    monitoring_active_ = false;
    if (monitoring_thread_.joinable()) {
        monitoring_thread_.join();
    }
}

float GNAPowerManager::get_current_power_mw() const {
    return current_power_mw_.load();
}

uint32_t GNAPowerManager::get_temperature_celsius() const {
    return current_temp_c_.load();
}

bool GNAPowerManager::is_thermal_throttled() const {
    return thermal_throttled_.load();
}

void GNAPowerManager::set_power_callback(GNAPowerCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    power_callback_ = std::move(callback);
}

void GNAPowerManager::monitoring_loop() {
    while (monitoring_active_.load()) {
        read_power_consumption();
        read_temperature();
        apply_thermal_throttling();

        // Notify callback if registered
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (power_callback_) {
                power_callback_(current_power_mw_.load(), current_temp_c_.load());
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(POWER_MONITORING_INTERVAL_MS));
    }
}

void GNAPowerManager::read_power_consumption() {
    // Read power consumption from sysfs (simplified implementation)
    std::ifstream power_file(GNA_POWER_SYSFS);
    if (power_file.is_open()) {
        float power;
        power_file >> power;
        current_power_mw_ = power;
    } else {
        // Fallback: estimate based on activity
        current_power_mw_ = 30.0f; // Estimated 30mW baseline
    }
}

void GNAPowerManager::read_temperature() {
    // Read temperature from thermal zone
    std::ifstream temp_file(GNA_TEMP_SYSFS);
    if (temp_file.is_open()) {
        int temp_millidegrees;
        temp_file >> temp_millidegrees;
        current_temp_c_ = temp_millidegrees / 1000;
    } else {
        current_temp_c_ = 45; // Default safe temperature
    }
}

void GNAPowerManager::apply_thermal_throttling() {
    uint32_t temp = current_temp_c_.load();
    bool should_throttle = temp > config_.thermal_throttle_temp;

    if (should_throttle != thermal_throttled_.load()) {
        thermal_throttled_ = should_throttle;
        if (should_throttle) {
            emergency_power_reduction();
        }
    }
}

void GNAPowerManager::optimize_for_idle() {
    // Reduce GNA to minimum power state
    // Implementation would involve GNA driver calls
}

void GNAPowerManager::optimize_for_detection() {
    // Set GNA to balanced power/performance state
}

void GNAPowerManager::emergency_power_reduction() {
    // Emergency power reduction for thermal protection
    optimize_for_idle();
}

// GNAVoiceDetector Implementation
GNAVoiceDetector::GNAVoiceDetector(const GNAConfig& config)
    : config_(config), initialized_(false), detection_active_(false),
      model_loaded_(false), processing_active_(false) {

    feature_extractor_ = std::make_unique<GNAFeatureExtractor>(config_);
    power_manager_ = std::make_unique<GNAPowerManager>(config_);

    // Initialize audio buffers
    audio_buffer_.resize(config_.frame_size);
    feature_buffer_.resize(NUM_MFCC_COEFFS);
}

GNAVoiceDetector::~GNAVoiceDetector() {
    shutdown();
}

bool GNAVoiceDetector::initialize() {
    if (initialized_.load()) {
        return true;
    }

    // Initialize OpenVINO for GNA
    if (!initialize_openvino()) {
        std::cerr << "Failed to initialize OpenVINO for GNA" << std::endl;
        return false;
    }

    // Start power management
    if (!power_manager_->start_monitoring()) {
        std::cerr << "Failed to start power monitoring" << std::endl;
        return false;
    }

    // Load default model if specified
    if (!config_.gna_model_path.empty()) {
        if (!load_gna_model(config_.gna_model_path)) {
            std::cerr << "Warning: Failed to load default GNA model" << std::endl;
        }
    }

    initialized_ = true;
    return true;
}

bool GNAVoiceDetector::start_detection() {
    if (!initialized_.load()) {
        return false;
    }

    if (detection_active_.load()) {
        return true;
    }

    processing_active_ = true;
    processing_thread_ = std::thread(&GNAVoiceDetector::processing_loop, this);
    detection_active_ = true;

    return true;
}

void GNAVoiceDetector::stop_detection() {
    detection_active_ = false;
    processing_active_ = false;

    processing_cv_.notify_all();

    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

void GNAVoiceDetector::shutdown() {
    stop_detection();
    power_manager_->stop_monitoring();
    initialized_ = false;
}

bool GNAVoiceDetector::process_audio(const float* samples, size_t num_samples) {
    if (!detection_active_.load() || num_samples != config_.frame_size) {
        return false;
    }

    // Copy audio samples to buffer
    std::memcpy(audio_buffer_.data(), samples, num_samples * sizeof(float));

    // Signal processing thread
    processing_cv_.notify_one();

    return true;
}

void GNAVoiceDetector::set_detection_callback(GNADetectionCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    detection_callback_ = std::move(callback);
}

void GNAVoiceDetector::set_power_callback(GNAPowerCallback callback) {
    if (power_manager_) {
        power_manager_->set_power_callback(std::move(callback));
    }
}

GNADetectionResult GNAVoiceDetector::get_last_result() const {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return last_result_;
}

bool GNAVoiceDetector::load_gna_model(const std::string& model_path) {
#ifdef ENABLE_OPENVINO
    try {
        if (!ov_core_) {
            return false;
        }

        // Load model
        auto model = ov_core_->read_model(model_path);

        // Compile for GNA device
        compiled_model_ = std::make_unique<ov::CompiledModel>(
            ov_core_->compile_model(model, "GNA")
        );

        // Create inference request
        infer_request_ = std::make_unique<ov::InferRequest>(
            compiled_model_->create_infer_request()
        );

        model_loaded_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load GNA model: " << e.what() << std::endl;
        return false;
    }
#else
    // OpenVINO not available - use fallback detection
    std::cout << "OpenVINO not available, using fallback detection" << std::endl;
    model_loaded_ = true;
    return true;
#endif
}

float GNAVoiceDetector::get_power_consumption_mw() const {
    return power_manager_->get_current_power_mw();
}

bool GNAVoiceDetector::initialize_openvino() {
#ifdef ENABLE_OPENVINO
    try {
        ov_core_ = std::make_unique<ov::Core>();

        // Check for GNA device availability
        auto available_devices = ov_core_->get_available_devices();
        bool gna_available = std::find(available_devices.begin(),
                                      available_devices.end(),
                                      "GNA") != available_devices.end();

        if (!gna_available) {
            std::cerr << "GNA device not available in OpenVINO" << std::endl;
            return false;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "OpenVINO initialization failed: " << e.what() << std::endl;
        return false;
    }
#else
    // OpenVINO not compiled in - use software fallback
    std::cout << "OpenVINO not available, using software fallback" << std::endl;
    return true;
#endif
}

void GNAVoiceDetector::processing_loop() {
    while (processing_active_.load()) {
        std::unique_lock<std::mutex> lock(processing_mutex_);
        processing_cv_.wait(lock, [this] {
            return !processing_active_.load() || !audio_buffer_.empty();
        });

        if (!processing_active_.load()) {
            break;
        }

        // Extract features from audio
        AudioFeatures features = feature_extractor_->extract_features(
            audio_buffer_.data(), audio_buffer_.size()
        );

        // Perform voice activity detection
        bool voice_detected = detect_voice_activity(features);

        // Perform wake word detection if voice is active
        std::string detected_wake_word;
        bool wake_word_detected = false;
        if (voice_detected) {
            wake_word_detected = detect_wake_word(features, detected_wake_word);
        }

        // Create detection result
        GNADetectionResult result;
        result.voice_detected = voice_detected;
        result.wake_word_detected = wake_word_detected;
        result.wake_word = detected_wake_word;
        result.confidence = voice_detected ? 0.8f : 0.2f; // Simplified confidence
        result.timestamp_us = get_timestamp_us();
        result.power_consumption_mw = power_manager_->get_current_power_mw();
        result.temperature_celsius = power_manager_->get_temperature_celsius();

        // Update last result and notify callback
        {
            std::lock_guard<std::mutex> result_lock(result_mutex_);
            last_result_ = result;
        }

        notify_detection_result(result);
    }
}

bool GNAVoiceDetector::detect_voice_activity(const AudioFeatures& features) {
    // Simplified VAD based on energy and spectral features
    if (features.energy.empty()) {
        return false;
    }

    float energy = features.energy[0];
    float zcr = features.zcr;

    // Simple energy-based VAD with ZCR
    bool energy_above_threshold = energy > config_.energy_threshold;
    bool zcr_in_speech_range = zcr > 0.1f && zcr < 0.7f;

    return energy_above_threshold && zcr_in_speech_range;
}

bool GNAVoiceDetector::detect_wake_word(const AudioFeatures& features, std::string& detected_word) {
    // Simplified wake word detection using template matching
    if (wake_word_templates_.empty()) {
        return false;
    }

    float best_similarity = 0.0f;
    std::string best_match;

    for (const auto& [word, template_features] : wake_word_templates_) {
        float similarity = compute_wake_word_similarity(features.mfcc, template_features);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_match = word;
        }
    }

    if (best_similarity > config_.wake_word_threshold) {
        detected_word = best_match;
        return true;
    }

    return false;
}

float GNAVoiceDetector::compute_wake_word_similarity(
    const std::vector<float>& features,
    const std::vector<float>& template_features) {

    if (features.size() != template_features.size()) {
        return 0.0f;
    }

    // Compute cosine similarity
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < features.size(); ++i) {
        dot_product += features[i] * template_features[i];
        norm_a += features[i] * features[i];
        norm_b += template_features[i] * template_features[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 0.0f;
    }

    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

uint64_t GNAVoiceDetector::get_timestamp_us() const {
    return get_timestamp_microseconds();
}

void GNAVoiceDetector::notify_detection_result(const GNADetectionResult& result) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    if (detection_callback_) {
        detection_callback_(result);
    }
}

bool GNAVoiceDetector::add_wake_word(const std::string& word,
                                    const std::vector<float>& template_features) {
    wake_word_templates_[word] = template_features;
    return true;
}

bool GNAVoiceDetector::remove_wake_word(const std::string& word) {
    return wake_word_templates_.erase(word) > 0;
}

void GNAVoiceDetector::clear_wake_words() {
    wake_word_templates_.clear();
}

void GNAVoiceDetector::set_power_mode(const std::string& mode) {
    if (mode == "ultra_low") {
        power_manager_->optimize_for_idle();
    } else if (mode == "performance") {
        power_manager_->optimize_for_detection();
    }
    // "balanced" is default
}

} // namespace vtt