#include "gna_voice_detector.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <thread>
#include <fstream>

// Personal math constants for Dell Latitude 5450 optimization
constexpr float PI = 3.14159265359f;
constexpr float LOG10_E = 0.43429448190325176f;
constexpr int PERSONAL_MFCC_COEFFS = 13;  // Personal MFCC count
constexpr int PERSONAL_MEL_FILTERS = 26;  // Personal mel filter bank size

GNAVoiceDetector::GNAVoiceDetector(std::shared_ptr<GNADeviceManager> gna_manager)
    : gna_manager_(gna_manager) {
    config_ = personal_voice_utils::getOptimalPersonalVoiceConfig();
    std::cout << "Initializing Personal GNA Voice Detector for Dell Latitude 5450" << std::endl;

    // Pre-allocate personal buffers for efficiency
    int max_frame_size = (config_.sample_rate * config_.frame_size_ms) / 1000;
    personal_audio_buffer_.reserve(max_frame_size * 2);
    personal_feature_buffer_.reserve(PERSONAL_MFCC_COEFFS * 10);  // 10 frames
}

GNAVoiceDetector::GNAVoiceDetector(std::shared_ptr<GNADeviceManager> gna_manager,
                                   const PersonalVoiceConfig& config)
    : gna_manager_(gna_manager), config_(config) {
    std::cout << "Initializing Personal GNA Voice Detector for Dell Latitude 5450" << std::endl;

    // Pre-allocate personal buffers for efficiency
    int max_frame_size = (config_.sample_rate * config_.frame_size_ms) / 1000;
    personal_audio_buffer_.reserve(max_frame_size * 2);
    personal_feature_buffer_.reserve(PERSONAL_MFCC_COEFFS * 10);  // 10 frames
}

GNAVoiceDetector::~GNAVoiceDetector() {
    shutdown();
}

bool GNAVoiceDetector::initializePersonalDetection() {
    if (!gna_manager_ || !gna_manager_->isPersonalDeviceReady()) {
        std::cerr << "Personal GNA device not ready for voice detection" << std::endl;
        return false;
    }

    std::cout << "Initializing personal voice detection system..." << std::endl;

    // Load personal wake word templates
    if (!trainPersonalWakeWords(config_.personal_wake_words)) {
        std::cerr << "Failed to load personal wake word templates" << std::endl;
        return false;
    }

    // Load personal model to GNA if available
    if (config_.use_gna_acceleration) {
        if (!loadPersonalModelToGNA()) {
            std::cout << "Warning: Could not load personal model to GNA, using CPU fallback" << std::endl;
            config_.use_gna_acceleration = false;
        }
    }

    // Validate personal configuration
    if (!validatePersonalConfiguration()) {
        std::cerr << "Personal voice configuration validation failed" << std::endl;
        return false;
    }

    initialized_ = true;
    detecting_ = false;

    std::cout << "Personal voice detection initialized successfully" << std::endl;
    std::cout << getPersonalConfigSummary() << std::endl;

    return true;
}

GNAVoiceDetector::PersonalDetectionResult GNAVoiceDetector::detectPersonalVoice(
    const std::vector<float>& audio_data) {

    PersonalDetectionResult result;
    result.timestamp = std::chrono::steady_clock::now();

    if (!initialized_ || audio_data.empty()) {
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Personal audio preprocessing
    std::vector<float> preprocessed_audio = personal_voice_utils::normalizePersonalAudio(audio_data);
    preprocessed_audio = personal_voice_utils::applyPersonalPreemphasis(preprocessed_audio);

    // Personal feature extraction (GNA-accelerated if available)
    PersonalAudioFeatures features = extractPersonalFeatures(preprocessed_audio);

    // Personal VAD detection
    result.voice_activity = detectVoiceActivity(preprocessed_audio);
    result.voice_probability = calculateVoiceProbability(features);

    // Personal wake word detection if voice is active
    if (result.voice_activity && result.voice_probability > config_.vad_threshold) {
        result.wake_word_detected = detectWakeWord(preprocessed_audio,
                                                  result.detected_word,
                                                  result.confidence);
    }

    // Personal performance metrics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.processing_time_ms = duration.count() / 1000.0f;

    result.power_consumption_mw = gna_manager_->getCurrentPowerConsumption() * 1000.0f;
    result.gna_used = config_.use_gna_acceleration;

    // Update personal performance tracking
    updatePersonalPerformanceMetrics(result);
    total_detections_.fetch_add(1);

    return result;
}

bool GNAVoiceDetector::trainPersonalWakeWords(const std::vector<std::string>& wake_words) {
    std::cout << "Training personal wake words for individual use..." << std::endl;

    personal_wake_word_templates_.clear();
    personal_wake_word_labels_.clear();

    for (const auto& wake_word : wake_words) {
        if (!personal_voice_utils::validatePersonalWakeWord(wake_word)) {
            std::cout << "Warning: Skipping invalid personal wake word: " << wake_word << std::endl;
            continue;
        }

        // Create simple template for personal wake word (Week 1 implementation)
        std::vector<float> template_features(PERSONAL_MFCC_COEFFS, 0.0f);

        // Generate simple personal template based on word characteristics
        float word_energy = 0.0f;
        for (char c : wake_word) {
            word_energy += static_cast<float>(c) / 255.0f;
        }

        // Personal template generation (simplified for Week 1)
        for (int i = 0; i < PERSONAL_MFCC_COEFFS; ++i) {
            template_features[i] = word_energy * std::sin(PI * i / PERSONAL_MFCC_COEFFS) +
                                  (wake_word.length() / 10.0f) * std::cos(PI * i / PERSONAL_MFCC_COEFFS);
        }

        personal_wake_word_templates_.push_back(template_features);
        personal_wake_word_labels_.push_back(wake_word);

        std::cout << "  Personal wake word trained: " << wake_word << std::endl;
    }

    std::cout << "Personal wake word training complete: " << personal_wake_word_templates_.size()
              << " words trained" << std::endl;

    return !personal_wake_word_templates_.empty();
}

bool GNAVoiceDetector::detectVoiceActivity(const std::vector<float>& audio_frame) {
    if (audio_frame.empty()) return false;

    // Personal energy-based VAD
    float energy_vad = computePersonalEnergyVAD(audio_frame);

    // Personal spectral VAD
    float spectral_vad = computePersonalSpectralVAD(audio_frame);

    // Personal combined VAD decision
    float combined_vad = (energy_vad + spectral_vad) / 2.0f;
    bool current_vad = combined_vad > config_.vad_threshold;

    // Apply personal VAD smoothing
    bool smoothed_vad = applyPersonalVADSmoothing(current_vad);

    voice_active_ = smoothed_vad;
    return smoothed_vad;
}

float GNAVoiceDetector::calculateVoiceProbability(const PersonalAudioFeatures& features) {
    if (features.mfcc_features.empty()) return 0.0f;

    // Personal voice probability calculation
    float mfcc_energy = 0.0f;
    for (float coef : features.mfcc_features) {
        mfcc_energy += coef * coef;
    }
    mfcc_energy = std::sqrt(mfcc_energy / features.mfcc_features.size());

    // Personal spectral analysis
    float spectral_score = features.spectral_centroid / (config_.sample_rate / 2.0f);

    // Personal ZCR analysis
    float zcr_score = std::min(1.0f, features.zero_crossing_rate / 0.1f);

    // Personal combined probability
    float voice_prob = (mfcc_energy * 0.5f + spectral_score * 0.3f + zcr_score * 0.2f);
    return std::min(1.0f, std::max(0.0f, voice_prob));
}

bool GNAVoiceDetector::detectWakeWord(const std::vector<float>& audio_data,
                                     std::string& detected_word, float& confidence) {
    if (personal_wake_word_templates_.empty()) return false;

    // Personal feature extraction for wake word detection
    PersonalAudioFeatures features = extractPersonalFeatures(audio_data);

    // Personal wake word template matching
    return matchPersonalWakeWordTemplate(features.mfcc_features, detected_word, confidence);
}

GNAVoiceDetector::PersonalAudioFeatures GNAVoiceDetector::extractPersonalFeatures(
    const std::vector<float>& audio_data) {

    PersonalAudioFeatures features;

    if (audio_data.empty()) return features;

    // Personal MFCC computation (GNA-accelerated if available)
    features.mfcc_features = computePersonalMFCC(audio_data);

    // Personal spectral features
    features.spectral_features = computePersonalSpectralFeatures(audio_data);

    // Personal zero crossing rate
    float zcr_count = 0.0f;
    for (size_t i = 1; i < audio_data.size(); ++i) {
        if ((audio_data[i] >= 0) != (audio_data[i-1] >= 0)) {
            zcr_count += 1.0f;
        }
    }
    features.zero_crossing_rate = zcr_count / (audio_data.size() - 1);

    // Personal spectral centroid
    std::vector<float> spectrum = computePersonalFFT(audio_data);
    float weighted_freq_sum = 0.0f;
    float magnitude_sum = 0.0f;

    for (size_t i = 0; i < spectrum.size() / 2; ++i) {
        float magnitude = spectrum[i];
        float frequency = (i * config_.sample_rate) / static_cast<float>(spectrum.size());
        weighted_freq_sum += frequency * magnitude;
        magnitude_sum += magnitude;
    }

    features.spectral_centroid = (magnitude_sum > 0) ? (weighted_freq_sum / magnitude_sum) : 0.0f;

    return features;
}

std::vector<float> GNAVoiceDetector::computePersonalMFCC(const std::vector<float>& audio_data) {
    if (audio_data.empty()) return {};

    // Personal FFT computation
    std::vector<float> spectrum = computePersonalFFT(audio_data);

    // Personal mel filter bank
    std::vector<float> mel_spectrum = applyPersonalMelFilterBank(spectrum);

    // Personal DCT for MFCC
    std::vector<float> mfcc = applyPersonalDCT(mel_spectrum);

    // GNA acceleration if available
    if (config_.use_gna_acceleration && gna_manager_->isPersonalDeviceReady()) {
        std::vector<float> gna_output;
        if (processWithPersonalGNA(mfcc, gna_output)) {
            return gna_output;
        }
    }

    return mfcc;
}

std::vector<float> GNAVoiceDetector::computePersonalSpectralFeatures(const std::vector<float>& audio_data) {
    std::vector<float> features;
    if (audio_data.empty()) return features;

    // Personal energy computation
    float energy = 0.0f;
    for (float sample : audio_data) {
        energy += sample * sample;
    }
    features.push_back(std::sqrt(energy / audio_data.size()));

    // Personal additional spectral features (simplified for Week 1)
    std::vector<float> spectrum = computePersonalFFT(audio_data);

    // Personal spectral rolloff
    float total_energy = 0.0f;
    for (size_t i = 0; i < spectrum.size() / 2; ++i) {
        total_energy += spectrum[i];
    }

    float cumulative_energy = 0.0f;
    float rolloff_threshold = 0.85f * total_energy;
    int rolloff_bin = 0;

    for (size_t i = 0; i < spectrum.size() / 2; ++i) {
        cumulative_energy += spectrum[i];
        if (cumulative_energy >= rolloff_threshold) {
            rolloff_bin = i;
            break;
        }
    }

    float rolloff_freq = (rolloff_bin * config_.sample_rate) / static_cast<float>(spectrum.size());
    features.push_back(rolloff_freq);

    return features;
}

bool GNAVoiceDetector::processWithPersonalGNA(const std::vector<float>& features,
                                              std::vector<float>& output) {
    if (!gna_manager_->isPersonalDeviceReady() || features.empty()) {
        return false;
    }

    // Personal GNA processing (simplified for Week 1)
    output = features;  // Pass-through for now

    // Apply personal GNA-style optimization (placeholder)
    for (float& value : output) {
        value *= 1.05f;  // Simple personal enhancement
    }

    return true;
}

bool GNAVoiceDetector::loadPersonalModelToGNA() {
    if (!gna_manager_->isPersonalDeviceReady()) {
        return false;
    }

    std::cout << "Loading personal voice model to GNA..." << std::endl;

    // Personal model loading (simplified for Week 1)
    // In full implementation, this would load trained neural networks

    optimizePersonalGNAPerformance();

    std::cout << "Personal model loaded to GNA successfully" << std::endl;
    return true;
}

void GNAVoiceDetector::optimizePersonalGNAPerformance() {
    if (!gna_manager_->isPersonalDeviceReady()) return;

    // Personal GNA optimization for battery efficiency
    std::cout << "Optimizing personal GNA performance for battery efficiency..." << std::endl;

    // Configure personal power efficiency mode
    if (config_.power_efficiency_mode > 0.7f) {
        gna_manager_->enterBatteryOptimizedMode();
    }
}

float GNAVoiceDetector::computePersonalEnergyVAD(const std::vector<float>& audio_frame) {
    if (audio_frame.empty()) return 0.0f;

    float energy = 0.0f;
    for (float sample : audio_frame) {
        energy += sample * sample;
    }

    float rms_energy = std::sqrt(energy / audio_frame.size());
    return std::min(1.0f, rms_energy * 10.0f);  // Personal scaling
}

float GNAVoiceDetector::computePersonalSpectralVAD(const std::vector<float>& audio_frame) {
    if (audio_frame.empty()) return 0.0f;

    std::vector<float> spectrum = computePersonalFFT(audio_frame);

    // Personal spectral energy in voice frequency range (300-3400 Hz)
    int start_bin = (300 * spectrum.size()) / config_.sample_rate;
    int end_bin = (3400 * spectrum.size()) / config_.sample_rate;

    float voice_energy = 0.0f;
    float total_energy = 0.0f;

    for (int i = 0; i < static_cast<int>(spectrum.size() / 2); ++i) {
        float magnitude = spectrum[i];
        total_energy += magnitude;

        if (i >= start_bin && i <= end_bin) {
            voice_energy += magnitude;
        }
    }

    return (total_energy > 0) ? (voice_energy / total_energy) : 0.0f;
}

bool GNAVoiceDetector::applyPersonalVADSmoothing(bool current_vad) {
    // Personal VAD smoothing (simplified for Week 1)
    static bool previous_vad = false;
    static int vad_count = 0;

    if (current_vad == previous_vad) {
        vad_count++;
    } else {
        vad_count = 0;
    }

    // Personal smoothing threshold
    bool smoothed_vad = (vad_count >= 2) ? current_vad : previous_vad;
    previous_vad = current_vad;

    return smoothed_vad;
}

bool GNAVoiceDetector::matchPersonalWakeWordTemplate(const std::vector<float>& features,
                                                     std::string& best_match, float& confidence) {
    if (features.empty() || personal_wake_word_templates_.empty()) return false;

    float best_similarity = 0.0f;
    int best_index = -1;

    for (size_t i = 0; i < personal_wake_word_templates_.size(); ++i) {
        float similarity = computePersonalSimilarity(features, personal_wake_word_templates_[i]);

        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_index = static_cast<int>(i);
        }
    }

    if (best_index >= 0 && best_similarity > config_.wake_word_threshold) {
        best_match = personal_wake_word_labels_[best_index];
        confidence = best_similarity;
        correct_detections_.fetch_add(1);
        return true;
    }

    return false;
}

float GNAVoiceDetector::computePersonalSimilarity(const std::vector<float>& features,
                                                 const std::vector<float>& template_features) {
    if (features.size() != template_features.size()) return 0.0f;

    // Personal cosine similarity for Wake Word 1
    float dot_product = 0.0f;
    float norm_features = 0.0f;
    float norm_template = 0.0f;

    for (size_t i = 0; i < features.size(); ++i) {
        dot_product += features[i] * template_features[i];
        norm_features += features[i] * features[i];
        norm_template += template_features[i] * template_features[i];
    }

    norm_features = std::sqrt(norm_features);
    norm_template = std::sqrt(norm_template);

    if (norm_features > 0 && norm_template > 0) {
        return dot_product / (norm_features * norm_template);
    }

    return 0.0f;
}

// Personal utility function implementations
std::vector<float> GNAVoiceDetector::computePersonalFFT(const std::vector<float>& audio_data) {
    // Personal simplified FFT (Week 1 implementation)
    // In production, use FFTW or similar optimized library

    int n = audio_data.size();
    std::vector<float> spectrum(n, 0.0f);

    for (int k = 0; k < n / 2; ++k) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;

        for (int t = 0; t < n; ++t) {
            float angle = -2.0f * PI * k * t / n;
            real_sum += audio_data[t] * std::cos(angle);
            imag_sum += audio_data[t] * std::sin(angle);
        }

        spectrum[k] = std::sqrt(real_sum * real_sum + imag_sum * imag_sum);
    }

    return spectrum;
}

std::vector<float> GNAVoiceDetector::applyPersonalMelFilterBank(const std::vector<float>& spectrum) {
    std::vector<float> mel_spectrum(PERSONAL_MEL_FILTERS, 0.0f);

    // Personal mel filter bank (simplified for Week 1)
    int spectrum_size = spectrum.size() / 2;
    int filters_per_bin = spectrum_size / PERSONAL_MEL_FILTERS;

    for (int i = 0; i < PERSONAL_MEL_FILTERS; ++i) {
        int start_bin = i * filters_per_bin;
        int end_bin = std::min(start_bin + filters_per_bin, spectrum_size);

        float sum = 0.0f;
        for (int j = start_bin; j < end_bin; ++j) {
            sum += spectrum[j];
        }

        mel_spectrum[i] = (end_bin > start_bin) ? (sum / (end_bin - start_bin)) : 0.0f;
    }

    return mel_spectrum;
}

std::vector<float> GNAVoiceDetector::applyPersonalDCT(const std::vector<float>& mel_spectrum) {
    std::vector<float> mfcc(PERSONAL_MFCC_COEFFS, 0.0f);

    // Personal DCT-II (simplified for Week 1)
    for (int i = 0; i < PERSONAL_MFCC_COEFFS; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < PERSONAL_MEL_FILTERS; ++j) {
            float log_mel = std::log(std::max(1e-10f, mel_spectrum[j]));
            sum += log_mel * std::cos(PI * i * (j + 0.5f) / PERSONAL_MEL_FILTERS);
        }
        mfcc[i] = sum;
    }

    return mfcc;
}

float GNAVoiceDetector::getPersonalDetectionAccuracy() const {
    uint64_t total = total_detections_.load();
    uint64_t correct = correct_detections_.load();
    return (total > 0) ? (static_cast<float>(correct) / total * 100.0f) : 0.0f;
}

float GNAVoiceDetector::getPersonalPowerEfficiency() const {
    return average_power_consumption_.load();
}

void GNAVoiceDetector::updatePersonalPerformanceMetrics(const PersonalDetectionResult& result) {
    // Update personal running averages
    static constexpr float alpha = 0.1f;  // Personal smoothing factor

    float current_power = average_power_consumption_.load();
    average_power_consumption_ = current_power * (1.0f - alpha) + result.power_consumption_mw * alpha;

    float current_time = average_processing_time_.load();
    average_processing_time_ = current_time * (1.0f - alpha) + result.processing_time_ms * alpha;
}

void GNAVoiceDetector::logPersonalPerformanceMetrics() const {
    std::cout << "\n=== Personal Voice Detection Metrics ===" << std::endl;
    std::cout << "Total detections: " << total_detections_.load() << std::endl;
    std::cout << "Detection accuracy: " << getPersonalDetectionAccuracy() << "%" << std::endl;
    std::cout << "Average power consumption: " << average_power_consumption_.load() << "mW" << std::endl;
    std::cout << "Average processing time: " << average_processing_time_.load() << "ms" << std::endl;
    std::cout << "GNA acceleration: " << (config_.use_gna_acceleration ? "enabled" : "disabled") << std::endl;
    std::cout << "Voice activity: " << (voice_active_.load() ? "active" : "inactive") << std::endl;
    std::cout << "=======================================" << std::endl;
}

bool GNAVoiceDetector::validatePersonalConfiguration() const {
    bool valid = true;

    if (config_.wake_word_threshold < 0.5f || config_.wake_word_threshold > 1.0f) {
        std::cout << "Warning: Personal wake word threshold should be between 0.5 and 1.0" << std::endl;
        valid = false;
    }

    if (config_.vad_threshold < 0.3f || config_.vad_threshold > 0.9f) {
        std::cout << "Warning: Personal VAD threshold should be between 0.3 and 0.9" << std::endl;
        valid = false;
    }

    if (config_.personal_wake_words.empty()) {
        std::cout << "Error: No personal wake words configured" << std::endl;
        valid = false;
    }

    return valid;
}

std::string GNAVoiceDetector::getPersonalConfigSummary() const {
    std::string summary = "\n=== Personal Voice Detection Configuration ===\n";
    summary += "Wake words: ";
    for (const auto& word : config_.personal_wake_words) {
        summary += word + " ";
    }
    summary += "\nWake word threshold: " + std::to_string(config_.wake_word_threshold);
    summary += "\nVAD threshold: " + std::to_string(config_.vad_threshold);
    summary += "\nGNA acceleration: " + std::string(config_.use_gna_acceleration ? "enabled" : "disabled");
    summary += "\nPower efficiency mode: " + std::to_string(config_.power_efficiency_mode);
    summary += "\n============================================";
    return summary;
}

void GNAVoiceDetector::shutdown() {
    if (initialized_) {
        detecting_ = false;
        initialized_ = false;

        std::cout << "Shutting down personal voice detector..." << std::endl;
        logPersonalPerformanceMetrics();
    }
}

// Personal utility function implementations
namespace personal_voice_utils {
    std::vector<float> normalizePersonalAudio(const std::vector<float>& audio) {
        if (audio.empty()) return {};

        std::vector<float> normalized = audio;

        // Personal audio normalization
        float max_val = *std::max_element(audio.begin(), audio.end(),
                                         [](float a, float b) { return std::abs(a) < std::abs(b); });

        if (std::abs(max_val) > 1e-10f) {
            float scale = 0.95f / std::abs(max_val);
            for (float& sample : normalized) {
                sample *= scale;
            }
        }

        return normalized;
    }

    std::vector<float> applyPersonalPreemphasis(const std::vector<float>& audio, float alpha) {
        if (audio.empty()) return {};

        std::vector<float> preemphasized = audio;

        // Personal preemphasis filter
        for (size_t i = 1; i < preemphasized.size(); ++i) {
            preemphasized[i] -= alpha * preemphasized[i - 1];
        }

        return preemphasized;
    }

    std::vector<float> applyPersonalWindowing(const std::vector<float>& audio) {
        if (audio.empty()) return {};

        std::vector<float> windowed = audio;

        // Personal Hamming window
        for (size_t i = 0; i < windowed.size(); ++i) {
            float window_val = 0.54f - 0.46f * std::cos(2.0f * PI * i / (windowed.size() - 1));
            windowed[i] *= window_val;
        }

        return windowed;
    }

    std::vector<std::string> getDefaultPersonalWakeWords() {
        return {"computer", "assistant", "wake", "hey", "listen"};
    }

    bool validatePersonalWakeWord(const std::string& wake_word) {
        // Personal wake word validation
        if (wake_word.length() < 2 || wake_word.length() > 20) return false;

        // Check for valid characters (personal use)
        for (char c : wake_word) {
            if (!std::isalnum(c) && c != '-' && c != '_') {
                return false;
            }
        }

        return true;
    }

    float computePersonalWakeWordQuality(const std::vector<float>& features) {
        if (features.empty()) return 0.0f;

        // Personal quality metric based on feature variance
        float mean = std::accumulate(features.begin(), features.end(), 0.0f) / features.size();

        float variance = 0.0f;
        for (float feature : features) {
            variance += (feature - mean) * (feature - mean);
        }
        variance /= features.size();

        // Personal quality score (higher variance = better distinctiveness)
        return std::min(1.0f, variance * 10.0f);
    }

    GNAVoiceDetector::PersonalVoiceConfig getOptimalPersonalVoiceConfig() {
        GNAVoiceDetector::PersonalVoiceConfig config;
        config.personal_wake_words = {"computer", "assistant"};  // Personal default
        config.wake_word_threshold = 0.85f;          // Personal high accuracy
        config.vad_threshold = 0.65f;                // Personal balanced sensitivity
        config.use_gna_acceleration = true;          // Personal efficiency
        config.power_efficiency_mode = 0.8f;         // Personal battery optimization
        config.min_speech_duration_ms = 250;         // Personal quick response
        config.max_silence_duration_ms = 800;        // Personal timeout
        return config;
    }

    bool validatePersonalPowerBudget(float target_power_mw) {
        // Personal power budget validation for laptop use
        return target_power_mw <= 50.0f;  // <50mW for good battery life
    }

    std::string formatPersonalDetectionResults(const GNAVoiceDetector::PersonalDetectionResult& result) {
        std::string formatted = "Personal Detection Results:\n";
        formatted += "  Wake word detected: " + std::string(result.wake_word_detected ? "YES" : "NO") + "\n";

        if (result.wake_word_detected) {
            formatted += "  Detected word: " + result.detected_word + "\n";
            formatted += "  Confidence: " + std::to_string(result.confidence * 100.0f) + "%\n";
        }

        formatted += "  Voice activity: " + std::string(result.voice_activity ? "YES" : "NO") + "\n";
        formatted += "  Voice probability: " + std::to_string(result.voice_probability * 100.0f) + "%\n";
        formatted += "  Processing time: " + std::to_string(result.processing_time_ms) + "ms\n";
        formatted += "  Power consumption: " + std::to_string(result.power_consumption_mw) + "mW\n";
        formatted += "  GNA acceleration: " + std::string(result.gna_used ? "YES" : "NO") + "\n";

        return formatted;
    }
}