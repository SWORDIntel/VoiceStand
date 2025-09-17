#pragma once

#include "gna_device_manager.h"
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <chrono>

/**
 * Personal GNA-Accelerated Voice Detector for Dell Latitude 5450
 * Week 1 Implementation: Personal wake word detection and VAD
 *
 * Focus: Personal speech patterns, battery efficiency, 90%+ accuracy
 */

class GNAVoiceDetector {
public:
    struct PersonalVoiceConfig {
        // Personal wake word settings
        std::vector<std::string> personal_wake_words = {"computer", "assistant", "wake"};
        float wake_word_threshold = 0.85f;  // Personal accuracy target
        bool adaptive_threshold = true;

        // Personal VAD settings
        float vad_threshold = 0.65f;        // Personal voice activity threshold
        int min_speech_duration_ms = 300;   // Personal minimum speech length
        int max_silence_duration_ms = 1000; // Personal silence timeout

        // Personal audio processing
        int sample_rate = 16000;            // Standard for personal use
        int frame_size_ms = 30;             // Personal frame size
        int hop_length_ms = 10;             // Personal hop length

        // Personal GNA optimization
        bool use_gna_acceleration = true;   // Enable GNA for personal efficiency
        int gna_context_window = 512;       // Personal context size
        float power_efficiency_mode = 0.8f; // Personal power vs performance balance
    };

    struct PersonalDetectionResult {
        bool wake_word_detected = false;
        std::string detected_word;
        float confidence = 0.0f;
        std::chrono::steady_clock::time_point timestamp;

        // Personal VAD results
        bool voice_activity = false;
        float voice_probability = 0.0f;
        int speech_duration_ms = 0;

        // Personal performance metrics
        float processing_time_ms = 0.0f;
        float power_consumption_mw = 0.0f;
        bool gna_used = false;
    };

    struct PersonalAudioFeatures {
        std::vector<float> mfcc_features;     // Personal MFCC for wake words
        std::vector<float> energy_features;   // Personal energy analysis
        std::vector<float> spectral_features; // Personal spectral characteristics
        float zero_crossing_rate = 0.0f;      // Personal ZCR for VAD
        float spectral_centroid = 0.0f;       // Personal spectral analysis
    };

    // Constructor for personal voice detection
    GNAVoiceDetector(std::shared_ptr<GNADeviceManager> gna_manager);
    GNAVoiceDetector(std::shared_ptr<GNADeviceManager> gna_manager,
                     const PersonalVoiceConfig& config);
    ~GNAVoiceDetector();

    // Core personal voice detection
    bool initializePersonalDetection();
    PersonalDetectionResult detectPersonalVoice(const std::vector<float>& audio_data);
    bool trainPersonalWakeWords(const std::vector<std::string>& wake_words);
    void shutdown();

    // Personal VAD operations
    bool detectVoiceActivity(const std::vector<float>& audio_frame);
    float calculateVoiceProbability(const PersonalAudioFeatures& features);
    bool isPersonalSpeechActive() const;

    // Personal wake word detection
    bool detectWakeWord(const std::vector<float>& audio_data, std::string& detected_word, float& confidence);
    void updatePersonalWakeWordThreshold(float threshold);
    bool addPersonalWakeWord(const std::string& wake_word);

    // Personal feature extraction (GNA-accelerated)
    PersonalAudioFeatures extractPersonalFeatures(const std::vector<float>& audio_data);
    std::vector<float> computePersonalMFCC(const std::vector<float>& audio_data);
    std::vector<float> computePersonalSpectralFeatures(const std::vector<float>& audio_data);

    // Personal performance monitoring
    float getPersonalDetectionAccuracy() const;
    float getPersonalPowerEfficiency() const;
    void logPersonalPerformanceMetrics() const;

    // Personal configuration updates
    void updatePersonalVoiceProfile();
    void adjustPersonalSensitivity(float sensitivity);
    bool validatePersonalConfiguration() const;

private:
    std::shared_ptr<GNADeviceManager> gna_manager_;
    PersonalVoiceConfig config_;

    // Personal detection state
    std::atomic<bool> initialized_{false};
    std::atomic<bool> detecting_{false};
    std::atomic<bool> voice_active_{false};

    // Personal wake word templates (GNA-optimized)
    std::vector<std::vector<float>> personal_wake_word_templates_;
    std::vector<std::string> personal_wake_word_labels_;

    // Personal performance tracking
    mutable std::atomic<uint64_t> total_detections_{0};
    mutable std::atomic<uint64_t> correct_detections_{0};
    mutable std::atomic<float> average_power_consumption_{0.0f};
    mutable std::atomic<float> average_processing_time_{0.0f};

    // Personal audio processing buffers
    std::vector<float> personal_audio_buffer_;
    std::vector<float> personal_feature_buffer_;

    // Personal GNA acceleration methods
    bool processWithPersonalGNA(const std::vector<float>& features, std::vector<float>& output);
    bool loadPersonalModelToGNA();
    void optimizePersonalGNAPerformance();

    // Personal feature computation helpers
    std::vector<float> computePersonalFFT(const std::vector<float>& audio_data);
    std::vector<float> applyPersonalMelFilterBank(const std::vector<float>& spectrum);
    std::vector<float> applyPersonalDCT(const std::vector<float>& mel_spectrum);

    // Personal wake word matching
    float computePersonalSimilarity(const std::vector<float>& features,
                                   const std::vector<float>& template_features);
    bool matchPersonalWakeWordTemplate(const std::vector<float>& features,
                                      std::string& best_match, float& confidence);

    // Personal VAD helpers
    float computePersonalEnergyVAD(const std::vector<float>& audio_frame);
    float computePersonalSpectralVAD(const std::vector<float>& audio_frame);
    bool applyPersonalVADSmoothing(bool current_vad);

    // Personal optimization and monitoring
    void updatePersonalPerformanceMetrics(const PersonalDetectionResult& result);
    void adaptPersonalThresholds();
    std::string getPersonalConfigSummary() const;
};

// Personal utility functions for voice detection
namespace personal_voice_utils {
    // Personal audio preprocessing
    std::vector<float> normalizePersonalAudio(const std::vector<float>& audio);
    std::vector<float> applyPersonalPreemphasis(const std::vector<float>& audio, float alpha = 0.97f);
    std::vector<float> applyPersonalWindowing(const std::vector<float>& audio);

    // Personal wake word utilities
    std::vector<std::string> getDefaultPersonalWakeWords();
    bool validatePersonalWakeWord(const std::string& wake_word);
    float computePersonalWakeWordQuality(const std::vector<float>& features);

    // Personal performance utilities
    GNAVoiceDetector::PersonalVoiceConfig getOptimalPersonalVoiceConfig();
    bool validatePersonalPowerBudget(float target_power_mw);
    std::string formatPersonalDetectionResults(const GNAVoiceDetector::PersonalDetectionResult& result);
}