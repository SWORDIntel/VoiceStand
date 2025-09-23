#pragma once

#include <vector>
#include <string>
#include <map>
#include <json/json.h>
#include <chrono>

namespace vtt {

// Audio feature extraction and serialization utilities
class AudioFeatureSerializer {
public:
    struct AudioFeatures {
        // Basic time-domain features
        double energy = 0.0;
        double zero_crossing_rate = 0.0;
        double rms = 0.0;

        // Spectral features
        double spectral_centroid = 0.0;
        double spectral_rolloff = 0.0;
        double spectral_flux = 0.0;
        std::vector<double> mfcc_coefficients;

        // Voice activity features
        double voice_activity_probability = 0.0;
        double speech_rate = 0.0;
        double pause_ratio = 0.0;

        // Prosodic features
        double fundamental_frequency = 0.0;
        double pitch_variance = 0.0;
        double intensity = 0.0;

        // Noise characteristics
        double snr = 0.0;
        double noise_floor = 0.0;

        // Metadata
        std::chrono::steady_clock::time_point timestamp;
        double duration_seconds = 0.0;
        uint32_t sample_rate = 16000;
        size_t num_samples = 0;
    };

    struct RecognitionFeatures {
        // Recognition context
        std::string text;
        double confidence = 0.0;
        std::string language = "en";
        std::string domain = "general";

        // Model information
        std::string model_name;
        std::vector<std::string> model_ensemble;
        std::map<std::string, double> model_confidences;

        // Performance metrics
        std::chrono::milliseconds processing_time{0};
        bool is_final_result = false;
        double uncertainty_score = 0.0;

        // Speaker characteristics
        std::string speaker_id;
        double speaker_confidence = 0.0;
        std::string accent_type;

        // Correction information
        std::string ground_truth;
        std::string correction_type;
        bool user_corrected = false;
    };

    struct ContextFeatures {
        // Environmental context
        double background_noise_level = 0.0;
        std::string audio_source_type; // "microphone", "phone", "recording"
        bool is_streaming = false;

        // Content context
        std::vector<std::string> previous_utterances;
        std::string conversation_topic;
        std::map<std::string, double> topic_probabilities;

        // User context
        std::string user_id;
        std::string session_id;
        std::chrono::steady_clock::time_point session_start;

        // Technical context
        std::string device_type;
        std::string os_version;
        std::string app_version;
    };

public:
    AudioFeatureSerializer();
    ~AudioFeatureSerializer();

    // Audio feature extraction
    AudioFeatures extract_audio_features(const float* samples, size_t num_samples,
                                       uint32_t sample_rate = 16000);

    // MFCC feature extraction
    std::vector<double> extract_mfcc(const float* samples, size_t num_samples,
                                   uint32_t sample_rate = 16000,
                                   int num_coefficients = 13);

    // Spectral feature extraction
    void extract_spectral_features(const float* samples, size_t num_samples,
                                 uint32_t sample_rate, AudioFeatures& features);

    // Voice activity detection features
    double calculate_voice_activity_probability(const float* samples, size_t num_samples);

    // Prosodic feature extraction
    void extract_prosodic_features(const float* samples, size_t num_samples,
                                 uint32_t sample_rate, AudioFeatures& features);

    // JSON serialization
    Json::Value serialize_audio_features(const AudioFeatures& features);
    Json::Value serialize_recognition_features(const RecognitionFeatures& features);
    Json::Value serialize_context_features(const ContextFeatures& features);

    // Combined serialization for API submission
    Json::Value serialize_complete_submission(const AudioFeatures& audio_features,
                                            const RecognitionFeatures& recognition_features,
                                            const ContextFeatures& context_features);

    // Deserialization
    AudioFeatures deserialize_audio_features(const Json::Value& json);
    RecognitionFeatures deserialize_recognition_features(const Json::Value& json);
    ContextFeatures deserialize_context_features(const Json::Value& json);

    // Feature vector conversion (for ML models)
    std::vector<double> audio_features_to_vector(const AudioFeatures& features);
    std::vector<double> recognition_features_to_vector(const RecognitionFeatures& features);
    std::vector<double> context_features_to_vector(const ContextFeatures& features);

    // Combined feature vector
    std::vector<double> create_combined_feature_vector(const AudioFeatures& audio_features,
                                                     const RecognitionFeatures& recognition_features,
                                                     const ContextFeatures& context_features);

    // Utility functions
    static std::string timestamp_to_iso8601(const std::chrono::steady_clock::time_point& timestamp);
    static std::chrono::steady_clock::time_point iso8601_to_timestamp(const std::string& iso_string);

    // Feature normalization
    void normalize_features(std::vector<double>& features);
    void standardize_features(std::vector<double>& features,
                            const std::vector<double>& means,
                            const std::vector<double>& std_devs);

private:
    // FFT for spectral analysis
    void compute_fft(const std::vector<double>& input, std::vector<std::complex<double>>& output);
    void apply_hamming_window(std::vector<double>& samples);

    // Mel-scale conversion
    std::vector<double> hz_to_mel_scale(const std::vector<double>& frequencies);
    std::vector<double> mel_to_hz_scale(const std::vector<double>& mel_frequencies);

    // Statistical calculations
    double calculate_mean(const std::vector<double>& values);
    double calculate_variance(const std::vector<double>& values, double mean);
    double calculate_percentile(const std::vector<double>& values, double percentile);

    // Internal state
    struct InternalState;
    std::unique_ptr<InternalState> internal_state_;
};

// Specialized feature extractors
class MFCCExtractor {
public:
    MFCCExtractor(int num_coefficients = 13, int num_filters = 26,
                  double min_freq = 0.0, double max_freq = 8000.0);

    std::vector<double> extract(const float* samples, size_t num_samples, uint32_t sample_rate);

private:
    int num_coefficients_;
    int num_filters_;
    double min_freq_;
    double max_freq_;
    std::vector<std::vector<double>> mel_filter_bank_;

    void initialize_mel_filter_bank(uint32_t sample_rate, size_t fft_size);
    std::vector<double> apply_dct(const std::vector<double>& mel_energies);
};

class SpectralFeatureExtractor {
public:
    struct SpectralFeatures {
        double centroid = 0.0;
        double rolloff = 0.0;
        double flux = 0.0;
        double flatness = 0.0;
        double bandwidth = 0.0;
        std::vector<double> chroma;
    };

    SpectralFeatures extract(const float* samples, size_t num_samples, uint32_t sample_rate);

private:
    double calculate_spectral_centroid(const std::vector<double>& magnitude_spectrum,
                                     const std::vector<double>& frequencies);
    double calculate_spectral_rolloff(const std::vector<double>& magnitude_spectrum,
                                    const std::vector<double>& frequencies,
                                    double rolloff_threshold = 0.85);
};

class PitchExtractor {
public:
    struct PitchFeatures {
        double fundamental_frequency = 0.0;
        double pitch_confidence = 0.0;
        std::vector<double> pitch_contour;
        double pitch_variance = 0.0;
        double voicing_probability = 0.0;
    };

    PitchFeatures extract(const float* samples, size_t num_samples, uint32_t sample_rate);

private:
    double estimate_f0_autocorrelation(const float* samples, size_t num_samples,
                                     uint32_t sample_rate);
    double calculate_pitch_confidence(const std::vector<double>& autocorrelation);
};

// Feature aggregation for different time scales
class FeatureAggregator {
public:
    enum class AggregationType {
        MEAN,
        VARIANCE,
        MIN,
        MAX,
        PERCENTILE_25,
        PERCENTILE_50,
        PERCENTILE_75,
        DELTA,
        DELTA_DELTA
    };

    // Aggregate features over time windows
    std::vector<double> aggregate_features(const std::vector<std::vector<double>>& feature_frames,
                                         AggregationType type);

    // Create statistical summary of features
    std::map<std::string, double> create_feature_summary(const std::vector<double>& features);

    // Temporal derivative features
    std::vector<double> calculate_delta_features(const std::vector<std::vector<double>>& feature_frames);
    std::vector<double> calculate_delta_delta_features(const std::vector<std::vector<double>>& feature_frames);
};

} // namespace vtt