#include "audio_feature_serializer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <complex>
#ifdef HAVE_FFTW3
#include <fftw3.h>
#endif
#include <sstream>
#include <iomanip>

namespace vtt {

// Internal state for FFTW plans and buffers
struct AudioFeatureSerializer::InternalState {
#ifdef HAVE_FFTW3
    fftw_plan fft_plan = nullptr;
    size_t fft_size = 0;
    double* fft_input = nullptr;
    fftw_complex* fft_output = nullptr;
#endif

    ~InternalState() {
#ifdef HAVE_FFTW3
        if (fft_plan) {
            fftw_destroy_plan(fft_plan);
        }
        if (fft_input) {
            fftw_free(fft_input);
        }
        if (fft_output) {
            fftw_free(fft_output);
        }
#endif
    }
};

AudioFeatureSerializer::AudioFeatureSerializer()
    : internal_state_(std::make_unique<InternalState>()) {
}

AudioFeatureSerializer::~AudioFeatureSerializer() = default;

AudioFeatureSerializer::AudioFeatures AudioFeatureSerializer::extract_audio_features(
    const float* samples, size_t num_samples, uint32_t sample_rate) {

    AudioFeatures features;

    if (!samples || num_samples == 0) {
        return features;
    }

    features.timestamp = std::chrono::steady_clock::now();
    features.sample_rate = sample_rate;
    features.num_samples = num_samples;
    features.duration_seconds = static_cast<double>(num_samples) / sample_rate;

    // Basic time-domain features
    double sum = 0.0;
    double sum_squares = 0.0;
    int zero_crossings = 0;

    for (size_t i = 0; i < num_samples; ++i) {
        sum += samples[i];
        sum_squares += samples[i] * samples[i];

        // Count zero crossings
        if (i > 0 && ((samples[i-1] >= 0.0f) != (samples[i] >= 0.0f))) {
            zero_crossings++;
        }
    }

    features.energy = sum_squares / num_samples;
    features.rms = std::sqrt(features.energy);
    features.zero_crossing_rate = static_cast<double>(zero_crossings) / num_samples;

    // Extract spectral features
    extract_spectral_features(samples, num_samples, sample_rate, features);

    // Extract MFCC coefficients
    features.mfcc_coefficients = extract_mfcc(samples, num_samples, sample_rate, 13);

    // Voice activity detection
    features.voice_activity_probability = calculate_voice_activity_probability(samples, num_samples);

    // Extract prosodic features
    extract_prosodic_features(samples, num_samples, sample_rate, features);

    // Estimate SNR (simplified)
    std::vector<double> sample_vec(samples, samples + num_samples);
    std::sort(sample_vec.begin(), sample_vec.end());

    double noise_floor = calculate_percentile(sample_vec, 10.0); // 10th percentile as noise estimate
    double signal_power = features.energy;
    double noise_power = noise_floor * noise_floor;

    features.noise_floor = noise_floor;
    features.snr = (noise_power > 0.0) ? 10.0 * std::log10(signal_power / noise_power) : 60.0;

    return features;
}

std::vector<double> AudioFeatureSerializer::extract_mfcc(const float* samples, size_t num_samples,
                                                        uint32_t sample_rate, int num_coefficients) {
    MFCCExtractor extractor(num_coefficients);
    return extractor.extract(samples, num_samples, sample_rate);
}

void AudioFeatureSerializer::extract_spectral_features(const float* samples, size_t num_samples,
                                                      uint32_t sample_rate, AudioFeatures& features) {
    SpectralFeatureExtractor extractor;
    auto spectral_features = extractor.extract(samples, num_samples, sample_rate);

    features.spectral_centroid = spectral_features.centroid;
    features.spectral_rolloff = spectral_features.rolloff;
    features.spectral_flux = spectral_features.flux;
}

double AudioFeatureSerializer::calculate_voice_activity_probability(const float* samples, size_t num_samples) {
    if (!samples || num_samples == 0) {
        return 0.0;
    }

    // Simple energy-based VAD
    double energy = 0.0;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += samples[i] * samples[i];
    }
    energy /= num_samples;

    // Threshold-based classification (can be improved)
    double energy_threshold = 1e-4;
    double probability = std::min(1.0, energy / energy_threshold);

    return probability;
}

void AudioFeatureSerializer::extract_prosodic_features(const float* samples, size_t num_samples,
                                                     uint32_t sample_rate, AudioFeatures& features) {
    PitchExtractor pitch_extractor;
    auto pitch_features = pitch_extractor.extract(samples, num_samples, sample_rate);

    features.fundamental_frequency = pitch_features.fundamental_frequency;
    features.pitch_variance = pitch_features.pitch_variance;

    // Intensity (average power in dB)
    double power = features.energy;
    features.intensity = (power > 0.0) ? 10.0 * std::log10(power) : -60.0;
}

Json::Value AudioFeatureSerializer::serialize_audio_features(const AudioFeatures& features) {
    Json::Value json;

    // Basic features
    json["energy"] = features.energy;
    json["zero_crossing_rate"] = features.zero_crossing_rate;
    json["rms"] = features.rms;

    // Spectral features
    json["spectral_centroid"] = features.spectral_centroid;
    json["spectral_rolloff"] = features.spectral_rolloff;
    json["spectral_flux"] = features.spectral_flux;

    // MFCC coefficients
    Json::Value mfcc_array(Json::arrayValue);
    for (double coeff : features.mfcc_coefficients) {
        mfcc_array.append(coeff);
    }
    json["mfcc_coefficients"] = mfcc_array;

    // Voice activity
    json["voice_activity_probability"] = features.voice_activity_probability;
    json["speech_rate"] = features.speech_rate;
    json["pause_ratio"] = features.pause_ratio;

    // Prosodic features
    json["fundamental_frequency"] = features.fundamental_frequency;
    json["pitch_variance"] = features.pitch_variance;
    json["intensity"] = features.intensity;

    // Noise characteristics
    json["snr"] = features.snr;
    json["noise_floor"] = features.noise_floor;

    // Metadata
    json["timestamp"] = timestamp_to_iso8601(features.timestamp);
    json["duration_seconds"] = features.duration_seconds;
    json["sample_rate"] = static_cast<int>(features.sample_rate);
    json["num_samples"] = static_cast<Json::UInt64>(features.num_samples);

    return json;
}

Json::Value AudioFeatureSerializer::serialize_recognition_features(const RecognitionFeatures& features) {
    Json::Value json;

    // Recognition context
    json["text"] = features.text;
    json["confidence"] = features.confidence;
    json["language"] = features.language;
    json["domain"] = features.domain;

    // Model information
    json["model_name"] = features.model_name;

    Json::Value ensemble_array(Json::arrayValue);
    for (const auto& model : features.model_ensemble) {
        ensemble_array.append(model);
    }
    json["model_ensemble"] = ensemble_array;

    Json::Value model_confidences;
    for (const auto& pair : features.model_confidences) {
        model_confidences[pair.first] = pair.second;
    }
    json["model_confidences"] = model_confidences;

    // Performance metrics
    json["processing_time_ms"] = static_cast<int>(features.processing_time.count());
    json["is_final_result"] = features.is_final_result;
    json["uncertainty_score"] = features.uncertainty_score;

    // Speaker characteristics
    json["speaker_id"] = features.speaker_id;
    json["speaker_confidence"] = features.speaker_confidence;
    json["accent_type"] = features.accent_type;

    // Correction information
    json["ground_truth"] = features.ground_truth;
    json["correction_type"] = features.correction_type;
    json["user_corrected"] = features.user_corrected;

    return json;
}

Json::Value AudioFeatureSerializer::serialize_context_features(const ContextFeatures& features) {
    Json::Value json;

    // Environmental context
    json["background_noise_level"] = features.background_noise_level;
    json["audio_source_type"] = features.audio_source_type;
    json["is_streaming"] = features.is_streaming;

    // Content context
    Json::Value previous_utterances(Json::arrayValue);
    for (const auto& utterance : features.previous_utterances) {
        previous_utterances.append(utterance);
    }
    json["previous_utterances"] = previous_utterances;

    json["conversation_topic"] = features.conversation_topic;

    Json::Value topic_probs;
    for (const auto& pair : features.topic_probabilities) {
        topic_probs[pair.first] = pair.second;
    }
    json["topic_probabilities"] = topic_probs;

    // User context
    json["user_id"] = features.user_id;
    json["session_id"] = features.session_id;
    json["session_start"] = timestamp_to_iso8601(features.session_start);

    // Technical context
    json["device_type"] = features.device_type;
    json["os_version"] = features.os_version;
    json["app_version"] = features.app_version;

    return json;
}

Json::Value AudioFeatureSerializer::serialize_complete_submission(
    const AudioFeatures& audio_features,
    const RecognitionFeatures& recognition_features,
    const ContextFeatures& context_features) {

    Json::Value submission;

    submission["audio_features"] = serialize_audio_features(audio_features);
    submission["recognition_features"] = serialize_recognition_features(recognition_features);
    submission["context_features"] = serialize_context_features(context_features);

    // Combined feature vector for ML models
    auto combined_vector = create_combined_feature_vector(audio_features, recognition_features, context_features);
    Json::Value feature_vector(Json::arrayValue);
    for (double feature : combined_vector) {
        feature_vector.append(feature);
    }
    submission["combined_feature_vector"] = feature_vector;

    // Submission metadata
    submission["submission_timestamp"] = timestamp_to_iso8601(std::chrono::steady_clock::now());
    submission["feature_version"] = "1.0";

    return submission;
}

std::vector<double> AudioFeatureSerializer::audio_features_to_vector(const AudioFeatures& features) {
    std::vector<double> vector;

    // Basic features
    vector.push_back(features.energy);
    vector.push_back(features.zero_crossing_rate);
    vector.push_back(features.rms);

    // Spectral features
    vector.push_back(features.spectral_centroid);
    vector.push_back(features.spectral_rolloff);
    vector.push_back(features.spectral_flux);

    // MFCC coefficients
    vector.insert(vector.end(), features.mfcc_coefficients.begin(), features.mfcc_coefficients.end());

    // Voice activity
    vector.push_back(features.voice_activity_probability);
    vector.push_back(features.speech_rate);
    vector.push_back(features.pause_ratio);

    // Prosodic features
    vector.push_back(features.fundamental_frequency);
    vector.push_back(features.pitch_variance);
    vector.push_back(features.intensity);

    // Noise characteristics
    vector.push_back(features.snr);
    vector.push_back(features.noise_floor);

    // Metadata
    vector.push_back(features.duration_seconds);
    vector.push_back(static_cast<double>(features.sample_rate));

    return vector;
}

std::vector<double> AudioFeatureSerializer::recognition_features_to_vector(const RecognitionFeatures& features) {
    std::vector<double> vector;

    // Confidence and uncertainty
    vector.push_back(features.confidence);
    vector.push_back(features.uncertainty_score);

    // Performance metrics
    vector.push_back(static_cast<double>(features.processing_time.count()));
    vector.push_back(features.is_final_result ? 1.0 : 0.0);

    // Speaker characteristics
    vector.push_back(features.speaker_confidence);

    // Correction information
    vector.push_back(features.user_corrected ? 1.0 : 0.0);

    // Text length
    vector.push_back(static_cast<double>(features.text.length()));

    return vector;
}

std::vector<double> AudioFeatureSerializer::context_features_to_vector(const ContextFeatures& features) {
    std::vector<double> vector;

    // Environmental context
    vector.push_back(features.background_noise_level);
    vector.push_back(features.is_streaming ? 1.0 : 0.0);

    // Content context
    vector.push_back(static_cast<double>(features.previous_utterances.size()));

    // Topic probabilities (top 3)
    std::vector<std::pair<std::string, double>> sorted_topics(
        features.topic_probabilities.begin(), features.topic_probabilities.end());
    std::sort(sorted_topics.begin(), sorted_topics.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (size_t i = 0; i < 3 && i < sorted_topics.size(); ++i) {
        vector.push_back(sorted_topics[i].second);
    }

    // Pad to ensure consistent size
    while (vector.size() < 6) {
        vector.push_back(0.0);
    }

    return vector;
}

std::vector<double> AudioFeatureSerializer::create_combined_feature_vector(
    const AudioFeatures& audio_features,
    const RecognitionFeatures& recognition_features,
    const ContextFeatures& context_features) {

    auto audio_vec = audio_features_to_vector(audio_features);
    auto recognition_vec = recognition_features_to_vector(recognition_features);
    auto context_vec = context_features_to_vector(context_features);

    // Combine all feature vectors
    std::vector<double> combined;
    combined.reserve(audio_vec.size() + recognition_vec.size() + context_vec.size());

    combined.insert(combined.end(), audio_vec.begin(), audio_vec.end());
    combined.insert(combined.end(), recognition_vec.begin(), recognition_vec.end());
    combined.insert(combined.end(), context_vec.begin(), context_vec.end());

    return combined;
}

std::string AudioFeatureSerializer::timestamp_to_iso8601(const std::chrono::steady_clock::time_point& timestamp) {
    // Convert steady_clock to system_clock approximation
    auto now_steady = std::chrono::steady_clock::now();
    auto now_system = std::chrono::system_clock::now();
    auto system_timestamp = now_system + (timestamp - now_steady);

    auto time_t = std::chrono::system_clock::to_time_t(system_timestamp);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        system_timestamp.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << milliseconds.count() << 'Z';

    return oss.str();
}

void AudioFeatureSerializer::normalize_features(std::vector<double>& features) {
    if (features.empty()) {
        return;
    }

    double min_val = *std::min_element(features.begin(), features.end());
    double max_val = *std::max_element(features.begin(), features.end());
    double range = max_val - min_val;

    if (range > 0.0) {
        for (auto& feature : features) {
            feature = (feature - min_val) / range;
        }
    }
}

double AudioFeatureSerializer::calculate_mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double AudioFeatureSerializer::calculate_variance(const std::vector<double>& values, double mean) {
    if (values.empty()) {
        return 0.0;
    }

    double sum_squared_diff = 0.0;
    for (double value : values) {
        double diff = value - mean;
        sum_squared_diff += diff * diff;
    }

    return sum_squared_diff / values.size();
}

double AudioFeatureSerializer::calculate_percentile(const std::vector<double>& values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }

    std::vector<double> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());

    double index = percentile / 100.0 * (sorted_values.size() - 1);
    size_t lower_index = static_cast<size_t>(index);
    size_t upper_index = std::min(lower_index + 1, sorted_values.size() - 1);

    double weight = index - lower_index;
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight;
}

// MFCCExtractor implementation
MFCCExtractor::MFCCExtractor(int num_coefficients, int num_filters, double min_freq, double max_freq)
    : num_coefficients_(num_coefficients), num_filters_(num_filters),
      min_freq_(min_freq), max_freq_(max_freq) {
}

std::vector<double> MFCCExtractor::extract(const float* samples, size_t num_samples, uint32_t sample_rate) {
    std::vector<double> mfcc_coefficients(num_coefficients_, 0.0);

    if (!samples || num_samples == 0) {
        return mfcc_coefficients;
    }

    // This is a simplified MFCC implementation
    // In a production system, you would use a more sophisticated implementation

    size_t fft_size = 1024;
    if (num_samples < fft_size) {
        fft_size = num_samples;
    }

    // Apply window and compute FFT
    std::vector<double> windowed_samples(samples, samples + fft_size);

    // Simple Hamming window
    for (size_t i = 0; i < fft_size; ++i) {
        double window = 0.54 - 0.46 * std::cos(2.0 * M_PI * i / (fft_size - 1));
        windowed_samples[i] *= window;
    }

    // Compute power spectrum (simplified)
    for (size_t i = 0; i < std::min(static_cast<size_t>(num_coefficients_), fft_size/2); ++i) {
        double real_part = windowed_samples[i];
        double imag_part = (i < fft_size/2) ? windowed_samples[fft_size - 1 - i] : 0.0;
        mfcc_coefficients[i] = std::log(real_part * real_part + imag_part * imag_part + 1e-10);
    }

    return mfcc_coefficients;
}

// SpectralFeatureExtractor implementation
SpectralFeatureExtractor::SpectralFeatures SpectralFeatureExtractor::extract(
    const float* samples, size_t num_samples, uint32_t sample_rate) {

    SpectralFeatures features;

    if (!samples || num_samples == 0) {
        return features;
    }

    size_t fft_size = std::min(size_t(1024), num_samples);
    std::vector<double> magnitude_spectrum(fft_size / 2);
    std::vector<double> frequencies(fft_size / 2);

    // Generate frequency bins
    for (size_t i = 0; i < fft_size / 2; ++i) {
        frequencies[i] = static_cast<double>(i) * sample_rate / fft_size;
    }

    // Simplified magnitude spectrum calculation
    for (size_t i = 0; i < fft_size / 2; ++i) {
        if (i < num_samples) {
            magnitude_spectrum[i] = std::abs(samples[i]);
        } else {
            magnitude_spectrum[i] = 0.0;
        }
    }

    features.centroid = calculate_spectral_centroid(magnitude_spectrum, frequencies);
    features.rolloff = calculate_spectral_rolloff(magnitude_spectrum, frequencies);

    return features;
}

double SpectralFeatureExtractor::calculate_spectral_centroid(
    const std::vector<double>& magnitude_spectrum,
    const std::vector<double>& frequencies) {

    double weighted_sum = 0.0;
    double magnitude_sum = 0.0;

    for (size_t i = 0; i < magnitude_spectrum.size(); ++i) {
        weighted_sum += frequencies[i] * magnitude_spectrum[i];
        magnitude_sum += magnitude_spectrum[i];
    }

    return (magnitude_sum > 0.0) ? weighted_sum / magnitude_sum : 0.0;
}

double SpectralFeatureExtractor::calculate_spectral_rolloff(
    const std::vector<double>& magnitude_spectrum,
    const std::vector<double>& frequencies,
    double rolloff_threshold) {

    double total_energy = 0.0;
    for (double mag : magnitude_spectrum) {
        total_energy += mag * mag;
    }

    double cumulative_energy = 0.0;
    double threshold_energy = rolloff_threshold * total_energy;

    for (size_t i = 0; i < magnitude_spectrum.size(); ++i) {
        cumulative_energy += magnitude_spectrum[i] * magnitude_spectrum[i];
        if (cumulative_energy >= threshold_energy) {
            return frequencies[i];
        }
    }

    return frequencies.empty() ? 0.0 : frequencies.back();
}

// PitchExtractor implementation
PitchExtractor::PitchFeatures PitchExtractor::extract(const float* samples, size_t num_samples, uint32_t sample_rate) {
    PitchFeatures features;

    if (!samples || num_samples == 0) {
        return features;
    }

    features.fundamental_frequency = estimate_f0_autocorrelation(samples, num_samples, sample_rate);
    features.pitch_confidence = 0.8; // Simplified confidence estimate

    // Calculate pitch variance (simplified)
    std::vector<double> pitch_values = {features.fundamental_frequency};
    features.pitch_variance = 0.1; // Placeholder

    features.voicing_probability = (features.fundamental_frequency > 50.0) ? 0.8 : 0.2;

    return features;
}

double PitchExtractor::estimate_f0_autocorrelation(const float* samples, size_t num_samples, uint32_t sample_rate) {
    if (num_samples < 2) {
        return 0.0;
    }

    // Simplified autocorrelation-based F0 estimation
    size_t max_lag = std::min(size_t(sample_rate / 50), num_samples / 2); // Max lag for 50 Hz
    size_t min_lag = sample_rate / 800; // Min lag for 800 Hz

    double max_correlation = 0.0;
    size_t best_lag = min_lag;

    for (size_t lag = min_lag; lag < max_lag; ++lag) {
        double correlation = 0.0;
        double norm1 = 0.0, norm2 = 0.0;

        for (size_t i = 0; i < num_samples - lag; ++i) {
            correlation += samples[i] * samples[i + lag];
            norm1 += samples[i] * samples[i];
            norm2 += samples[i + lag] * samples[i + lag];
        }

        if (norm1 > 0.0 && norm2 > 0.0) {
            correlation /= std::sqrt(norm1 * norm2);
            if (correlation > max_correlation) {
                max_correlation = correlation;
                best_lag = lag;
            }
        }
    }

    return (max_correlation > 0.3) ? static_cast<double>(sample_rate) / best_lag : 0.0;
}

} // namespace vtt