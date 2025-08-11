#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace vtt {

// Speaker embedding for identification
class SpeakerEmbedding {
public:
    static constexpr size_t EMBEDDING_DIM = 256;
    using Vector = std::vector<float>;
    
    SpeakerEmbedding() : embedding_(EMBEDDING_DIM, 0.0f) {}
    explicit SpeakerEmbedding(const Vector& emb) : embedding_(emb) {}
    
    // Compute cosine similarity between embeddings
    float similarity(const SpeakerEmbedding& other) const {
        float dot_product = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;
        
        for (size_t i = 0; i < EMBEDDING_DIM; ++i) {
            dot_product += embedding_[i] * other.embedding_[i];
            norm_a += embedding_[i] * embedding_[i];
            norm_b += other.embedding_[i] * other.embedding_[i];
        }
        
        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);
        
        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;
        }
        
        return dot_product / (norm_a * norm_b);
    }
    
    // Update embedding with new sample (running average)
    void update(const Vector& new_embedding, float weight = 0.1f) {
        for (size_t i = 0; i < EMBEDDING_DIM; ++i) {
            embedding_[i] = (1.0f - weight) * embedding_[i] + weight * new_embedding[i];
        }
        normalize();
    }
    
    void normalize() {
        float norm = 0.0f;
        for (float val : embedding_) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0.0f) {
            for (float& val : embedding_) {
                val /= norm;
            }
        }
    }
    
    const Vector& get() const { return embedding_; }
    
private:
    Vector embedding_;
};

// Speaker profile
struct SpeakerProfile {
    std::string id;
    std::string name;
    SpeakerEmbedding embedding;
    size_t sample_count = 0;
    float confidence = 0.0f;
    std::chrono::steady_clock::time_point last_seen;
    
    SpeakerProfile(const std::string& speaker_id) 
        : id(speaker_id)
        , name("Speaker " + speaker_id)
        , last_seen(std::chrono::steady_clock::now()) {}
};

// Speaker diarization system
class SpeakerDiarization {
public:
    struct Config {
        float similarity_threshold = 0.85f;  // Threshold for same speaker
        size_t max_speakers = 10;            // Maximum number of speakers to track
        size_t embedding_window_ms = 500;    // Window size for embedding extraction
        float min_speech_duration_ms = 200;  // Minimum speech duration to process
        bool auto_create_profiles = true;    // Auto-create new speaker profiles
    };
    
    struct SegmentInfo {
        std::string speaker_id;
        size_t start_sample;
        size_t end_sample;
        float confidence;
        std::string text;
    };
    
    SpeakerDiarization(const Config& config = Config());
    
    // Process audio and identify speaker
    std::string process_audio(const float* samples, size_t num_samples, 
                             uint32_t sample_rate);
    
    // Add transcription for current segment
    void add_transcription(const std::string& text);
    
    // Get or create speaker profile
    std::shared_ptr<SpeakerProfile> get_or_create_speaker(
        const SpeakerEmbedding& embedding);
    
    // Get all segments with speaker labels
    std::vector<SegmentInfo> get_segments() const { return segments_; }
    
    // Get all speaker profiles
    std::unordered_map<std::string, std::shared_ptr<SpeakerProfile>> 
        get_speakers() const { return speakers_; }
    
    // Clear all data
    void clear();
    
    // Export diarization results
    std::string export_results() const;
    
private:
    // Extract speaker embedding from audio
    SpeakerEmbedding extract_embedding(const float* samples, size_t num_samples,
                                       uint32_t sample_rate);
    
    // Simple MFCC feature extraction for speaker embedding
    std::vector<float> extract_mfcc(const float* samples, size_t num_samples,
                                    uint32_t sample_rate);
    
    // Find best matching speaker
    std::shared_ptr<SpeakerProfile> find_best_speaker(
        const SpeakerEmbedding& embedding);
    
    // Generate new speaker ID
    std::string generate_speaker_id();
    
    Config config_;
    std::unordered_map<std::string, std::shared_ptr<SpeakerProfile>> speakers_;
    std::vector<SegmentInfo> segments_;
    std::string current_speaker_id_;
    size_t current_segment_start_ = 0;
    size_t total_samples_processed_ = 0;
    int next_speaker_num_ = 1;
};

// Implementation
inline SpeakerDiarization::SpeakerDiarization(const Config& config) 
    : config_(config) {
}

inline std::string SpeakerDiarization::process_audio(const float* samples, 
                                                    size_t num_samples,
                                                    uint32_t sample_rate) {
    // Extract speaker embedding
    auto embedding = extract_embedding(samples, num_samples, sample_rate);
    
    // Find or create speaker
    auto speaker = get_or_create_speaker(embedding);
    
    // Check if speaker changed
    if (speaker->id != current_speaker_id_) {
        // Save previous segment if exists
        if (!current_speaker_id_.empty()) {
            SegmentInfo segment;
            segment.speaker_id = current_speaker_id_;
            segment.start_sample = current_segment_start_;
            segment.end_sample = total_samples_processed_;
            segment.confidence = 0.9f;  // Placeholder
            segments_.push_back(segment);
        }
        
        // Start new segment
        current_speaker_id_ = speaker->id;
        current_segment_start_ = total_samples_processed_;
    }
    
    total_samples_processed_ += num_samples;
    speaker->last_seen = std::chrono::steady_clock::now();
    
    return speaker->name;
}

inline void SpeakerDiarization::add_transcription(const std::string& text) {
    if (!segments_.empty()) {
        segments_.back().text = text;
    }
}

inline std::shared_ptr<SpeakerProfile> SpeakerDiarization::get_or_create_speaker(
    const SpeakerEmbedding& embedding) {
    
    // Find best matching existing speaker
    auto best_speaker = find_best_speaker(embedding);
    
    if (best_speaker) {
        // Update existing speaker embedding
        best_speaker->embedding.update(embedding.get(), 0.1f);
        best_speaker->sample_count++;
        return best_speaker;
    }
    
    // Create new speaker if auto-create enabled
    if (config_.auto_create_profiles && speakers_.size() < config_.max_speakers) {
        std::string new_id = generate_speaker_id();
        auto new_speaker = std::make_shared<SpeakerProfile>(new_id);
        new_speaker->embedding = embedding;
        new_speaker->sample_count = 1;
        new_speaker->confidence = 0.8f;
        speakers_[new_id] = new_speaker;
        return new_speaker;
    }
    
    // Return unknown speaker
    if (speakers_.find("unknown") == speakers_.end()) {
        speakers_["unknown"] = std::make_shared<SpeakerProfile>("unknown");
    }
    return speakers_["unknown"];
}

inline SpeakerEmbedding SpeakerDiarization::extract_embedding(
    const float* samples, size_t num_samples, uint32_t sample_rate) {
    
    // Extract MFCC features
    auto features = extract_mfcc(samples, num_samples, sample_rate);
    
    // Simple embedding: use statistical features of MFCCs
    SpeakerEmbedding::Vector embedding(SpeakerEmbedding::EMBEDDING_DIM, 0.0f);
    
    // Fill embedding with MFCC statistics
    size_t feature_dim = std::min(features.size(), embedding.size() / 4);
    
    for (size_t i = 0; i < feature_dim && i < features.size(); ++i) {
        size_t base_idx = i * 4;
        if (base_idx + 3 < embedding.size()) {
            embedding[base_idx] = features[i];  // Mean
            embedding[base_idx + 1] = std::abs(features[i]);  // Abs
            embedding[base_idx + 2] = features[i] * features[i];  // Power
            embedding[base_idx + 3] = std::tanh(features[i]);  // Tanh
        }
    }
    
    SpeakerEmbedding result(embedding);
    result.normalize();
    return result;
}

inline std::vector<float> SpeakerDiarization::extract_mfcc(
    const float* samples, size_t num_samples, uint32_t sample_rate) {
    
    // Simplified MFCC extraction (placeholder)
    // In production, use proper MFCC library
    const size_t num_coeffs = 13;
    std::vector<float> mfcc(num_coeffs, 0.0f);
    
    // Simple spectral features as placeholder
    const size_t fft_size = 512;
    const size_t hop_size = 256;
    
    for (size_t i = 0; i < num_samples - fft_size; i += hop_size) {
        // Compute energy in different frequency bands
        for (size_t j = 0; j < num_coeffs; ++j) {
            float band_energy = 0.0f;
            size_t band_start = j * (fft_size / num_coeffs);
            size_t band_end = (j + 1) * (fft_size / num_coeffs);
            
            for (size_t k = band_start; k < band_end && i + k < num_samples; ++k) {
                band_energy += samples[i + k] * samples[i + k];
            }
            
            mfcc[j] += std::log1p(band_energy);
        }
    }
    
    // Normalize
    float sum = 0.0f;
    for (float& val : mfcc) {
        sum += val * val;
    }
    if (sum > 0.0f) {
        float norm = std::sqrt(sum);
        for (float& val : mfcc) {
            val /= norm;
        }
    }
    
    return mfcc;
}

inline std::shared_ptr<SpeakerProfile> SpeakerDiarization::find_best_speaker(
    const SpeakerEmbedding& embedding) {
    
    std::shared_ptr<SpeakerProfile> best_speaker = nullptr;
    float best_similarity = config_.similarity_threshold;
    
    for (const auto& [id, speaker] : speakers_) {
        float similarity = embedding.similarity(speaker->embedding);
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_speaker = speaker;
        }
    }
    
    return best_speaker;
}

inline std::string SpeakerDiarization::generate_speaker_id() {
    return std::to_string(next_speaker_num_++);
}

inline void SpeakerDiarization::clear() {
    speakers_.clear();
    segments_.clear();
    current_speaker_id_.clear();
    current_segment_start_ = 0;
    total_samples_processed_ = 0;
    next_speaker_num_ = 1;
}

inline std::string SpeakerDiarization::export_results() const {
    std::stringstream ss;
    ss << "=== Speaker Diarization Results ===\n";
    ss << "Total speakers: " << speakers_.size() << "\n";
    ss << "Total segments: " << segments_.size() << "\n\n";
    
    for (const auto& segment : segments_) {
        float start_time = segment.start_sample / 16000.0f;
        float end_time = segment.end_sample / 16000.0f;
        ss << "[" << std::fixed << std::setprecision(2) 
           << start_time << "s - " << end_time << "s] "
           << "Speaker " << segment.speaker_id << ": "
           << segment.text << "\n";
    }
    
    return ss.str();
}

}  // namespace vtt