#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <deque>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace vtt {

// Wake word detection using template matching
class WakeWordDetector {
public:
    struct Config {
        float detection_threshold = 0.75f;
        size_t buffer_size_ms = 2000;  // 2 second buffer
        size_t window_size_ms = 500;   // 500ms detection window
        uint32_t sample_rate = 16000;
        bool use_vad = true;           // Use VAD pre-filtering
        float vad_threshold = 0.01f;
    };
    
    struct WakeWord {
        std::string word;
        std::vector<float> template_features;
        float threshold;
        std::function<void()> callback;
        size_t detection_count = 0;
        
        WakeWord(const std::string& w, float thresh = 0.75f) 
            : word(w), threshold(thresh) {}
    };
    
    WakeWordDetector(const Config& config = Config());
    
    // Register a wake word with optional callback
    bool register_wake_word(const std::string& word, 
                           std::function<void()> callback = nullptr,
                           float threshold = 0.75f);
    
    // Train wake word from audio samples
    bool train_wake_word(const std::string& word,
                        const std::vector<std::vector<float>>& samples);
    
    // Process audio and detect wake words
    std::string process_audio(const float* samples, size_t num_samples);
    
    // Check if any wake word was recently detected
    bool is_activated() const { return activated_; }
    
    // Get last detected wake word
    std::string get_last_wake_word() const { return last_wake_word_; }
    
    // Clear detection state
    void reset();
    
    // Save/load wake word models
    bool save_model(const std::string& filepath);
    bool load_model(const std::string& filepath);
    
    // Get statistics
    struct Stats {
        size_t total_detections = 0;
        size_t false_positives = 0;
        float average_confidence = 0.0f;
        std::unordered_map<std::string, size_t> word_counts;
    };
    
    const Stats& get_stats() const { return stats_; }
    
private:
    // Extract features for wake word detection
    std::vector<float> extract_features(const float* samples, size_t num_samples);
    
    // Compute MFCC features
    std::vector<float> compute_mfcc(const float* samples, size_t num_samples);
    
    // Dynamic time warping for template matching
    float dtw_distance(const std::vector<float>& seq1, 
                      const std::vector<float>& seq2);
    
    // Cross-correlation for template matching
    float cross_correlation(const std::vector<float>& signal,
                           const std::vector<float>& template_);
    
    // Simple VAD check
    bool is_speech(const float* samples, size_t num_samples);
    
    // Sliding window detection
    bool detect_in_window(const std::vector<float>& features);
    
    Config config_;
    std::vector<std::unique_ptr<WakeWord>> wake_words_;
    std::deque<float> audio_buffer_;
    bool activated_ = false;
    std::string last_wake_word_;
    std::chrono::steady_clock::time_point last_detection_time_;
    Stats stats_;
    
    // Feature extraction parameters
    static constexpr size_t NUM_MFCC = 13;
    static constexpr size_t FFT_SIZE = 512;
    static constexpr size_t HOP_SIZE = 160;  // 10ms at 16kHz
};

// Implementation
inline WakeWordDetector::WakeWordDetector(const Config& config) 
    : config_(config) {
    size_t buffer_samples = (config.buffer_size_ms * config.sample_rate) / 1000;
    audio_buffer_.resize(buffer_samples, 0.0f);
}

inline bool WakeWordDetector::register_wake_word(const std::string& word,
                                                std::function<void()> callback,
                                                float threshold) {
    auto wake_word = std::make_unique<WakeWord>(word, threshold);
    wake_word->callback = callback;
    
    // Generate simple template (in production, would load pre-trained model)
    // For now, create a placeholder template
    wake_word->template_features.resize(NUM_MFCC * 10, 0.0f);
    
    // Simple pattern based on word length and phonemes
    for (size_t i = 0; i < wake_word->template_features.size(); ++i) {
        wake_word->template_features[i] = 
            std::sin(2.0f * M_PI * i / wake_word->template_features.size()) *
            (1.0f + 0.1f * std::hash<std::string>{}(word));
    }
    
    wake_words_.push_back(std::move(wake_word));
    return true;
}

inline bool WakeWordDetector::train_wake_word(const std::string& word,
                                             const std::vector<std::vector<float>>& samples) {
    // Find the wake word
    auto it = std::find_if(wake_words_.begin(), wake_words_.end(),
                           [&word](const auto& w) { return w->word == word; });
    
    if (it == wake_words_.end()) {
        return false;
    }
    
    // Extract features from all samples and average them
    std::vector<float> avg_features(NUM_MFCC * 10, 0.0f);
    
    for (const auto& sample : samples) {
        auto features = extract_features(sample.data(), sample.size());
        
        // Accumulate features
        for (size_t i = 0; i < std::min(features.size(), avg_features.size()); ++i) {
            avg_features[i] += features[i];
        }
    }
    
    // Average the features
    if (!samples.empty()) {
        for (float& f : avg_features) {
            f /= samples.size();
        }
    }
    
    (*it)->template_features = avg_features;
    return true;
}

inline std::string WakeWordDetector::process_audio(const float* samples, size_t num_samples) {
    // Add samples to buffer
    for (size_t i = 0; i < num_samples; ++i) {
        audio_buffer_.push_back(samples[i]);
        if (audio_buffer_.size() > (config_.buffer_size_ms * config_.sample_rate) / 1000) {
            audio_buffer_.pop_front();
        }
    }
    
    // Check VAD if enabled
    if (config_.use_vad && !is_speech(samples, num_samples)) {
        return "";
    }
    
    // Extract features from current window
    size_t window_samples = (config_.window_size_ms * config_.sample_rate) / 1000;
    if (audio_buffer_.size() < window_samples) {
        return "";
    }
    
    std::vector<float> window(audio_buffer_.end() - window_samples, audio_buffer_.end());
    auto features = extract_features(window.data(), window.size());
    
    // Check each wake word
    std::string detected_word;
    float best_score = 0.0f;
    
    for (const auto& wake_word : wake_words_) {
        // Compute similarity score
        float score = 1.0f - dtw_distance(features, wake_word->template_features);
        
        if (score > wake_word->threshold && score > best_score) {
            best_score = score;
            detected_word = wake_word->word;
        }
    }
    
    // Handle detection
    if (!detected_word.empty()) {
        auto now = std::chrono::steady_clock::now();
        auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_detection_time_).count();
        
        // Debounce detections (avoid multiple triggers)
        if (time_since_last > 1000) {  // 1 second debounce
            activated_ = true;
            last_wake_word_ = detected_word;
            last_detection_time_ = now;
            
            // Update stats
            stats_.total_detections++;
            stats_.word_counts[detected_word]++;
            stats_.average_confidence = 
                (stats_.average_confidence * (stats_.total_detections - 1) + best_score) 
                / stats_.total_detections;
            
            // Call callback if registered
            auto it = std::find_if(wake_words_.begin(), wake_words_.end(),
                                  [&detected_word](const auto& w) { 
                                      return w->word == detected_word; 
                                  });
            
            if (it != wake_words_.end() && (*it)->callback) {
                (*it)->callback();
            }
            
            return detected_word;
        }
    }
    
    return "";
}

inline std::vector<float> WakeWordDetector::extract_features(const float* samples, 
                                                            size_t num_samples) {
    return compute_mfcc(samples, num_samples);
}

inline std::vector<float> WakeWordDetector::compute_mfcc(const float* samples, 
                                                        size_t num_samples) {
    std::vector<float> mfcc_features;
    
    // Simplified MFCC calculation
    for (size_t i = 0; i < num_samples - FFT_SIZE; i += HOP_SIZE) {
        // Compute energy in mel-frequency bands
        std::vector<float> frame_mfcc(NUM_MFCC, 0.0f);
        
        for (size_t j = 0; j < NUM_MFCC; ++j) {
            float band_energy = 0.0f;
            size_t band_start = j * (FFT_SIZE / NUM_MFCC);
            size_t band_end = (j + 1) * (FFT_SIZE / NUM_MFCC);
            
            for (size_t k = band_start; k < band_end && i + k < num_samples; ++k) {
                band_energy += samples[i + k] * samples[i + k];
            }
            
            frame_mfcc[j] = std::log1p(band_energy);
        }
        
        // Add frame MFCCs to features
        mfcc_features.insert(mfcc_features.end(), frame_mfcc.begin(), frame_mfcc.end());
    }
    
    return mfcc_features;
}

inline float WakeWordDetector::dtw_distance(const std::vector<float>& seq1,
                                           const std::vector<float>& seq2) {
    size_t n = seq1.size();
    size_t m = seq2.size();
    
    if (n == 0 || m == 0) return 1.0f;
    
    // Simple DTW implementation
    std::vector<std::vector<float>> dtw(n + 1, std::vector<float>(m + 1, INFINITY));
    dtw[0][0] = 0.0f;
    
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            float cost = std::abs(seq1[i-1] - seq2[j-1]);
            dtw[i][j] = cost + std::min({dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]});
        }
    }
    
    // Normalize by path length
    return dtw[n][m] / (n + m);
}

inline bool WakeWordDetector::is_speech(const float* samples, size_t num_samples) {
    if (!samples || num_samples == 0) return false;
    
    // Simple energy-based VAD
    float energy = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += samples[i] * samples[i];
    }
    energy = std::sqrt(energy / num_samples);
    
    return energy > config_.vad_threshold;
}

inline void WakeWordDetector::reset() {
    activated_ = false;
    last_wake_word_.clear();
    audio_buffer_.clear();
}

inline bool WakeWordDetector::save_model(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;
    
    // Save number of wake words
    size_t num_words = wake_words_.size();
    file.write(reinterpret_cast<const char*>(&num_words), sizeof(num_words));
    
    // Save each wake word
    for (const auto& wake_word : wake_words_) {
        // Save word string
        size_t word_len = wake_word->word.length();
        file.write(reinterpret_cast<const char*>(&word_len), sizeof(word_len));
        file.write(wake_word->word.c_str(), word_len);
        
        // Save threshold
        file.write(reinterpret_cast<const char*>(&wake_word->threshold), 
                  sizeof(wake_word->threshold));
        
        // Save template features
        size_t feat_size = wake_word->template_features.size();
        file.write(reinterpret_cast<const char*>(&feat_size), sizeof(feat_size));
        file.write(reinterpret_cast<const char*>(wake_word->template_features.data()),
                  feat_size * sizeof(float));
    }
    
    return true;
}

inline bool WakeWordDetector::load_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return false;
    
    wake_words_.clear();
    
    // Load number of wake words
    size_t num_words;
    file.read(reinterpret_cast<char*>(&num_words), sizeof(num_words));
    
    // Load each wake word
    for (size_t i = 0; i < num_words; ++i) {
        // Load word string
        size_t word_len;
        file.read(reinterpret_cast<char*>(&word_len), sizeof(word_len));
        
        std::string word(word_len, '\0');
        file.read(&word[0], word_len);
        
        // Load threshold
        float threshold;
        file.read(reinterpret_cast<char*>(&threshold), sizeof(threshold));
        
        auto wake_word = std::make_unique<WakeWord>(word, threshold);
        
        // Load template features
        size_t feat_size;
        file.read(reinterpret_cast<char*>(&feat_size), sizeof(feat_size));
        
        wake_word->template_features.resize(feat_size);
        file.read(reinterpret_cast<char*>(wake_word->template_features.data()),
                 feat_size * sizeof(float));
        
        wake_words_.push_back(std::move(wake_word));
    }
    
    return true;
}

}  // namespace vtt