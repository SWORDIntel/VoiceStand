#pragma once

#include "whisper_processor.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

namespace vtt {

// Forward declarations
class AdaptiveLearningSystem;
class LearningSystemIntegration;
class UKDialectOptimizer;
struct AudioContext;
struct RecognitionResult;

// Advanced ensemble recognition system for 94-99% accuracy
class EnsembleWhisperProcessor {
public:
    enum class LearningMode {
        CONSERVATIVE,   // Use proven high-accuracy models
        BALANCED,       // Balance accuracy vs speed
        EXPERIMENTAL,   // Test new model combinations
        ADAPTIVE        // Dynamically adjust based on performance
    };

    struct EnsembleConfig {
        // Model ensemble settings
        std::vector<std::string> model_paths = {
            "models/ggml-small.bin",     // 79% baseline
            "models/ggml-medium.bin",    // 85% baseline
            "models/ggml-large.bin",     // 88% baseline
            "models/uk-english-fine-tuned-small.bin",  // UK specialization
            "models/uk-english-fine-tuned-medium.bin"  // UK specialization
        };

        // Learning system settings
        double accuracy_target = 0.95;
        double confidence_threshold = 0.85;
        size_t min_ensemble_size = 3;
        size_t max_ensemble_size = 5;

        // UK English specialization
        bool enable_uk_dialect_optimization = true;
        bool enable_accent_adaptation = true;
        bool enable_vocabulary_learning = true;

        // Performance optimization
        bool enable_model_caching = true;
        bool enable_parallel_inference = true;
        size_t max_parallel_models = 3;

        // Learning parameters
        LearningMode learning_mode = LearningMode::ADAPTIVE;
        std::chrono::seconds learning_update_interval{300}; // 5 minutes
    };

    // Using global RecognitionResult type from learning_api_client.h
    // to avoid type conflicts and maintain API compatibility

    struct PerformanceMetrics {
        std::atomic<size_t> total_recognitions{0};
        std::atomic<double> current_accuracy{0.88};
        std::atomic<double> average_confidence{0.0};
        std::atomic<size_t> ensemble_agreements{0};
        std::atomic<size_t> model_switches{0};
        std::atomic<double> uk_dialect_accuracy{0.0};
        std::map<std::string, double> model_individual_accuracies;
    };

private:
    // Core processors
    std::vector<std::unique_ptr<WhisperProcessor>> models_;
    std::unique_ptr<AdaptiveLearningSystem> learning_system_;

    // Configuration
    EnsembleConfig config_;
    PerformanceMetrics metrics_;

    // Threading and synchronization
    std::mutex model_mutex_;
    std::mutex learning_mutex_;
    std::thread learning_thread_;
    std::atomic<bool> learning_active_{false};

    // Model selection and weighting
    std::map<std::string, double> model_weights_;
    std::map<std::string, std::chrono::steady_clock::time_point> model_last_updated_;
    std::deque<vtt::RecognitionResult> recent_results_;

    // UK English specialization
    std::unique_ptr<UKDialectOptimizer> uk_optimizer_;
    std::map<std::string, double> uk_vocabulary_scores_;

    // API learning integration
    std::shared_ptr<LearningSystemIntegration> learning_integration_;
    mutable std::mutex learning_integration_mutex_;

public:
    EnsembleWhisperProcessor();
    EnsembleWhisperProcessor(const EnsembleConfig& config);
    ~EnsembleWhisperProcessor();

    // Core functionality
    bool initialize();
    vtt::RecognitionResult recognize(const float* audio_data, size_t num_samples,
                              uint32_t sample_rate);
    void cleanup();

    // Ensemble methods
    vtt::RecognitionResult ensemble_recognize(const float* audio_data,
                                       size_t num_samples);
    std::string combine_results(const std::vector<vtt::RecognitionResult>& results);
    double calculate_ensemble_confidence(const std::vector<vtt::RecognitionResult>& results);

    // Learning and adaptation
    void update_model_weights(const vtt::RecognitionResult& result,
                            const std::string& ground_truth);
    void adapt_to_speaker(const std::vector<vtt::RecognitionResult>& speaker_history);
    bool should_retrain_models();
    void trigger_model_retraining();

    // UK English specialization
    void optimize_for_uk_dialect(const std::vector<std::string>& uk_samples);
    void add_uk_vocabulary(const std::map<std::string, double>& new_words);
    double calculate_uk_dialect_score(const std::string& text);

    // Model management
    bool add_model(const std::string& model_path, double initial_weight = 1.0);
    bool remove_model(const std::string& model_path);
    void rebalance_model_weights();
    std::vector<std::string> select_optimal_models(const AudioContext& context);

    // Performance monitoring
    const PerformanceMetrics& get_metrics() const { return metrics_; }
    double get_current_accuracy() const { return metrics_.current_accuracy.load(); }
    void report_ground_truth(const std::string& recognized_text,
                           const std::string& actual_text);

    // Learning system interface
    void set_learning_mode(LearningMode mode);
    LearningMode get_learning_mode() const { return config_.learning_mode; }
    void start_continuous_learning();
    void stop_continuous_learning();

    // API learning integration
    void set_learning_integration(std::shared_ptr<LearningSystemIntegration> integration);
    std::shared_ptr<LearningSystemIntegration> get_learning_integration() const;

private:
    // Internal methods
    void learning_loop();
    void update_accuracy_metrics(const vtt::RecognitionResult& result,
                               const std::string& ground_truth = "");
    void log_performance_data(const vtt::RecognitionResult& result);
    std::vector<size_t> select_models_for_inference(const AudioContext& context);
    void preprocess_for_uk_dialect(float* audio_data, size_t num_samples);
    bool is_uk_english_text(const std::string& text);
    void update_uk_vocabulary_learning(const std::string& recognized_text);
};

// UK English dialect optimizer
class UKDialectOptimizer {
public:
    struct UKDialectFeatures {
        double rhoticity_score = 0.0;        // R-dropping characteristics
        double vowel_system_score = 0.0;     // UK vowel system detection
        double lexical_choice_score = 0.0;   // British vs American vocabulary
        double intonation_score = 0.0;       // British intonation patterns
    };

    UKDialectOptimizer();

    UKDialectFeatures analyze_uk_features(const float* audio_data,
                                        size_t num_samples);
    double calculate_uk_probability(const UKDialectFeatures& features);
    std::string optimize_text_for_uk(const std::string& raw_text);
    void add_uk_vocabulary_mapping(const std::string& american_term,
                                 const std::string& british_term);

private:
    std::map<std::string, std::string> american_to_british_vocab_;
    std::vector<std::string> uk_specific_phrases_;
    std::map<std::string, double> uk_pronunciation_patterns_;
};

// Audio context for model selection
struct AudioContext {
    double noise_level = 0.0;
    double speech_rate = 1.0;           // Words per minute ratio
    bool is_phone_call = false;
    bool is_conference_call = false;
    std::string speaker_id = "";
    double accent_strength = 0.0;       // 0.0 = neutral, 1.0 = strong regional
    bool is_technical_content = false;
    std::string domain = "general";     // medical, legal, technical, etc.
};

} // namespace vtt