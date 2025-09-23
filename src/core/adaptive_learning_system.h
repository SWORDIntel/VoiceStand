#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>
#include <deque>
#include <json/json.h>

namespace vtt {

// Forward declarations
class LearningSystemIntegration;
struct RecognitionResult;

// Advanced learning system based on claude-backups architecture
class AdaptiveLearningSystem {
public:
    enum class PatternType {
        ACOUSTIC_PATTERN,      // Audio characteristics leading to better recognition
        LEXICAL_PATTERN,       // Word/phrase patterns with high accuracy
        CONTEXTUAL_PATTERN,    // Context-dependent recognition improvements
        SPEAKER_PATTERN,       // Speaker-specific adaptations
        UK_DIALECT_PATTERN     // UK English dialect-specific patterns
    };

    struct LearningPattern {
        PatternType type;
        std::string pattern_id;
        std::map<std::string, double> features;
        double confidence_score = 0.0;
        double accuracy_improvement = 0.0;
        size_t usage_count = 0;
        std::chrono::steady_clock::time_point last_used;
        std::chrono::steady_clock::time_point created_at;
        bool is_uk_specific = false;
    };

    struct LearningInsight {
        std::string insight_type;
        std::string description;
        double confidence = 0.0;
        std::map<std::string, double> recommended_adjustments;
        std::vector<std::string> affected_models;
        bool requires_retraining = false;
    };

    struct LearningStats {
        std::atomic<size_t> patterns_learned{0};
        std::atomic<size_t> successful_adaptations{0};
        std::atomic<double> average_accuracy_improvement{0.0};
        std::atomic<size_t> uk_patterns_learned{0};
        std::atomic<double> uk_accuracy_improvement{0.0};
        std::chrono::steady_clock::time_point last_learning_update;

        // Delete copy constructor and assignment to make error explicit
        LearningStats(const LearningStats&) = delete;
        LearningStats& operator=(const LearningStats&) = delete;

        // Default constructor and move operations are still allowed
        LearningStats() = default;
        LearningStats(LearningStats&&) = default;
        LearningStats& operator=(LearningStats&&) = default;
    };

    // Non-atomic version for safe copying
    struct LearningStatsSnapshot {
        size_t patterns_learned;
        size_t successful_adaptations;
        double average_accuracy_improvement;
        size_t uk_patterns_learned;
        double uk_accuracy_improvement;
        std::chrono::steady_clock::time_point last_learning_update;
    };

    struct RecognitionContext {
        std::string text;
        double confidence;
        std::vector<double> acoustic_features;
        std::string speaker_id;
        std::string domain;
        bool is_uk_english = false;
        std::chrono::steady_clock::time_point timestamp;
        std::map<std::string, double> model_outputs;
    };

private:
    // Learning database (PostgreSQL with pgvector)
    std::string learning_db_url_;
    std::unique_ptr<class PostgreSQLLearningDB> learning_db_;

    // Pattern storage and analysis
    std::map<std::string, LearningPattern> learned_patterns_;
    std::deque<RecognitionContext> recognition_history_;
    std::mutex patterns_mutex_;
    std::mutex history_mutex_;

    // Learning configuration
    size_t max_history_size_ = 10000;
    double pattern_confidence_threshold_ = 0.7;
    double uk_pattern_bonus_ = 0.1;  // Extra weight for UK patterns
    size_t min_pattern_usage_ = 5;

    // Performance tracking
    LearningStats stats_;
    std::map<std::string, double> model_performance_history_;

    // UK English specialization
    std::unique_ptr<class UKEnglishSpecializer> uk_specializer_;
    std::map<std::string, double> uk_vocabulary_patterns_;

    // API learning integration
    std::shared_ptr<LearningSystemIntegration> learning_integration_;
    mutable std::mutex learning_integration_mutex_;

public:
    explicit AdaptiveLearningSystem(const std::string& db_url);
    ~AdaptiveLearningSystem();

    // Core learning functionality
    bool initialize();
    void cleanup();

    // Pattern learning and recognition
    void record_recognition(const RecognitionContext& context,
                          const std::string& ground_truth = "");
    std::vector<LearningInsight> analyze_patterns();
    LearningPattern extract_pattern(const std::vector<RecognitionContext>& contexts);
    bool should_apply_pattern(const LearningPattern& pattern,
                            const RecognitionContext& context);

    // Model optimization
    std::map<std::string, double> get_optimal_model_weights();
    std::vector<std::string> recommend_model_combinations();
    bool should_retrain_model(const std::string& model_name);
    Json::Value generate_training_recommendations();

    // UK English specialization
    void enable_uk_english_learning(bool enable = true);
    void add_uk_training_data(const std::vector<std::pair<std::string, std::string>>& audio_text_pairs);
    double calculate_uk_dialect_confidence(const RecognitionContext& context);
    std::map<std::string, double> get_uk_vocabulary_adjustments();

    // Performance analysis
    double predict_accuracy_improvement(const std::vector<std::string>& model_combination);
    std::vector<LearningInsight> identify_improvement_opportunities();
    LearningStatsSnapshot get_learning_stats() const {
        return {
            stats_.patterns_learned.load(),
            stats_.successful_adaptations.load(),
            stats_.average_accuracy_improvement.load(),
            stats_.uk_patterns_learned.load(),
            stats_.uk_accuracy_improvement.load(),
            stats_.last_learning_update
        };
    }

    // Database operations
    bool save_patterns_to_db();
    bool load_patterns_from_db();
    void track_performance(const std::string& model_name,
                         const std::string& task_type,
                         double accuracy,
                         std::chrono::milliseconds duration);

    // Real-time adaptation
    void start_real_time_learning();
    void stop_real_time_learning();
    void update_learning_parameters(const Json::Value& config);

    // API learning integration
    void set_learning_integration(std::shared_ptr<LearningSystemIntegration> integration);
    std::shared_ptr<LearningSystemIntegration> get_learning_integration() const;

private:
    // Internal learning methods
    void analyze_acoustic_patterns();
    void analyze_lexical_patterns();
    void analyze_contextual_patterns();
    void analyze_uk_dialect_patterns();

    // Pattern analysis helpers
    std::vector<double> extract_acoustic_features(const RecognitionContext& context);
    std::map<std::string, double> extract_lexical_features(const std::string& text);
    double calculate_pattern_similarity(const LearningPattern& pattern1,
                                      const LearningPattern& pattern2);

    // UK English specific methods
    bool is_uk_english_context(const RecognitionContext& context);
    void update_uk_vocabulary_model(const std::string& text);
    double calculate_uk_lexical_score(const std::string& text);

    // Database helpers
    void insert_pattern_to_db(const LearningPattern& pattern);
    void update_pattern_usage(const std::string& pattern_id);
    std::vector<LearningPattern> query_similar_patterns(const LearningPattern& pattern);
};

// PostgreSQL database interface for learning data
class PostgreSQLLearningDB {
public:
    explicit PostgreSQLLearningDB(const std::string& connection_url);
    ~PostgreSQLLearningDB();

    bool initialize();
    void cleanup();

    // Pattern storage
    bool store_learning_pattern(const AdaptiveLearningSystem::LearningPattern& pattern);
    std::vector<AdaptiveLearningSystem::LearningPattern> load_patterns_by_type(
        AdaptiveLearningSystem::PatternType type);

    // Performance tracking
    bool track_model_performance(const std::string& model_name,
                               double accuracy,
                               const std::chrono::steady_clock::time_point& timestamp);

    // Vector similarity search (using pgvector)
    std::vector<AdaptiveLearningSystem::LearningPattern> find_similar_patterns(
        const std::vector<double>& feature_vector,
        double similarity_threshold = 0.8,
        size_t max_results = 10);

    // UK English specific queries
    std::vector<std::string> get_uk_vocabulary_suggestions(const std::string& text);
    bool store_uk_training_example(const std::string& audio_features,
                                 const std::string& correct_text);

private:
    std::string connection_url_;
    void* connection_; // PQconn* (PostgreSQL connection)
    bool connected_ = false;

    // Schema initialization
    bool create_tables();
    bool create_vector_extensions();

    // Helper methods
    std::string escape_sql_string(const std::string& input);
    std::vector<double> parse_vector_from_db(const std::string& vector_text);
};

// UK English specialization component
class UKEnglishSpecializer {
public:
    UKEnglishSpecializer();
    ~UKEnglishSpecializer();

    // Vocabulary and linguistic patterns
    void add_uk_vocabulary(const std::map<std::string, std::string>& us_to_uk_mapping);
    std::string convert_to_uk_spelling(const std::string& text);
    double calculate_uk_language_score(const std::string& text);

    // Pronunciation and acoustic modeling
    void add_uk_pronunciation_rules(const std::map<std::string, std::vector<std::string>>& rules);
    std::vector<std::string> generate_uk_pronunciation_variants(const std::string& word);

    // Training data generation
    Json::Value generate_uk_fine_tuning_data();
    std::vector<std::pair<std::string, std::string>> get_uk_training_pairs();

private:
    std::map<std::string, std::string> american_to_british_vocab_;
    std::map<std::string, std::vector<std::string>> uk_pronunciation_variants_;
    std::vector<std::string> uk_specific_phrases_;
    std::map<std::string, double> uk_linguistic_markers_;

    // Internal methods
    bool is_british_spelling(const std::string& word);
    bool contains_uk_specific_terms(const std::string& text);
    double calculate_lexical_uk_score(const std::string& text);
};

} // namespace vtt