#include "adaptive_learning_system.h"
#include "learning_api_client.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

namespace vtt {

AdaptiveLearningSystem::AdaptiveLearningSystem(const std::string& db_url)
    : learning_db_url_(db_url) {

    // Initialize with default values
    stats_.last_learning_update = std::chrono::steady_clock::now();
}

AdaptiveLearningSystem::~AdaptiveLearningSystem() {
    cleanup();
}

bool AdaptiveLearningSystem::initialize() {
    try {
        // Initialize PostgreSQL learning database
        learning_db_ = std::make_unique<PostgreSQLLearningDB>(learning_db_url_);
        if (!learning_db_->initialize()) {
            std::cerr << "Warning: Failed to initialize learning database, continuing without persistence" << std::endl;
            learning_db_.reset();
        } else {
            std::cout << "Learning database initialized successfully" << std::endl;

            // Load existing patterns from database
            load_patterns_from_db();
        }

        // Initialize UK English specializer
        uk_specializer_ = std::make_unique<UKEnglishSpecializer>();

        std::cout << "Adaptive learning system initialized" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing adaptive learning system: " << e.what() << std::endl;
        return false;
    }
}

void AdaptiveLearningSystem::cleanup() {
    stop_real_time_learning();

    // Save patterns to database before cleanup
    if (learning_db_) {
        save_patterns_to_db();
        learning_db_->cleanup();
        learning_db_.reset();
    }

    uk_specializer_.reset();

    std::cout << "Adaptive learning system cleaned up" << std::endl;
}

void AdaptiveLearningSystem::record_recognition(const RecognitionContext& context, const std::string& ground_truth) {
    std::lock_guard<std::mutex> history_lock(history_mutex_);

    // Add to recognition history
    recognition_history_.push_back(context);

    // Maintain maximum history size
    if (recognition_history_.size() > max_history_size_) {
        recognition_history_.pop_front();
    }

    // Submit to learning API if available and ground truth provided
    if (!ground_truth.empty()) {
        std::lock_guard<std::mutex> integration_lock(learning_integration_mutex_);
        if (learning_integration_) {
            learning_integration_->handle_training_correction(context.text, ground_truth);
        }
    }

    // Analyze patterns if we have enough data
    if (recognition_history_.size() >= min_pattern_usage_ * 2) {
        // Trigger pattern analysis in background
        std::thread([this]() {
            this->analyze_patterns();
        }).detach();
    }

    std::cout << "Recorded recognition context (history size: " << recognition_history_.size() << ")" << std::endl;
}

std::vector<AdaptiveLearningSystem::LearningInsight> AdaptiveLearningSystem::analyze_patterns() {
    std::vector<LearningInsight> insights;

    // Perform different types of pattern analysis
    analyze_acoustic_patterns();
    analyze_lexical_patterns();
    analyze_contextual_patterns();
    analyze_uk_dialect_patterns();

    // Generate insights based on learned patterns
    std::lock_guard<std::mutex> patterns_lock(patterns_mutex_);

    for (const auto& pattern_pair : learned_patterns_) {
        const LearningPattern& pattern = pattern_pair.second;

        if (pattern.confidence_score > pattern_confidence_threshold_) {
            LearningInsight insight;
            insight.insight_type = "pattern_discovered";
            insight.description = "Discovered effective pattern: " + pattern.pattern_id;
            insight.confidence = pattern.confidence_score;

            // Generate recommendations based on pattern type
            switch (pattern.type) {
                case PatternType::ACOUSTIC_PATTERN:
                    insight.recommended_adjustments["acoustic_weight"] = pattern.accuracy_improvement;
                    break;
                case PatternType::LEXICAL_PATTERN:
                    insight.recommended_adjustments["vocabulary_weight"] = pattern.accuracy_improvement;
                    break;
                case PatternType::UK_DIALECT_PATTERN:
                    insight.recommended_adjustments["uk_dialect_weight"] = pattern.accuracy_improvement + uk_pattern_bonus_;
                    break;
                default:
                    insight.recommended_adjustments["general_weight"] = pattern.accuracy_improvement;
                    break;
            }

            insights.push_back(insight);
        }
    }

    std::cout << "Generated " << insights.size() << " learning insights" << std::endl;
    return insights;
}

AdaptiveLearningSystem::LearningPattern AdaptiveLearningSystem::extract_pattern(const std::vector<RecognitionContext>& contexts) {
    LearningPattern pattern;

    if (contexts.empty()) {
        return pattern;
    }

    // Generate pattern ID based on context characteristics
    std::ostringstream pattern_id_stream;
    pattern_id_stream << "pattern_" << contexts.size() << "_" <<
                        std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
    pattern.pattern_id = pattern_id_stream.str();

    // Analyze common features across contexts
    std::map<std::string, std::vector<double>> feature_values;
    double total_confidence = 0.0;
    size_t uk_contexts = 0;

    for (const auto& context : contexts) {
        total_confidence += context.confidence;

        if (context.is_uk_english) {
            uk_contexts++;
        }

        // Extract and aggregate features
        for (size_t i = 0; i < context.acoustic_features.size(); ++i) {
            std::string feature_key = "acoustic_" + std::to_string(i);
            feature_values[feature_key].push_back(context.acoustic_features[i]);
        }

        // Domain and speaker features
        feature_values["domain_" + context.domain].push_back(1.0);
        if (!context.speaker_id.empty()) {
            feature_values["speaker_" + context.speaker_id].push_back(1.0);
        }
    }

    // Calculate average features
    for (const auto& feature_pair : feature_values) {
        if (!feature_pair.second.empty()) {
            double avg = std::accumulate(feature_pair.second.begin(), feature_pair.second.end(), 0.0) /
                        feature_pair.second.size();
            pattern.features[feature_pair.first] = avg;
        }
    }

    // Set pattern metadata
    pattern.confidence_score = total_confidence / contexts.size();
    pattern.usage_count = contexts.size();
    pattern.created_at = std::chrono::steady_clock::now();
    pattern.last_used = pattern.created_at;
    pattern.is_uk_specific = (static_cast<double>(uk_contexts) / contexts.size()) > 0.7;

    // Determine pattern type based on dominant features
    if (pattern.is_uk_specific) {
        pattern.type = PatternType::UK_DIALECT_PATTERN;
    } else if (pattern.features.count("acoustic_0") > 0) {
        pattern.type = PatternType::ACOUSTIC_PATTERN;
    } else {
        pattern.type = PatternType::LEXICAL_PATTERN;
    }

    return pattern;
}

std::map<std::string, double> AdaptiveLearningSystem::get_optimal_model_weights() {
    std::map<std::string, double> weights;

    std::lock_guard<std::mutex> lock(patterns_mutex_);

    // Calculate weights based on learned patterns
    for (const auto& pattern_pair : learned_patterns_) {
        const LearningPattern& pattern = pattern_pair.second;

        if (pattern.confidence_score > pattern_confidence_threshold_) {
            // Weight adjustment based on pattern accuracy improvement
            double weight_adjustment = pattern.accuracy_improvement * pattern.confidence_score;

            // Apply to relevant models based on pattern type
            std::string weight_key;
            switch (pattern.type) {
                case PatternType::ACOUSTIC_PATTERN:
                    weight_key = "acoustic_model";
                    break;
                case PatternType::LEXICAL_PATTERN:
                    weight_key = "lexical_model";
                    break;
                case PatternType::UK_DIALECT_PATTERN:
                    weight_key = "uk_model";
                    weight_adjustment += uk_pattern_bonus_;
                    break;
                default:
                    weight_key = "general_model";
                    break;
            }

            weights[weight_key] = std::max(0.1, std::min(2.0, 1.0 + weight_adjustment));
        }
    }

    // Default weights if no patterns found
    if (weights.empty()) {
        weights["general_model"] = 1.0;
        weights["acoustic_model"] = 1.0;
        weights["lexical_model"] = 1.0;
        weights["uk_model"] = 1.0;
    }

    return weights;
}

std::vector<std::string> AdaptiveLearningSystem::recommend_model_combinations() {
    std::vector<std::string> recommendations;

    auto optimal_weights = get_optimal_model_weights();

    // Sort models by weight
    std::vector<std::pair<std::string, double>> sorted_weights(optimal_weights.begin(), optimal_weights.end());
    std::sort(sorted_weights.begin(), sorted_weights.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Recommend top models
    for (size_t i = 0; i < std::min(size_t(3), sorted_weights.size()); ++i) {
        recommendations.push_back(sorted_weights[i].first);
    }

    return recommendations;
}

Json::Value AdaptiveLearningSystem::generate_training_recommendations() {
    Json::Value recommendations;

    auto insights = analyze_patterns();

    for (const auto& insight : insights) {
        Json::Value insight_json;
        insight_json["type"] = insight.insight_type;
        insight_json["description"] = insight.description;
        insight_json["confidence"] = insight.confidence;
        insight_json["requires_retraining"] = insight.requires_retraining;

        Json::Value adjustments;
        for (const auto& adj : insight.recommended_adjustments) {
            adjustments[adj.first] = adj.second;
        }
        insight_json["recommended_adjustments"] = adjustments;

        recommendations.append(insight_json);
    }

    return recommendations;
}

void AdaptiveLearningSystem::enable_uk_english_learning(bool enable) {
    if (enable && !uk_specializer_) {
        uk_specializer_ = std::make_unique<UKEnglishSpecializer>();
    } else if (!enable) {
        uk_specializer_.reset();
    }

    std::cout << "UK English learning " << (enable ? "enabled" : "disabled") << std::endl;
}

double AdaptiveLearningSystem::calculate_uk_dialect_confidence(const RecognitionContext& context) {
    if (!uk_specializer_) {
        return 0.0;
    }

    return uk_specializer_->calculate_uk_language_score(context.text);
}

void AdaptiveLearningSystem::start_real_time_learning() {
    std::cout << "Real-time learning started" << std::endl;
    stats_.last_learning_update = std::chrono::steady_clock::now();
}

void AdaptiveLearningSystem::stop_real_time_learning() {
    std::cout << "Real-time learning stopped" << std::endl;
}

void AdaptiveLearningSystem::update_learning_parameters(const Json::Value& config) {
    if (config.isMember("pattern_confidence_threshold")) {
        pattern_confidence_threshold_ = config["pattern_confidence_threshold"].asDouble();
    }

    if (config.isMember("uk_pattern_bonus")) {
        uk_pattern_bonus_ = config["uk_pattern_bonus"].asDouble();
    }

    if (config.isMember("max_history_size")) {
        max_history_size_ = config["max_history_size"].asUInt64();
    }

    std::cout << "Learning parameters updated" << std::endl;
}

void AdaptiveLearningSystem::set_learning_integration(std::shared_ptr<LearningSystemIntegration> integration) {
    std::lock_guard<std::mutex> lock(learning_integration_mutex_);
    learning_integration_ = integration;

    if (learning_integration_) {
        learning_integration_->integrate_with_adaptive_learning(this);
    }

    std::cout << "Learning integration set for adaptive learning system" << std::endl;
}

std::shared_ptr<LearningSystemIntegration> AdaptiveLearningSystem::get_learning_integration() const {
    std::lock_guard<std::mutex> lock(learning_integration_mutex_);
    return learning_integration_;
}

// Pattern analysis methods
void AdaptiveLearningSystem::analyze_acoustic_patterns() {
    std::lock_guard<std::mutex> history_lock(history_mutex_);

    if (recognition_history_.size() < min_pattern_usage_) {
        return;
    }

    // Group contexts by acoustic similarity
    std::vector<std::vector<RecognitionContext>> acoustic_groups;

    for (const auto& context : recognition_history_) {
        bool found_group = false;

        for (auto& group : acoustic_groups) {
            if (!group.empty()) {
                // Simple acoustic similarity check (can be enhanced)
                double similarity = 0.0;
                const auto& ref_features = group[0].acoustic_features;

                if (ref_features.size() == context.acoustic_features.size()) {
                    for (size_t i = 0; i < ref_features.size(); ++i) {
                        similarity += std::abs(ref_features[i] - context.acoustic_features[i]);
                    }
                    similarity /= ref_features.size();
                }

                if (similarity < 0.1) { // Similar threshold
                    group.push_back(context);
                    found_group = true;
                    break;
                }
            }
        }

        if (!found_group) {
            acoustic_groups.push_back({context});
        }
    }

    // Extract patterns from large groups
    for (const auto& group : acoustic_groups) {
        if (group.size() >= min_pattern_usage_) {
            LearningPattern pattern = extract_pattern(group);
            pattern.type = PatternType::ACOUSTIC_PATTERN;
            pattern.accuracy_improvement = 0.05; // Estimated improvement

            std::lock_guard<std::mutex> patterns_lock(patterns_mutex_);
            learned_patterns_[pattern.pattern_id] = pattern;
            stats_.patterns_learned++;
        }
    }
}

void AdaptiveLearningSystem::analyze_lexical_patterns() {
    std::lock_guard<std::mutex> history_lock(history_mutex_);

    // Analyze word frequency and context patterns
    std::map<std::string, std::vector<RecognitionContext>> word_contexts;

    for (const auto& context : recognition_history_) {
        std::istringstream iss(context.text);
        std::string word;

        while (iss >> word) {
            // Clean word (remove punctuation)
            word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);

            if (!word.empty()) {
                word_contexts[word].push_back(context);
            }
        }
    }

    // Find patterns for frequently used words
    for (const auto& word_pair : word_contexts) {
        if (word_pair.second.size() >= min_pattern_usage_) {
            LearningPattern pattern = extract_pattern(word_pair.second);
            pattern.type = PatternType::LEXICAL_PATTERN;
            pattern.pattern_id = "lexical_" + word_pair.first;
            pattern.accuracy_improvement = 0.03;

            std::lock_guard<std::mutex> patterns_lock(patterns_mutex_);
            learned_patterns_[pattern.pattern_id] = pattern;
            stats_.patterns_learned++;
        }
    }
}

void AdaptiveLearningSystem::analyze_contextual_patterns() {
    std::lock_guard<std::mutex> history_lock(history_mutex_);

    // Group by domain and speaker
    std::map<std::string, std::vector<RecognitionContext>> domain_contexts;
    std::map<std::string, std::vector<RecognitionContext>> speaker_contexts;

    for (const auto& context : recognition_history_) {
        domain_contexts[context.domain].push_back(context);
        if (!context.speaker_id.empty()) {
            speaker_contexts[context.speaker_id].push_back(context);
        }
    }

    // Extract domain patterns
    for (const auto& domain_pair : domain_contexts) {
        if (domain_pair.second.size() >= min_pattern_usage_) {
            LearningPattern pattern = extract_pattern(domain_pair.second);
            pattern.type = PatternType::CONTEXTUAL_PATTERN;
            pattern.pattern_id = "domain_" + domain_pair.first;
            pattern.accuracy_improvement = 0.04;

            std::lock_guard<std::mutex> patterns_lock(patterns_mutex_);
            learned_patterns_[pattern.pattern_id] = pattern;
            stats_.patterns_learned++;
        }
    }

    // Extract speaker patterns
    for (const auto& speaker_pair : speaker_contexts) {
        if (speaker_pair.second.size() >= min_pattern_usage_) {
            LearningPattern pattern = extract_pattern(speaker_pair.second);
            pattern.type = PatternType::SPEAKER_PATTERN;
            pattern.pattern_id = "speaker_" + speaker_pair.first;
            pattern.accuracy_improvement = 0.06;

            std::lock_guard<std::mutex> patterns_lock(patterns_mutex_);
            learned_patterns_[pattern.pattern_id] = pattern;
            stats_.patterns_learned++;
        }
    }
}

void AdaptiveLearningSystem::analyze_uk_dialect_patterns() {
    if (!uk_specializer_) {
        return;
    }

    std::lock_guard<std::mutex> history_lock(history_mutex_);

    std::vector<RecognitionContext> uk_contexts;

    for (const auto& context : recognition_history_) {
        if (context.is_uk_english || uk_specializer_->calculate_uk_language_score(context.text) > 0.7) {
            uk_contexts.push_back(context);
        }
    }

    if (uk_contexts.size() >= min_pattern_usage_) {
        LearningPattern pattern = extract_pattern(uk_contexts);
        pattern.type = PatternType::UK_DIALECT_PATTERN;
        pattern.pattern_id = "uk_dialect_pattern";
        pattern.accuracy_improvement = 0.08 + uk_pattern_bonus_;
        pattern.is_uk_specific = true;

        std::lock_guard<std::mutex> patterns_lock(patterns_mutex_);
        learned_patterns_[pattern.pattern_id] = pattern;
        stats_.patterns_learned++;
        stats_.uk_patterns_learned++;
    }
}

bool AdaptiveLearningSystem::save_patterns_to_db() {
    if (!learning_db_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(patterns_mutex_);

    for (const auto& pattern_pair : learned_patterns_) {
        learning_db_->store_learning_pattern(pattern_pair.second);
    }

    std::cout << "Saved " << learned_patterns_.size() << " patterns to database" << std::endl;
    return true;
}

bool AdaptiveLearningSystem::load_patterns_from_db() {
    if (!learning_db_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(patterns_mutex_);

    // Load patterns of all types
    for (int type = 0; type <= static_cast<int>(PatternType::UK_DIALECT_PATTERN); ++type) {
        auto patterns = learning_db_->load_patterns_by_type(static_cast<PatternType>(type));

        for (const auto& pattern : patterns) {
            learned_patterns_[pattern.pattern_id] = pattern;
        }
    }

    std::cout << "Loaded " << learned_patterns_.size() << " patterns from database" << std::endl;
    return true;
}

// PostgreSQLLearningDB implementation
PostgreSQLLearningDB::PostgreSQLLearningDB(const std::string& connection_url)
    : connection_url_(connection_url), connection_(nullptr) {
}

PostgreSQLLearningDB::~PostgreSQLLearningDB() {
    cleanup();
}

bool PostgreSQLLearningDB::initialize() {
    // This is a placeholder implementation
    // In a real implementation, you would use libpq to connect to PostgreSQL
    std::cout << "PostgreSQL learning DB initialized (placeholder)" << std::endl;
    connected_ = true;
    return true;
}

void PostgreSQLLearningDB::cleanup() {
    if (connected_) {
        // Close PostgreSQL connection
        connected_ = false;
        std::cout << "PostgreSQL learning DB cleaned up" << std::endl;
    }
}

bool PostgreSQLLearningDB::store_learning_pattern(const AdaptiveLearningSystem::LearningPattern& pattern) {
    if (!connected_) {
        return false;
    }

    // Placeholder implementation
    std::cout << "Stored pattern: " << pattern.pattern_id << std::endl;
    return true;
}

std::vector<AdaptiveLearningSystem::LearningPattern> PostgreSQLLearningDB::load_patterns_by_type(AdaptiveLearningSystem::PatternType type) {
    std::vector<AdaptiveLearningSystem::LearningPattern> patterns;

    if (!connected_) {
        return patterns;
    }

    // Placeholder implementation
    std::cout << "Loaded patterns of type: " << static_cast<int>(type) << std::endl;
    return patterns;
}

// UKEnglishSpecializer implementation
UKEnglishSpecializer::UKEnglishSpecializer() {
    // Initialize UK vocabulary mappings
    american_to_british_vocab_["color"] = "colour";
    american_to_british_vocab_["center"] = "centre";
    american_to_british_vocab_["theater"] = "theatre";
    american_to_british_vocab_["organize"] = "organise";
    american_to_british_vocab_["realize"] = "realise";
    american_to_british_vocab_["defense"] = "defence";
    american_to_british_vocab_["license"] = "licence";

    // UK-specific phrases
    uk_specific_phrases_.push_back("lift");
    uk_specific_phrases_.push_back("lorry");
    uk_specific_phrases_.push_back("biscuit");
    uk_specific_phrases_.push_back("jumper");
    uk_specific_phrases_.push_back("queue");
    uk_specific_phrases_.push_back("brilliant");
    uk_specific_phrases_.push_back("cheerio");
    uk_specific_phrases_.push_back("bloke");
    uk_specific_phrases_.push_back("bloody");
    uk_specific_phrases_.push_back("rubbish");
}

UKEnglishSpecializer::~UKEnglishSpecializer() {
}

double UKEnglishSpecializer::calculate_uk_language_score(const std::string& text) {
    double score = 0.0;
    size_t word_count = 0;

    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        word_count++;

        // Remove punctuation and convert to lowercase
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        // Check for British spellings
        if (is_british_spelling(word)) {
            score += 2.0;
        }

        // Check for UK-specific terms
        if (std::find(uk_specific_phrases_.begin(), uk_specific_phrases_.end(), word) != uk_specific_phrases_.end()) {
            score += 3.0;
        }
    }

    // Normalize by word count
    if (word_count > 0) {
        score /= word_count;
    }

    return std::min(1.0, score / 2.0); // Normalize to 0-1 range
}

std::string UKEnglishSpecializer::convert_to_uk_spelling(const std::string& text) {
    std::string uk_text = text;

    for (const auto& mapping : american_to_british_vocab_) {
        size_t pos = 0;
        while ((pos = uk_text.find(mapping.first, pos)) != std::string::npos) {
            uk_text.replace(pos, mapping.first.length(), mapping.second);
            pos += mapping.second.length();
        }
    }

    return uk_text;
}

bool UKEnglishSpecializer::is_british_spelling(const std::string& word) {
    // Check if word is a British spelling
    for (const auto& mapping : american_to_british_vocab_) {
        if (word == mapping.second) {
            return true;
        }
    }

    // Check for common British spelling patterns
    if (word.length() > 4) {
        if (word.substr(word.length() - 4) == "ised" ||
            word.substr(word.length() - 4) == "tion" ||
            word.substr(word.length() - 3) == "our" ||
            word.substr(word.length() - 2) == "re") {
            return true;
        }
    }

    return false;
}

} // namespace vtt