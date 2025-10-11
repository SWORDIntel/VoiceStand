#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <queue>
#include <memory>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <json/json.h>

namespace vtt {

// Auto-correction learning system that adapts to user corrections
class AutoCorrection {
public:
    // Correction entry
    struct Correction {
        std::string original;
        std::string corrected;
        size_t frequency = 1;
        float confidence = 0.0f;
        std::chrono::steady_clock::time_point last_used;
        
        Correction(const std::string& orig, const std::string& corr)
            : original(orig), corrected(corr),
              last_used(std::chrono::steady_clock::now()) {}
    };
    
    // N-gram context for better prediction
    struct NGramContext {
        std::vector<std::string> preceding_words;
        std::vector<std::string> following_words;
        size_t n = 3;  // Trigram by default
        
        std::string to_string() const {
            std::string result;
            for (const auto& word : preceding_words) {
                result += word + " ";
            }
            result += "[WORD] ";
            for (const auto& word : following_words) {
                result += word + " ";
            }
            return result;
        }
    };
    
    // Correction suggestion
    struct Suggestion {
        std::string text;
        float confidence;
        std::string reason;
        
        Suggestion(const std::string& t, float c, const std::string& r = "")
            : text(t), confidence(c), reason(r) {}
    };
    
    AutoCorrection();
    ~AutoCorrection();
    
    // Learn from user correction
    void learn_correction(const std::string& original, 
                         const std::string& corrected,
                         const NGramContext& context = NGramContext());
    
    // Apply corrections to text
    std::string apply_corrections(const std::string& text);
    
    // Get suggestions for a word
    std::vector<Suggestion> get_suggestions(const std::string& word,
                                           const NGramContext& context = NGramContext());
    
    // Save/load learned corrections
    bool save_model(const std::string& filepath);
    bool load_model(const std::string& filepath);
    
    // Enable/disable auto-correction
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }
    
    // Set minimum confidence for auto-correction
    void set_min_confidence(float confidence) { min_confidence_ = confidence; }
    
    // Clear learned corrections
    void clear_corrections();
    
    // Get statistics
    struct Stats {
        size_t total_corrections = 0;
        size_t unique_corrections = 0;
        size_t corrections_applied = 0;
        float average_confidence = 0.0f;
        
        Json::Value to_json() const {
            Json::Value stats;
            stats["total_corrections"] = static_cast<Json::Int64>(total_corrections);
            stats["unique_corrections"] = static_cast<Json::Int64>(unique_corrections);
            stats["corrections_applied"] = static_cast<Json::Int64>(corrections_applied);
            stats["average_confidence"] = average_confidence;
            return stats;
        }
    };
    
    Stats get_stats() const { return stats_; }
    
private:
    // Correction database
    std::unordered_map<std::string, std::vector<std::shared_ptr<Correction>>> corrections_;
    
    // Context-aware corrections
    std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Correction>>> 
        context_corrections_;
    
    // Common misspellings and their corrections
    void load_common_corrections();
    
    // Calculate edit distance between strings
    int levenshtein_distance(const std::string& s1, const std::string& s2);
    
    // Find similar words using fuzzy matching
    std::vector<std::string> find_similar_words(const std::string& word, int max_distance = 2);
    
    // Calculate confidence score for correction
    float calculate_confidence(const std::shared_ptr<Correction>& correction,
                              const NGramContext& context);
    
    // Apply single correction
    std::string apply_single_correction(const std::string& word,
                                       const NGramContext& context);
    
    // Extract n-gram context from text
    NGramContext extract_context(const std::vector<std::string>& words, size_t word_index);
    
    // Tokenize text into words
    std::vector<std::string> tokenize(const std::string& text);
    
    // Join words back into text
    std::string join_words(const std::vector<std::string>& words);
    
    bool enabled_ = true;
    float min_confidence_ = 0.7f;
    Stats stats_;
    
    // Learning parameters
    float learning_rate_ = 0.1f;
    size_t max_corrections_per_word_ = 10;
    
    // Common corrections database
    std::unordered_map<std::string, std::string> common_corrections_;
};

// Implementation
inline AutoCorrection::AutoCorrection() {
    load_common_corrections();
}

inline AutoCorrection::~AutoCorrection() {
    // Auto-save on destruction
    save_model("auto_corrections.json");
}

inline void AutoCorrection::learn_correction(const std::string& original,
                                            const std::string& corrected,
                                            const NGramContext& context) {
    if (original == corrected) return;
    
    // Store basic correction
    auto& correction_list = corrections_[original];
    
    // Check if this correction already exists
    auto it = std::find_if(correction_list.begin(), correction_list.end(),
        [&corrected](const auto& c) { return c->corrected == corrected; });
    
    if (it != correction_list.end()) {
        // Update existing correction
        (*it)->frequency++;
        (*it)->last_used = std::chrono::steady_clock::now();
    } else {
        // Add new correction
        auto correction = std::make_shared<Correction>(original, corrected);
        correction_list.push_back(correction);
        
        // Limit corrections per word
        if (correction_list.size() > max_corrections_per_word_) {
            // Remove least frequent
            std::sort(correction_list.begin(), correction_list.end(),
                [](const auto& a, const auto& b) { return a->frequency > b->frequency; });
            correction_list.resize(max_corrections_per_word_);
        }
    }
    
    // Store context-aware correction if context provided
    if (!context.preceding_words.empty() || !context.following_words.empty()) {
        std::string context_key = context.to_string();
        context_corrections_[original][context_key] = 
            std::make_shared<Correction>(original, corrected);
    }
    
    // Update statistics
    stats_.total_corrections++;
    stats_.unique_corrections = corrections_.size();
}

inline std::string AutoCorrection::apply_corrections(const std::string& text) {
    if (!enabled_) return text;
    
    auto words = tokenize(text);
    
    for (size_t i = 0; i < words.size(); ++i) {
        auto context = extract_context(words, i);
        words[i] = apply_single_correction(words[i], context);
    }
    
    return join_words(words);
}

inline std::string AutoCorrection::apply_single_correction(const std::string& word,
                                                          const NGramContext& context) {
    // Check for exact match corrections
    auto it = corrections_.find(word);
    if (it != corrections_.end() && !it->second.empty()) {
        // Find best correction based on frequency and context
        std::shared_ptr<Correction> best_correction = nullptr;
        float best_confidence = 0.0f;
        
        for (const auto& correction : it->second) {
            float confidence = calculate_confidence(correction, context);
            if (confidence > best_confidence && confidence >= min_confidence_) {
                best_confidence = confidence;
                best_correction = correction;
            }
        }
        
        if (best_correction) {
            stats_.corrections_applied++;
            return best_correction->corrected;
        }
    }
    
    // Check common corrections
    auto common_it = common_corrections_.find(word);
    if (common_it != common_corrections_.end()) {
        return common_it->second;
    }
    
    return word;
}

inline float AutoCorrection::calculate_confidence(const std::shared_ptr<Correction>& correction,
                                                 const NGramContext& context) {
    float confidence = 0.0f;
    
    // Base confidence from frequency
    confidence = std::min(1.0f, correction->frequency / 10.0f) * 0.5f;
    
    // Boost confidence if used recently
    auto now = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::hours>(
        now - correction->last_used).count();
    if (time_diff < 24) {
        confidence += 0.2f;
    }
    
    // Context matching bonus
    if (!context.preceding_words.empty() || !context.following_words.empty()) {
        std::string context_key = context.to_string();
        auto ctx_it = context_corrections_.find(correction->original);
        if (ctx_it != context_corrections_.end()) {
            auto ctx_corr_it = ctx_it->second.find(context_key);
            if (ctx_corr_it != ctx_it->second.end() && 
                ctx_corr_it->second->corrected == correction->corrected) {
                confidence += 0.3f;
            }
        }
    }
    
    return std::min(1.0f, confidence);
}

inline std::vector<AutoCorrection::Suggestion> AutoCorrection::get_suggestions(
    const std::string& word, const NGramContext& context) {
    
    std::vector<Suggestion> suggestions;
    
    // Get learned corrections
    auto it = corrections_.find(word);
    if (it != corrections_.end()) {
        for (const auto& correction : it->second) {
            float confidence = calculate_confidence(correction, context);
            suggestions.emplace_back(correction->corrected, confidence, 
                                    "Learned from previous corrections");
        }
    }
    
    // Get common corrections
    auto common_it = common_corrections_.find(word);
    if (common_it != common_corrections_.end()) {
        suggestions.emplace_back(common_it->second, 0.8f, "Common correction");
    }
    
    // Find similar words
    auto similar = find_similar_words(word, 2);
    for (const auto& sim_word : similar) {
        suggestions.emplace_back(sim_word, 0.5f, "Similar word");
    }
    
    // Sort by confidence
    std::sort(suggestions.begin(), suggestions.end(),
        [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
    
    return suggestions;
}

inline void AutoCorrection::load_common_corrections() {
    // Load common English corrections
    common_corrections_ = {
        {"teh", "the"},
        {"recieve", "receive"},
        {"occured", "occurred"},
        {"untill", "until"},
        {"wich", "which"},
        {"alot", "a lot"},
        {"definately", "definitely"},
        {"seperate", "separate"},
        {"occassion", "occasion"},
        {"aquire", "acquire"},
        {"wierd", "weird"},
        {"therefor", "therefore"},
        {"goverment", "government"},
        {"accomodate", "accommodate"},
        {"acheive", "achieve"},
        {"appearence", "appearance"},
        {"begining", "beginning"},
        {"beleive", "believe"},
        {"calender", "calendar"},
        {"concious", "conscious"}
    };
}

inline int AutoCorrection::levenshtein_distance(const std::string& s1, const std::string& s2) {
    size_t len1 = s1.length();
    size_t len2 = s2.length();
    
    std::vector<std::vector<int>> dp(len1 + 1, std::vector<int>(len2 + 1));
    
    for (size_t i = 0; i <= len1; ++i) dp[i][0] = i;
    for (size_t j = 0; j <= len2; ++j) dp[0][j] = j;
    
    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            int cost = (s1[i-1] == s2[j-1]) ? 0 : 1;
            dp[i][j] = std::min({
                dp[i-1][j] + 1,      // deletion
                dp[i][j-1] + 1,      // insertion
                dp[i-1][j-1] + cost  // substitution
            });
        }
    }
    
    return dp[len1][len2];
}

inline std::vector<std::string> AutoCorrection::find_similar_words(const std::string& word, 
                                                                  int max_distance) {
    std::vector<std::string> similar;
    
    // Check against known corrections
    for (const auto& [orig, corrections_list] : corrections_) {
        if (levenshtein_distance(word, orig) <= max_distance) {
            for (const auto& correction : corrections_list) {
                similar.push_back(correction->corrected);
            }
        }
    }
    
    // Check common words
    for (const auto& [common_wrong, common_right] : common_corrections_) {
        if (levenshtein_distance(word, common_wrong) <= max_distance) {
            similar.push_back(common_right);
        }
    }
    
    return similar;
}

inline AutoCorrection::NGramContext AutoCorrection::extract_context(
    const std::vector<std::string>& words, size_t word_index) {
    
    NGramContext context;
    context.n = 3;  // Trigram
    
    // Get preceding words
    for (size_t i = 1; i <= context.n && word_index >= i; ++i) {
        context.preceding_words.insert(context.preceding_words.begin(), 
                                      words[word_index - i]);
    }
    
    // Get following words
    for (size_t i = 1; i <= context.n && word_index + i < words.size(); ++i) {
        context.following_words.push_back(words[word_index + i]);
    }
    
    return context;
}

inline std::vector<std::string> AutoCorrection::tokenize(const std::string& text) {
    std::vector<std::string> words;
    std::stringstream ss(text);
    std::string word;
    
    while (ss >> word) {
        words.push_back(word);
    }
    
    return words;
}

inline std::string AutoCorrection::join_words(const std::vector<std::string>& words) {
    std::string result;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) result += " ";
        result += words[i];
    }
    return result;
}

inline bool AutoCorrection::save_model(const std::string& filepath) {
    Json::Value root;
    Json::Value corrections_json;
    
    for (const auto& [word, correction_list] : corrections_) {
        Json::Value word_corrections;
        for (const auto& correction : correction_list) {
            Json::Value corr;
            corr["corrected"] = correction->corrected;
            corr["frequency"] = static_cast<Json::Int64>(correction->frequency);
            word_corrections.append(corr);
        }
        corrections_json[word] = word_corrections;
    }
    
    root["corrections"] = corrections_json;
    root["stats"] = stats_.to_json();
    
    Json::StreamWriterBuilder builder;
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    
    std::ofstream file(filepath);
    if (!file.is_open()) return false;
    
    writer->write(root, &file);
    return true;
}

inline bool AutoCorrection::load_model(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    
    Json::Value root;
    Json::Reader reader;
    
    if (!reader.parse(file, root)) return false;
    
    corrections_.clear();
    
    const Json::Value& corrections_json = root["corrections"];
    for (const auto& word : corrections_json.getMemberNames()) {
        const Json::Value& word_corrections = corrections_json[word];
        
        for (const auto& corr : word_corrections) {
            auto correction = std::make_shared<Correction>(
                word, corr["corrected"].asString());
            correction->frequency = corr["frequency"].asInt64();
            corrections_[word].push_back(correction);
        }
    }
    
    // Load stats
    const Json::Value& stats_json = root["stats"];
    stats_.total_corrections = stats_json["total_corrections"].asInt64();
    stats_.unique_corrections = stats_json["unique_corrections"].asInt64();
    stats_.corrections_applied = stats_json["corrections_applied"].asInt64();
    stats_.average_confidence = stats_json["average_confidence"].asFloat();
    
    return true;
}

}  // namespace vtt