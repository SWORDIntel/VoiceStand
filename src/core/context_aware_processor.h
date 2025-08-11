#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <deque>
#include <chrono>
#include <algorithm>
#include <json/json.h>

namespace vtt {

// Context-aware processing for improved transcription accuracy
class ContextAwareProcessor {
public:
    // Context types
    enum class ContextType {
        GENERAL,
        TECHNICAL,
        MEDICAL,
        LEGAL,
        BUSINESS,
        ACADEMIC,
        CASUAL,
        CUSTOM
    };
    
    // Domain-specific vocabulary
    struct DomainVocabulary {
        std::string name;
        std::unordered_set<std::string> terms;
        std::unordered_map<std::string, std::string> abbreviations;
        std::unordered_map<std::string, float> term_weights;
        float confidence_boost = 0.1f;
        
        DomainVocabulary(const std::string& n) : name(n) {}
    };
    
    // Conversation context
    struct ConversationContext {
        std::deque<std::string> recent_sentences;
        std::unordered_map<std::string, size_t> topic_frequencies;
        std::vector<std::string> detected_entities;
        ContextType current_type = ContextType::GENERAL;
        std::chrono::steady_clock::time_point last_update;
        size_t turn_count = 0;
        
        void add_sentence(const std::string& sentence) {
            recent_sentences.push_back(sentence);
            if (recent_sentences.size() > 10) {
                recent_sentences.pop_front();
            }
            last_update = std::chrono::steady_clock::now();
            turn_count++;
        }
        
        std::string get_context_summary() const {
            std::string summary;
            for (const auto& sentence : recent_sentences) {
                summary += sentence + " ";
            }
            return summary;
        }
    };
    
    // Context hint for processing
    struct ContextHint {
        ContextType type;
        std::vector<std::string> keywords;
        std::vector<std::string> expected_terms;
        float confidence_threshold = 0.7f;
        
        ContextHint(ContextType t) : type(t) {}
    };
    
    ContextAwareProcessor();
    ~ContextAwareProcessor() = default;
    
    // Process text with context awareness
    std::string process_with_context(const std::string& text);
    
    // Update conversation context
    void update_context(const std::string& text);
    
    // Set current context type
    void set_context_type(ContextType type);
    
    // Auto-detect context from text
    ContextType detect_context(const std::string& text);
    
    // Add custom domain vocabulary
    void add_domain_vocabulary(const std::string& domain_name,
                               const std::vector<std::string>& terms,
                               const std::unordered_map<std::string, std::string>& abbreviations = {});
    
    // Get context hints for better recognition
    ContextHint get_context_hint() const;
    
    // Apply context-based corrections
    std::string apply_context_corrections(const std::string& text);
    
    // Expand abbreviations based on context
    std::string expand_abbreviations(const std::string& text);
    
    // Get suggested next words based on context
    std::vector<std::string> predict_next_words(const std::string& partial_sentence);
    
    // Save/load context model
    bool save_context_model(const std::string& filepath);
    bool load_context_model(const std::string& filepath);
    
    // Reset context
    void reset_context();
    
    // Get context statistics
    struct Stats {
        size_t total_processed = 0;
        size_t context_switches = 0;
        std::unordered_map<std::string, size_t> domain_usage;
        float average_confidence = 0.0f;
        
        Json::Value to_json() const {
            Json::Value stats;
            stats["total_processed"] = static_cast<Json::Int64>(total_processed);
            stats["context_switches"] = static_cast<Json::Int64>(context_switches);
            
            Json::Value domains;
            for (const auto& [domain, count] : domain_usage) {
                domains[domain] = static_cast<Json::Int64>(count);
            }
            stats["domain_usage"] = domains;
            stats["average_confidence"] = average_confidence;
            
            return stats;
        }
    };
    
    Stats get_stats() const { return stats_; }
    
private:
    // Initialize domain vocabularies
    void initialize_domains();
    
    // Detect technical context
    bool is_technical_context(const std::string& text);
    
    // Detect medical context
    bool is_medical_context(const std::string& text);
    
    // Detect legal context
    bool is_legal_context(const std::string& text);
    
    // Detect business context
    bool is_business_context(const std::string& text);
    
    // Extract entities from text
    std::vector<std::string> extract_entities(const std::string& text);
    
    // Calculate context confidence
    float calculate_context_confidence(const std::string& text, ContextType type);
    
    // Apply domain-specific corrections
    std::string apply_domain_corrections(const std::string& text, 
                                        const DomainVocabulary& domain);
    
    // N-gram based prediction
    std::vector<std::string> ngram_predict(const std::vector<std::string>& words, size_t n = 3);
    
    ConversationContext conversation_context_;
    std::unordered_map<ContextType, std::shared_ptr<DomainVocabulary>> domains_;
    ContextType current_context_ = ContextType::GENERAL;
    
    // N-gram model for prediction
    std::unordered_map<std::string, std::unordered_map<std::string, size_t>> bigrams_;
    std::unordered_map<std::string, std::unordered_map<std::string, size_t>> trigrams_;
    
    Stats stats_;
    
    // Context detection thresholds
    const float CONTEXT_DETECTION_THRESHOLD = 0.3f;
    const size_t MIN_TERMS_FOR_DETECTION = 3;
};

// Implementation
inline ContextAwareProcessor::ContextAwareProcessor() {
    initialize_domains();
}

inline void ContextAwareProcessor::initialize_domains() {
    // Technical domain
    auto tech = std::make_shared<DomainVocabulary>("technical");
    tech->terms = {
        "algorithm", "database", "server", "client", "API", "framework",
        "deployment", "repository", "commit", "branch", "merge", "pipeline",
        "container", "kubernetes", "docker", "microservice", "endpoint",
        "authentication", "authorization", "encryption", "hash", "token"
    };
    tech->abbreviations = {
        {"API", "Application Programming Interface"},
        {"CPU", "Central Processing Unit"},
        {"RAM", "Random Access Memory"},
        {"OS", "Operating System"},
        {"UI", "User Interface"},
        {"UX", "User Experience"},
        {"DB", "Database"},
        {"SQL", "Structured Query Language"}
    };
    domains_[ContextType::TECHNICAL] = tech;
    
    // Medical domain
    auto medical = std::make_shared<DomainVocabulary>("medical");
    medical->terms = {
        "patient", "diagnosis", "treatment", "symptom", "medication",
        "prescription", "dosage", "chronic", "acute", "syndrome",
        "pathology", "etiology", "prognosis", "therapy", "clinical"
    };
    medical->abbreviations = {
        {"BP", "Blood Pressure"},
        {"HR", "Heart Rate"},
        {"ECG", "Electrocardiogram"},
        {"MRI", "Magnetic Resonance Imaging"},
        {"CT", "Computed Tomography"},
        {"IV", "Intravenous"},
        {"ER", "Emergency Room"}
    };
    domains_[ContextType::MEDICAL] = medical;
    
    // Legal domain
    auto legal = std::make_shared<DomainVocabulary>("legal");
    legal->terms = {
        "plaintiff", "defendant", "litigation", "jurisdiction", "statute",
        "precedent", "testimony", "evidence", "counsel", "motion",
        "verdict", "settlement", "contract", "liability", "negligence"
    };
    legal->abbreviations = {
        {"LLC", "Limited Liability Company"},
        {"IP", "Intellectual Property"},
        {"NDA", "Non-Disclosure Agreement"},
        {"POA", "Power of Attorney"}
    };
    domains_[ContextType::LEGAL] = legal;
    
    // Business domain
    auto business = std::make_shared<DomainVocabulary>("business");
    business->terms = {
        "revenue", "profit", "margin", "quarterly", "stakeholder",
        "strategy", "marketing", "acquisition", "merger", "portfolio",
        "benchmark", "KPI", "ROI", "synergy", "leverage"
    };
    business->abbreviations = {
        {"CEO", "Chief Executive Officer"},
        {"CFO", "Chief Financial Officer"},
        {"ROI", "Return on Investment"},
        {"KPI", "Key Performance Indicator"},
        {"B2B", "Business to Business"},
        {"B2C", "Business to Consumer"}
    };
    domains_[ContextType::BUSINESS] = business;
}

inline std::string ContextAwareProcessor::process_with_context(const std::string& text) {
    stats_.total_processed++;
    
    // Update conversation context
    update_context(text);
    
    // Detect context if needed
    auto detected_context = detect_context(text);
    if (detected_context != current_context_) {
        current_context_ = detected_context;
        stats_.context_switches++;
    }
    
    // Apply context-based processing
    std::string processed = text;
    processed = apply_context_corrections(processed);
    processed = expand_abbreviations(processed);
    
    // Update domain usage stats
    if (domains_.find(current_context_) != domains_.end()) {
        stats_.domain_usage[domains_[current_context_]->name]++;
    }
    
    return processed;
}

inline void ContextAwareProcessor::update_context(const std::string& text) {
    conversation_context_.add_sentence(text);
    
    // Extract and store entities
    auto entities = extract_entities(text);
    for (const auto& entity : entities) {
        conversation_context_.detected_entities.push_back(entity);
    }
    
    // Update topic frequencies
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        // Convert to lowercase for frequency counting
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        conversation_context_.topic_frequencies[word]++;
    }
    
    // Update n-gram models for prediction
    std::vector<std::string> words;
    std::stringstream ws(text);
    while (ws >> word) {
        words.push_back(word);
    }
    
    // Build bigrams
    for (size_t i = 0; i < words.size() - 1; ++i) {
        bigrams_[words[i]][words[i + 1]]++;
    }
    
    // Build trigrams
    for (size_t i = 0; i < words.size() - 2; ++i) {
        std::string trigram_key = words[i] + " " + words[i + 1];
        trigrams_[trigram_key][words[i + 2]]++;
    }
}

inline ContextAwareProcessor::ContextType ContextAwareProcessor::detect_context(
    const std::string& text) {
    
    struct ContextScore {
        ContextType type;
        float score;
    };
    
    std::vector<ContextScore> scores;
    
    // Calculate scores for each context type
    if (is_technical_context(text)) {
        scores.push_back({ContextType::TECHNICAL, 
                         calculate_context_confidence(text, ContextType::TECHNICAL)});
    }
    
    if (is_medical_context(text)) {
        scores.push_back({ContextType::MEDICAL,
                         calculate_context_confidence(text, ContextType::MEDICAL)});
    }
    
    if (is_legal_context(text)) {
        scores.push_back({ContextType::LEGAL,
                         calculate_context_confidence(text, ContextType::LEGAL)});
    }
    
    if (is_business_context(text)) {
        scores.push_back({ContextType::BUSINESS,
                         calculate_context_confidence(text, ContextType::BUSINESS)});
    }
    
    // Find highest scoring context
    if (!scores.empty()) {
        auto max_score = std::max_element(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.score < b.score; });
        
        if (max_score->score >= CONTEXT_DETECTION_THRESHOLD) {
            return max_score->type;
        }
    }
    
    return ContextType::GENERAL;
}

inline bool ContextAwareProcessor::is_technical_context(const std::string& text) {
    if (domains_.find(ContextType::TECHNICAL) == domains_.end()) {
        return false;
    }
    
    const auto& tech_domain = domains_[ContextType::TECHNICAL];
    size_t term_count = 0;
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& term : tech_domain->terms) {
        if (lower_text.find(term) != std::string::npos) {
            term_count++;
        }
    }
    
    return term_count >= MIN_TERMS_FOR_DETECTION;
}

inline bool ContextAwareProcessor::is_medical_context(const std::string& text) {
    if (domains_.find(ContextType::MEDICAL) == domains_.end()) {
        return false;
    }
    
    const auto& medical_domain = domains_[ContextType::MEDICAL];
    size_t term_count = 0;
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& term : medical_domain->terms) {
        if (lower_text.find(term) != std::string::npos) {
            term_count++;
        }
    }
    
    return term_count >= MIN_TERMS_FOR_DETECTION;
}

inline bool ContextAwareProcessor::is_legal_context(const std::string& text) {
    if (domains_.find(ContextType::LEGAL) == domains_.end()) {
        return false;
    }
    
    const auto& legal_domain = domains_[ContextType::LEGAL];
    size_t term_count = 0;
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& term : legal_domain->terms) {
        if (lower_text.find(term) != std::string::npos) {
            term_count++;
        }
    }
    
    return term_count >= MIN_TERMS_FOR_DETECTION;
}

inline bool ContextAwareProcessor::is_business_context(const std::string& text) {
    if (domains_.find(ContextType::BUSINESS) == domains_.end()) {
        return false;
    }
    
    const auto& business_domain = domains_[ContextType::BUSINESS];
    size_t term_count = 0;
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& term : business_domain->terms) {
        if (lower_text.find(term) != std::string::npos) {
            term_count++;
        }
    }
    
    return term_count >= MIN_TERMS_FOR_DETECTION;
}

inline float ContextAwareProcessor::calculate_context_confidence(const std::string& text,
                                                                ContextType type) {
    if (domains_.find(type) == domains_.end()) {
        return 0.0f;
    }
    
    const auto& domain = domains_[type];
    float confidence = 0.0f;
    size_t term_count = 0;
    size_t total_words = 0;
    
    std::stringstream ss(text);
    std::string word;
    
    while (ss >> word) {
        total_words++;
        std::string lower_word = word;
        std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
        
        if (domain->terms.find(lower_word) != domain->terms.end()) {
            term_count++;
            
            // Check for term weight
            auto weight_it = domain->term_weights.find(lower_word);
            if (weight_it != domain->term_weights.end()) {
                confidence += weight_it->second;
            } else {
                confidence += 1.0f;
            }
        }
    }
    
    if (total_words > 0) {
        confidence = confidence / total_words;
    }
    
    return std::min(1.0f, confidence);
}

inline std::string ContextAwareProcessor::apply_context_corrections(const std::string& text) {
    if (domains_.find(current_context_) == domains_.end()) {
        return text;
    }
    
    return apply_domain_corrections(text, *domains_[current_context_]);
}

inline std::string ContextAwareProcessor::apply_domain_corrections(const std::string& text,
                                                                  const DomainVocabulary& domain) {
    std::string corrected = text;
    
    // Apply domain-specific term corrections
    // This is a simplified implementation - in production, use more sophisticated matching
    for (const auto& term : domain.terms) {
        // Case-insensitive replacement while preserving original case pattern
        size_t pos = 0;
        while ((pos = corrected.find(term, pos)) != std::string::npos) {
            pos += term.length();
        }
    }
    
    return corrected;
}

inline std::string ContextAwareProcessor::expand_abbreviations(const std::string& text) {
    if (domains_.find(current_context_) == domains_.end()) {
        return text;
    }
    
    const auto& domain = domains_[current_context_];
    std::string expanded = text;
    
    for (const auto& [abbrev, full] : domain->abbreviations) {
        size_t pos = 0;
        while ((pos = expanded.find(abbrev, pos)) != std::string::npos) {
            // Check if it's a whole word (not part of another word)
            bool is_whole_word = true;
            if (pos > 0 && std::isalnum(expanded[pos - 1])) {
                is_whole_word = false;
            }
            if (pos + abbrev.length() < expanded.length() && 
                std::isalnum(expanded[pos + abbrev.length()])) {
                is_whole_word = false;
            }
            
            if (is_whole_word) {
                expanded.replace(pos, abbrev.length(), full);
                pos += full.length();
            } else {
                pos += abbrev.length();
            }
        }
    }
    
    return expanded;
}

inline std::vector<std::string> ContextAwareProcessor::predict_next_words(
    const std::string& partial_sentence) {
    
    std::vector<std::string> predictions;
    
    // Tokenize the partial sentence
    std::vector<std::string> words;
    std::stringstream ss(partial_sentence);
    std::string word;
    
    while (ss >> word) {
        words.push_back(word);
    }
    
    if (words.empty()) {
        return predictions;
    }
    
    // Try trigram prediction first
    if (words.size() >= 2) {
        std::string trigram_key = words[words.size() - 2] + " " + words[words.size() - 1];
        auto trigram_it = trigrams_.find(trigram_key);
        
        if (trigram_it != trigrams_.end()) {
            // Sort by frequency and get top predictions
            std::vector<std::pair<std::string, size_t>> candidates;
            for (const auto& [next_word, count] : trigram_it->second) {
                candidates.push_back({next_word, count});
            }
            
            std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            
            for (size_t i = 0; i < std::min(size_t(5), candidates.size()); ++i) {
                predictions.push_back(candidates[i].first);
            }
        }
    }
    
    // Fall back to bigram prediction
    if (predictions.empty() && words.size() >= 1) {
        auto bigram_it = bigrams_.find(words.back());
        
        if (bigram_it != bigrams_.end()) {
            std::vector<std::pair<std::string, size_t>> candidates;
            for (const auto& [next_word, count] : bigram_it->second) {
                candidates.push_back({next_word, count});
            }
            
            std::sort(candidates.begin(), candidates.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            
            for (size_t i = 0; i < std::min(size_t(5), candidates.size()); ++i) {
                predictions.push_back(candidates[i].first);
            }
        }
    }
    
    return predictions;
}

inline std::vector<std::string> ContextAwareProcessor::extract_entities(const std::string& text) {
    std::vector<std::string> entities;
    
    // Simple entity extraction - look for capitalized words
    // In production, use NER (Named Entity Recognition)
    std::stringstream ss(text);
    std::string word;
    
    while (ss >> word) {
        if (!word.empty() && std::isupper(word[0])) {
            // Check if it's not at the beginning of a sentence
            entities.push_back(word);
        }
    }
    
    return entities;
}

inline void ContextAwareProcessor::reset_context() {
    conversation_context_ = ConversationContext();
    current_context_ = ContextType::GENERAL;
    bigrams_.clear();
    trigrams_.clear();
}

inline ContextAwareProcessor::ContextHint ContextAwareProcessor::get_context_hint() const {
    ContextHint hint(current_context_);
    
    // Add keywords from current domain
    if (domains_.find(current_context_) != domains_.end()) {
        const auto& domain = domains_.at(current_context_);
        for (const auto& term : domain->terms) {
            hint.keywords.push_back(term);
        }
    }
    
    // Add frequently used words from conversation
    std::vector<std::pair<std::string, size_t>> freq_words;
    for (const auto& [word, count] : conversation_context_.topic_frequencies) {
        freq_words.push_back({word, count});
    }
    
    std::sort(freq_words.begin(), freq_words.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    
    for (size_t i = 0; i < std::min(size_t(10), freq_words.size()); ++i) {
        hint.expected_terms.push_back(freq_words[i].first);
    }
    
    return hint;
}

}  // namespace vtt