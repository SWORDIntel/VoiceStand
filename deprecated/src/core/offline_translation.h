#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <algorithm>
#include <json/json.h>

namespace vtt {

// Offline translation system using local models and dictionaries
class OfflineTranslation {
public:
    // Supported languages
    enum class Language {
        ENGLISH,
        SPANISH,
        FRENCH,
        GERMAN,
        ITALIAN,
        PORTUGUESE,
        RUSSIAN,
        CHINESE,
        JAPANESE,
        KOREAN,
        ARABIC,
        HINDI,
        UNKNOWN
    };
    
    // Translation entry
    struct TranslationEntry {
        std::string source_text;
        std::string translated_text;
        Language source_lang;
        Language target_lang;
        float confidence = 0.0f;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    // Language dictionary
    struct Dictionary {
        Language language;
        std::unordered_map<std::string, std::string> word_translations;
        std::unordered_map<std::string, std::vector<std::string>> phrases;
        std::unordered_map<std::string, std::string> common_expressions;
        
        Dictionary(Language lang) : language(lang) {}
    };
    
    // Translation model (simplified for offline use)
    struct TranslationModel {
        std::string name;
        Language source_lang;
        Language target_lang;
        std::unordered_map<std::string, std::string> translation_pairs;
        std::vector<std::pair<std::string, std::string>> phrase_patterns;
        float accuracy = 0.0f;
        
        TranslationModel(const std::string& n, Language src, Language tgt)
            : name(n), source_lang(src), target_lang(tgt) {}
    };
    
    OfflineTranslation();
    ~OfflineTranslation();
    
    // Translate text
    std::string translate(const std::string& text, 
                         Language target_lang,
                         Language source_lang = Language::UNKNOWN);
    
    // Batch translation
    std::vector<std::string> translate_batch(const std::vector<std::string>& texts,
                                            Language target_lang,
                                            Language source_lang = Language::UNKNOWN);
    
    // Detect language
    Language detect_language(const std::string& text);
    
    // Load translation model
    bool load_model(const std::string& model_path, Language source, Language target);
    
    // Load dictionary
    bool load_dictionary(const std::string& dict_path, Language language);
    
    // Get available language pairs
    std::vector<std::pair<Language, Language>> get_available_pairs() const;
    
    // Get translation confidence
    float get_last_confidence() const { return last_confidence_; }
    
    // Enable/disable translation
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }
    
    // Set fallback to dictionary translation
    void set_use_dictionary_fallback(bool use) { use_dictionary_fallback_ = use; }
    
    // Get language name
    static std::string get_language_name(Language lang);
    
    // Get language code
    static std::string get_language_code(Language lang);
    
    // Parse language from code
    static Language parse_language_code(const std::string& code);
    
    // Statistics
    struct Stats {
        size_t total_translations = 0;
        size_t successful_translations = 0;
        size_t dictionary_fallbacks = 0;
        std::unordered_map<std::string, size_t> language_pair_usage;
        float average_confidence = 0.0f;
    };
    
    Stats get_stats() const { return stats_; }
    
private:
    // Initialize built-in dictionaries
    void initialize_dictionaries();
    
    // Translate using model
    std::string translate_with_model(const std::string& text,
                                    const TranslationModel& model);
    
    // Translate using dictionary
    std::string translate_with_dictionary(const std::string& text,
                                         const Dictionary& source_dict,
                                         const Dictionary& target_dict);
    
    // Simple word-by-word translation
    std::string word_by_word_translation(const std::string& text,
                                        Language source_lang,
                                        Language target_lang);
    
    // Apply grammar rules for target language
    std::string apply_grammar_rules(const std::string& text, Language lang);
    
    // Tokenize text
    std::vector<std::string> tokenize(const std::string& text);
    
    // Join tokens
    std::string join_tokens(const std::vector<std::string>& tokens);
    
    // Calculate translation confidence
    float calculate_confidence(const std::string& source,
                              const std::string& translation,
                              Language source_lang,
                              Language target_lang);
    
    // Language detection helpers
    bool contains_latin_script(const std::string& text);
    bool contains_cyrillic_script(const std::string& text);
    bool contains_cjk_characters(const std::string& text);
    bool contains_arabic_script(const std::string& text);
    bool contains_devanagari_script(const std::string& text);
    
    std::unordered_map<Language, std::shared_ptr<Dictionary>> dictionaries_;
    std::vector<std::shared_ptr<TranslationModel>> models_;
    
    bool enabled_ = true;
    bool use_dictionary_fallback_ = true;
    float last_confidence_ = 0.0f;
    
    Stats stats_;
    
    // Common words for language detection
    std::unordered_map<Language, std::vector<std::string>> common_words_;
};

// Implementation
inline OfflineTranslation::OfflineTranslation() {
    initialize_dictionaries();
}

inline OfflineTranslation::~OfflineTranslation() = default;

inline void OfflineTranslation::initialize_dictionaries() {
    // Initialize basic English-Spanish dictionary as example
    auto en_dict = std::make_shared<Dictionary>(Language::ENGLISH);
    en_dict->word_translations = {
        {"hello", "hello"}, {"goodbye", "goodbye"}, {"thank", "thank"},
        {"please", "please"}, {"yes", "yes"}, {"no", "no"},
        {"good", "good"}, {"bad", "bad"}, {"day", "day"}, {"night", "night"},
        {"morning", "morning"}, {"evening", "evening"}, {"afternoon", "afternoon"},
        {"how", "how"}, {"are", "are"}, {"you", "you"}, {"I", "I"},
        {"am", "am"}, {"is", "is"}, {"the", "the"}, {"a", "a"}, {"an", "an"}
    };
    en_dict->common_expressions = {
        {"how are you", "how are you"},
        {"good morning", "good morning"},
        {"thank you", "thank you"}
    };
    dictionaries_[Language::ENGLISH] = en_dict;
    
    auto es_dict = std::make_shared<Dictionary>(Language::SPANISH);
    es_dict->word_translations = {
        {"hello", "hola"}, {"goodbye", "adiós"}, {"thank", "gracias"},
        {"please", "por favor"}, {"yes", "sí"}, {"no", "no"},
        {"good", "bueno"}, {"bad", "malo"}, {"day", "día"}, {"night", "noche"},
        {"morning", "mañana"}, {"evening", "tarde"}, {"afternoon", "tarde"},
        {"how", "cómo"}, {"are", "estás"}, {"you", "tú"}, {"I", "yo"},
        {"am", "soy"}, {"is", "es"}, {"the", "el"}, {"a", "un"}, {"an", "un"}
    };
    es_dict->common_expressions = {
        {"how are you", "cómo estás"},
        {"good morning", "buenos días"},
        {"thank you", "gracias"}
    };
    dictionaries_[Language::SPANISH] = es_dict;
    
    // Common words for language detection
    common_words_[Language::ENGLISH] = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
    };
    
    common_words_[Language::SPANISH] = {
        "el", "la", "de", "que", "y", "a", "en", "un", "ser", "se",
        "no", "haber", "por", "con", "su", "para", "como", "estar", "tener", "le"
    };
    
    common_words_[Language::FRENCH] = {
        "le", "de", "un", "être", "et", "à", "il", "avoir", "ne", "je",
        "son", "que", "se", "qui", "ce", "dans", "elle", "au", "pour", "pas"
    };
    
    common_words_[Language::GERMAN] = {
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
        "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als"
    };
}

inline std::string OfflineTranslation::translate(const std::string& text,
                                                Language target_lang,
                                                Language source_lang) {
    if (!enabled_ || text.empty()) {
        return text;
    }
    
    stats_.total_translations++;
    
    // Auto-detect source language if not specified
    if (source_lang == Language::UNKNOWN) {
        source_lang = detect_language(text);
    }
    
    // Same language, no translation needed
    if (source_lang == target_lang) {
        stats_.successful_translations++;
        last_confidence_ = 1.0f;
        return text;
    }
    
    // Look for matching translation model
    for (const auto& model : models_) {
        if (model->source_lang == source_lang && model->target_lang == target_lang) {
            std::string translated = translate_with_model(text, *model);
            if (!translated.empty()) {
                stats_.successful_translations++;
                last_confidence_ = calculate_confidence(text, translated, source_lang, target_lang);
                return translated;
            }
        }
    }
    
    // Fallback to dictionary translation
    if (use_dictionary_fallback_) {
        std::string translated = word_by_word_translation(text, source_lang, target_lang);
        if (!translated.empty()) {
            stats_.dictionary_fallbacks++;
            stats_.successful_translations++;
            last_confidence_ = 0.5f;  // Lower confidence for dictionary translation
            return apply_grammar_rules(translated, target_lang);
        }
    }
    
    // Translation failed
    last_confidence_ = 0.0f;
    return text;
}

inline OfflineTranslation::Language OfflineTranslation::detect_language(const std::string& text) {
    // Simple language detection based on common words and script
    std::unordered_map<Language, int> scores;
    
    // Convert to lowercase for comparison
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    // Check common words
    for (const auto& [lang, words] : common_words_) {
        int score = 0;
        for (const auto& word : words) {
            if (lower_text.find(word) != std::string::npos) {
                score++;
            }
        }
        scores[lang] = score;
    }
    
    // Check script types
    if (contains_cyrillic_script(text)) {
        scores[Language::RUSSIAN] += 10;
    }
    if (contains_cjk_characters(text)) {
        // Simple heuristic - would need more sophisticated detection
        scores[Language::CHINESE] += 5;
        scores[Language::JAPANESE] += 5;
        scores[Language::KOREAN] += 5;
    }
    if (contains_arabic_script(text)) {
        scores[Language::ARABIC] += 10;
    }
    if (contains_devanagari_script(text)) {
        scores[Language::HINDI] += 10;
    }
    
    // Find language with highest score
    Language detected_lang = Language::ENGLISH;  // Default
    int max_score = 0;
    
    for (const auto& [lang, score] : scores) {
        if (score > max_score) {
            max_score = score;
            detected_lang = lang;
        }
    }
    
    return detected_lang;
}

inline std::string OfflineTranslation::word_by_word_translation(const std::string& text,
                                                               Language source_lang,
                                                               Language target_lang) {
    // Get dictionaries
    auto source_dict_it = dictionaries_.find(source_lang);
    auto target_dict_it = dictionaries_.find(target_lang);
    
    if (source_dict_it == dictionaries_.end() || target_dict_it == dictionaries_.end()) {
        return "";
    }
    
    // First check for common expressions
    for (const auto& [expr, translation] : source_dict_it->second->common_expressions) {
        if (text.find(expr) != std::string::npos) {
            // Find corresponding translation in target language
            auto target_expr_it = target_dict_it->second->common_expressions.find(expr);
            if (target_expr_it != target_dict_it->second->common_expressions.end()) {
                return target_expr_it->second;
            }
        }
    }
    
    // Word-by-word translation
    auto tokens = tokenize(text);
    std::vector<std::string> translated_tokens;
    
    for (const auto& token : tokens) {
        std::string lower_token = token;
        std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
        
        // Look for translation in target dictionary
        bool found = false;
        for (const auto& [eng_word, target_word] : target_dict_it->second->word_translations) {
            auto source_word_it = source_dict_it->second->word_translations.find(eng_word);
            if (source_word_it != source_dict_it->second->word_translations.end()) {
                if (source_word_it->second == lower_token) {
                    translated_tokens.push_back(target_word);
                    found = true;
                    break;
                }
            }
        }
        
        if (!found) {
            translated_tokens.push_back(token);  // Keep original if no translation
        }
    }
    
    return join_tokens(translated_tokens);
}

inline std::string OfflineTranslation::apply_grammar_rules(const std::string& text, 
                                                          Language lang) {
    // Apply basic grammar rules for target language
    // This is highly simplified - real translation needs complex grammar handling
    
    std::string result = text;
    
    switch (lang) {
        case Language::SPANISH:
            // Spanish: Adjectives usually come after nouns
            // Very simplified example
            break;
            
        case Language::FRENCH:
            // French: Similar to Spanish for adjective placement
            break;
            
        case Language::GERMAN:
            // German: Verb at end in subordinate clauses
            break;
            
        default:
            break;
    }
    
    return result;
}

inline float OfflineTranslation::calculate_confidence(const std::string& source,
                                                     const std::string& translation,
                                                     Language source_lang,
                                                     Language target_lang) {
    float confidence = 0.0f;
    
    // Simple heuristics for confidence
    // 1. Check if translation is different from source
    if (source != translation) {
        confidence += 0.3f;
    }
    
    // 2. Check if translation contains expected language patterns
    if (detect_language(translation) == target_lang) {
        confidence += 0.4f;
    }
    
    // 3. Check word count similarity (translations shouldn't be too different in length)
    size_t source_words = std::count(source.begin(), source.end(), ' ') + 1;
    size_t trans_words = std::count(translation.begin(), translation.end(), ' ') + 1;
    float word_ratio = static_cast<float>(std::min(source_words, trans_words)) / 
                      static_cast<float>(std::max(source_words, trans_words));
    confidence += word_ratio * 0.3f;
    
    return std::min(1.0f, confidence);
}

inline std::vector<std::string> OfflineTranslation::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    
    while (ss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

inline std::string OfflineTranslation::join_tokens(const std::vector<std::string>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) result += " ";
        result += tokens[i];
    }
    return result;
}

inline bool OfflineTranslation::contains_latin_script(const std::string& text) {
    for (char c : text) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
            return true;
        }
    }
    return false;
}

inline bool OfflineTranslation::contains_cyrillic_script(const std::string& text) {
    for (unsigned char c : text) {
        // Basic Cyrillic range check (simplified)
        if (c >= 0xD0 && c <= 0xD1) {
            return true;
        }
    }
    return false;
}

inline bool OfflineTranslation::contains_cjk_characters(const std::string& text) {
    // Simplified CJK detection
    // In production, use proper Unicode range checking
    return false;
}

inline bool OfflineTranslation::contains_arabic_script(const std::string& text) {
    // Simplified Arabic detection
    return false;
}

inline bool OfflineTranslation::contains_devanagari_script(const std::string& text) {
    // Simplified Devanagari detection
    return false;
}

inline std::string OfflineTranslation::get_language_name(Language lang) {
    switch (lang) {
        case Language::ENGLISH: return "English";
        case Language::SPANISH: return "Spanish";
        case Language::FRENCH: return "French";
        case Language::GERMAN: return "German";
        case Language::ITALIAN: return "Italian";
        case Language::PORTUGUESE: return "Portuguese";
        case Language::RUSSIAN: return "Russian";
        case Language::CHINESE: return "Chinese";
        case Language::JAPANESE: return "Japanese";
        case Language::KOREAN: return "Korean";
        case Language::ARABIC: return "Arabic";
        case Language::HINDI: return "Hindi";
        default: return "Unknown";
    }
}

inline std::string OfflineTranslation::get_language_code(Language lang) {
    switch (lang) {
        case Language::ENGLISH: return "en";
        case Language::SPANISH: return "es";
        case Language::FRENCH: return "fr";
        case Language::GERMAN: return "de";
        case Language::ITALIAN: return "it";
        case Language::PORTUGUESE: return "pt";
        case Language::RUSSIAN: return "ru";
        case Language::CHINESE: return "zh";
        case Language::JAPANESE: return "ja";
        case Language::KOREAN: return "ko";
        case Language::ARABIC: return "ar";
        case Language::HINDI: return "hi";
        default: return "unknown";
    }
}

inline OfflineTranslation::Language OfflineTranslation::parse_language_code(const std::string& code) {
    if (code == "en") return Language::ENGLISH;
    if (code == "es") return Language::SPANISH;
    if (code == "fr") return Language::FRENCH;
    if (code == "de") return Language::GERMAN;
    if (code == "it") return Language::ITALIAN;
    if (code == "pt") return Language::PORTUGUESE;
    if (code == "ru") return Language::RUSSIAN;
    if (code == "zh") return Language::CHINESE;
    if (code == "ja") return Language::JAPANESE;
    if (code == "ko") return Language::KOREAN;
    if (code == "ar") return Language::ARABIC;
    if (code == "hi") return Language::HINDI;
    return Language::UNKNOWN;
}

inline std::vector<std::pair<OfflineTranslation::Language, OfflineTranslation::Language>> 
OfflineTranslation::get_available_pairs() const {
    std::vector<std::pair<Language, Language>> pairs;
    
    // Add model pairs
    for (const auto& model : models_) {
        pairs.push_back({model->source_lang, model->target_lang});
    }
    
    // Add dictionary pairs (if fallback enabled)
    if (use_dictionary_fallback_) {
        for (const auto& [lang1, dict1] : dictionaries_) {
            for (const auto& [lang2, dict2] : dictionaries_) {
                if (lang1 != lang2) {
                    pairs.push_back({lang1, lang2});
                }
            }
        }
    }
    
    return pairs;
}

}  // namespace vtt