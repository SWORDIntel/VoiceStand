#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <algorithm>
#include <sstream>

namespace vtt {

// Punctuation restoration using rule-based and statistical methods
class PunctuationRestoration {
public:
    struct Config {
        bool auto_capitalize = true;
        bool add_periods = true;
        bool add_commas = true;
        bool add_questions = true;
        bool format_numbers = true;
        float confidence_threshold = 0.7f;
    };
    
    PunctuationRestoration(const Config& config);
    PunctuationRestoration();
    
    // Restore punctuation in text
    std::string restore(const std::string& text);
    
    // Process with context from previous sentences
    std::string restore_with_context(const std::string& text, 
                                    const std::string& previous_context);
    
    // Add custom rules
    void add_abbreviation(const std::string& abbrev);
    void add_sentence_starter(const std::string& word);
    void add_question_word(const std::string& word);
    
private:
    // Capitalize first letter and proper nouns
    std::string apply_capitalization(const std::string& text);
    
    // Add periods at sentence boundaries
    std::string add_sentence_punctuation(const std::string& text);
    
    // Add commas based on patterns
    std::string add_comma_punctuation(const std::string& text);
    
    // Detect and add question marks
    std::string add_question_marks(const std::string& text);
    
    // Format numbers and dates
    std::string format_numbers_and_dates(const std::string& text);
    
    // Check if word is likely end of sentence
    bool is_sentence_end(const std::string& word, const std::string& next_word);
    
    // Check if word typically starts a sentence
    bool is_sentence_start(const std::string& word);
    
    // Check if sentence is likely a question
    bool is_question(const std::vector<std::string>& words);
    
    // Split text into words
    std::vector<std::string> tokenize(const std::string& text);
    
    // Join words back into text
    std::string join_tokens(const std::vector<std::string>& tokens);
    
    Config config_;
    std::unordered_set<std::string> abbreviations_;
    std::unordered_set<std::string> sentence_starters_;
    std::unordered_set<std::string> question_words_;
    std::unordered_set<std::string> conjunctions_;
    std::unordered_set<std::string> prepositions_;
    
    // Common patterns
    std::vector<std::regex> comma_patterns_;
    std::vector<std::regex> number_patterns_;
};

// Implementation
inline PunctuationRestoration::PunctuationRestoration(const Config& config) 
    : config_(config) {
    
    // Initialize common abbreviations
    abbreviations_ = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", 
                     "inc", "ltd", "co", "corp", "vs", "etc", "eg", "ie"};
    
    // Common sentence starters
    sentence_starters_ = {"the", "a", "an", "this", "that", "these", "those",
                         "i", "you", "he", "she", "it", "we", "they",
                         "my", "your", "his", "her", "its", "our", "their"};
    
    // Question words
    question_words_ = {"what", "when", "where", "who", "whom", "whose",
                      "which", "why", "how", "is", "are", "am", "was",
                      "were", "do", "does", "did", "can", "could",
                      "will", "would", "should", "shall", "may", "might"};
    
    // Conjunctions that often have commas
    conjunctions_ = {"and", "but", "or", "nor", "for", "yet", "so",
                    "however", "therefore", "moreover", "furthermore",
                    "nevertheless", "nonetheless", "meanwhile"};
    
    // Prepositions that might need commas
    prepositions_ = {"after", "before", "during", "since", "until",
                    "although", "because", "if", "unless", "while"};
    
    // Regex patterns for comma insertion
    comma_patterns_ = {
        std::regex(R"(\b(however|therefore|moreover|furthermore|nevertheless)\b)"),
        std::regex(R"(\b(yes|no|well|oh|ah)\b)"),
        std::regex(R"(\d{1,3}(\d{3})+)")  // Numbers with thousands
    };
    
    // Number formatting patterns
    number_patterns_ = {
        std::regex(R"(\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b)"),  // Dates
        std::regex(R"(\b(\d{1,2}):(\d{2})(:(\d{2}))?\b)"),  // Times
        std::regex(R"(\$\s*(\d+(\.\d{2})?)\b)"),  // Currency
    };
}

inline std::string PunctuationRestoration::restore(const std::string& text) {
    if (text.empty()) return text;
    
    std::string result = text;
    
    // Apply punctuation restoration steps
    if (config_.add_periods) {
        result = add_sentence_punctuation(result);
    }
    
    if (config_.add_commas) {
        result = add_comma_punctuation(result);
    }
    
    if (config_.add_questions) {
        result = add_question_marks(result);
    }
    
    if (config_.auto_capitalize) {
        result = apply_capitalization(result);
    }
    
    if (config_.format_numbers) {
        result = format_numbers_and_dates(result);
    }
    
    return result;
}

inline std::string PunctuationRestoration::apply_capitalization(const std::string& text) {
    auto tokens = tokenize(text);
    if (tokens.empty()) return text;
    
    // Capitalize first word
    if (!tokens[0].empty()) {
        tokens[0][0] = std::toupper(tokens[0][0]);
    }
    
    // Capitalize after sentence endings
    for (size_t i = 1; i < tokens.size(); ++i) {
        const auto& prev = tokens[i-1];
        auto& curr = tokens[i];
        
        // Check if previous token ends with sentence punctuation
        if (!prev.empty() && !curr.empty()) {
            char last_char = prev.back();
            if (last_char == '.' || last_char == '!' || last_char == '?') {
                curr[0] = std::toupper(curr[0]);
            }
        }
        
        // Capitalize 'I'
        if (curr == "i") {
            curr = "I";
        }
        
        // Capitalize proper nouns (simple heuristic)
        // In production, would use NER (Named Entity Recognition)
    }
    
    return join_tokens(tokens);
}

inline std::string PunctuationRestoration::add_sentence_punctuation(const std::string& text) {
    auto tokens = tokenize(text);
    if (tokens.empty()) return text;
    
    std::vector<std::string> result;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        result.push_back(tokens[i]);
        
        // Check if this might be end of sentence
        if (i < tokens.size() - 1) {
            const auto& curr = tokens[i];
            const auto& next = tokens[i + 1];
            
            if (is_sentence_end(curr, next)) {
                // Add period if not already there
                if (!curr.empty() && curr.back() != '.' && 
                    curr.back() != '!' && curr.back() != '?') {
                    result.back() += ".";
                }
            }
        }
    }
    
    // Add final period if needed
    if (!result.empty()) {
        auto& last = result.back();
        if (!last.empty() && last.back() != '.' && 
            last.back() != '!' && last.back() != '?') {
            last += ".";
        }
    }
    
    return join_tokens(result);
}

inline std::string PunctuationRestoration::add_comma_punctuation(const std::string& text) {
    std::string result = text;
    
    // Add commas after conjunctions
    for (const auto& pattern : comma_patterns_) {
        result = std::regex_replace(result, pattern, "$1,");
    }
    
    // Add commas in lists (simple heuristic)
    auto tokens = tokenize(result);
    std::vector<std::string> new_tokens;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        new_tokens.push_back(tokens[i]);
        
        // Check for "X and Y and Z" pattern
        if (i + 2 < tokens.size() && 
            tokens[i + 1] == "and" && 
            i > 0 && tokens[i - 1] != "and") {
            // Might be a list
            if (!tokens[i].empty() && tokens[i].back() != ',') {
                new_tokens.back() += ",";
            }
        }
    }
    
    return join_tokens(new_tokens);
}

inline std::string PunctuationRestoration::add_question_marks(const std::string& text) {
    // Split into sentences
    std::vector<std::string> sentences;
    std::stringstream ss(text);
    std::string sentence;
    
    while (std::getline(ss, sentence, '.')) {
        if (!sentence.empty()) {
            auto words = tokenize(sentence);
            
            if (is_question(words)) {
                sentence += "?";
            } else if (sentence.back() != '?' && sentence.back() != '!') {
                sentence += ".";
            }
            
            sentences.push_back(sentence);
        }
    }
    
    // Join sentences
    std::string result;
    for (const auto& s : sentences) {
        result += s + " ";
    }
    
    // Remove trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}

inline bool PunctuationRestoration::is_sentence_end(const std::string& word, 
                                                   const std::string& next_word) {
    // Check if word is an abbreviation
    std::string lower_word = word;
    std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
    
    if (abbreviations_.find(lower_word) != abbreviations_.end()) {
        return false;
    }
    
    // Check if next word starts with capital (likely new sentence)
    if (!next_word.empty() && std::isupper(next_word[0])) {
        return true;
    }
    
    // Check if next word is a sentence starter
    std::string lower_next = next_word;
    std::transform(lower_next.begin(), lower_next.end(), lower_next.begin(), ::tolower);
    
    return sentence_starters_.find(lower_next) != sentence_starters_.end();
}

inline bool PunctuationRestoration::is_question(const std::vector<std::string>& words) {
    if (words.empty()) return false;
    
    // Check if starts with question word
    std::string first_word = words[0];
    std::transform(first_word.begin(), first_word.end(), first_word.begin(), ::tolower);
    
    return question_words_.find(first_word) != question_words_.end();
}

inline std::vector<std::string> PunctuationRestoration::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    
    while (ss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

inline std::string PunctuationRestoration::join_tokens(const std::vector<std::string>& tokens) {
    std::string result;
    for (size_t i = 0; i < tokens.size(); ++i) {
        result += tokens[i];
        if (i < tokens.size() - 1) {
            result += " ";
        }
    }
    return result;
}

inline std::string PunctuationRestoration::format_numbers_and_dates(const std::string& text) {
    std::string result = text;
    
    // Format thousands in numbers
    result = std::regex_replace(result, std::regex(R"(\b(\d{1,3})(\d{3})\b)"), "$1,$2");
    
    // Format dates (simple)
    result = std::regex_replace(result, 
                                std::regex(R"(\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b)"), 
                                "$1/$2/$3");
    
    return result;
}

}  // namespace vtt