#pragma once

#include "voice_commands.h"
#include "wake_word_detector.h"
#include <json/json.h>
#include <chrono>
#include <regex>
#include <unordered_map>
#include <queue>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <iostream>

namespace vtt {

/**
 * Personal Voice Commands Engine - Phase 2 Week 4
 * Building on GNA foundation with DTW-based pattern matching
 * Provides personal productivity automation and system control
 */
class PersonalVoiceCommands {
public:
    // Security privilege levels for commands
    enum class PrivilegeLevel {
        USER = 0,      // User-level commands (open app, minimize window)
        SYSTEM = 1,    // System commands (lock screen, switch workspace)
        ADMIN = 2      // Administrative commands (shutdown, install)
    };

    // Command execution context
    struct CommandContext {
        std::string raw_text;
        float confidence = 0.0f;
        std::chrono::steady_clock::time_point timestamp;
        std::string speaker_id;
        PrivilegeLevel required_privilege = PrivilegeLevel::USER;
        bool requires_confirmation = false;

        CommandContext(const std::string& text, float conf = 1.0f)
            : raw_text(text), confidence(conf), timestamp(std::chrono::steady_clock::now()) {}
    };

    // Personal macro definition
    struct PersonalMacro {
        std::string name;
        std::vector<std::string> voice_patterns;
        std::vector<std::string> commands;        // Shell commands to execute
        std::vector<std::string> applications;    // Applications to launch
        std::vector<std::string> keystrokes;      // Key combinations to send
        PrivilegeLevel privilege = PrivilegeLevel::USER;
        bool enabled = true;
        size_t execution_count = 0;
        std::chrono::steady_clock::time_point last_used;

        PersonalMacro(const std::string& n) : name(n), last_used(std::chrono::steady_clock::now()) {}
    };

    // Command execution result with detailed feedback
    struct ExecutionResult {
        bool success = false;
        std::string command_name;
        std::string response;
        std::string error_message;
        std::chrono::steady_clock::time_point timestamp;
        float confidence = 0.0f;
        int exit_code = 0;
        std::string output;
        float execution_time_ms = 0.0f;

        ExecutionResult() : timestamp(std::chrono::steady_clock::now()) {}
    };

    // Configuration for personal commands
    struct Config {
        float dtw_threshold;
        size_t max_command_history;
        bool enable_learning;
        bool enable_security_validation;
        bool enable_confirmation_for_system;
        std::string config_file;

        // Performance tuning
        size_t dtw_cache_size;
        float confidence_boost;
        size_t pattern_cache_ms;

        // Default constructor
        Config()
            : dtw_threshold(0.80f)
            , max_command_history(1000)
            , enable_learning(true)
            , enable_security_validation(true)
            , enable_confirmation_for_system(true)
            , config_file("~/.config/voice-to-text/personal_commands.json")
            , dtw_cache_size(100)
            , confidence_boost(0.1f)
            , pattern_cache_ms(5000)
        {}
    };

    using CommandCallback = std::function<void(const ExecutionResult&)>;
    using SecurityValidator = std::function<bool(const PersonalMacro&, const CommandContext&)>;

    PersonalVoiceCommands(const Config& config = Config());
    ~PersonalVoiceCommands();

    // Core functionality
    bool initialize();
    ExecutionResult process_voice_input(const std::string& text, float confidence = 1.0f);

    // Personal macro management
    bool register_personal_macro(const PersonalMacro& macro);
    bool remove_personal_macro(const std::string& name);
    bool update_personal_macro(const std::string& name, const PersonalMacro& updated_macro);
    std::vector<PersonalMacro> get_personal_macros() const;
    PersonalMacro* find_macro(const std::string& name);

    // Built-in system commands registration
    void register_system_commands();
    void register_application_commands();
    void register_window_management_commands();
    void register_productivity_commands();

    // DTW-based pattern matching (core engine)
    float compute_dtw_similarity(const std::string& input, const std::string& pattern);
    std::vector<float> extract_phonetic_features(const std::string& text);

    // Configuration and persistence
    bool save_config();
    bool load_config();
    Json::Value export_macros() const;
    bool import_macros(const Json::Value& macros_json);

    // Statistics and analytics
    struct CommandStats {
        size_t total_commands_executed = 0;
        size_t successful_executions = 0;
        size_t failed_executions = 0;
        size_t security_rejections = 0;
        float average_confidence = 0.0f;
        float average_execution_time_ms = 0.0f;
        std::unordered_map<std::string, size_t> command_usage_counts;
        std::unordered_map<std::string, float> command_success_rates;
    };

    CommandStats get_statistics() const;
    void print_usage_report() const;

    // Callbacks and event handling
    void set_command_callback(CommandCallback callback) { command_callback_ = callback; }
    void set_security_validator(SecurityValidator validator) { security_validator_ = validator; }

    // Security and privilege management
    void set_current_privilege_level(PrivilegeLevel level) { current_privilege_ = level; }
    PrivilegeLevel get_current_privilege_level() const { return current_privilege_; }
    bool validate_command_security(const PersonalMacro& macro, const CommandContext& context);

    // Learning and adaptation
    void learn_from_execution(const std::string& command_name, bool success, float confidence);
    void adapt_patterns_from_usage();
    std::vector<std::string> suggest_similar_commands(const std::string& input);

private:
    Config config_;
    std::unordered_map<std::string, std::unique_ptr<PersonalMacro>> personal_macros_;
    mutable std::mutex macros_mutex_;

    // DTW computation and caching
    struct DTWCacheEntry {
        float similarity;
        std::chrono::steady_clock::time_point timestamp;
    };
    mutable std::unordered_map<std::string, DTWCacheEntry> dtw_cache_;
    mutable std::mutex cache_mutex_;

    // Command execution
    ExecutionResult execute_shell_commands(const PersonalMacro& macro, const CommandContext& context);
    ExecutionResult execute_application_launch(const PersonalMacro& macro, const CommandContext& context);
    ExecutionResult execute_keystroke_sequence(const PersonalMacro& macro, const CommandContext& context);

    // Built-in command handlers
    void register_builtin_system_commands();
    ExecutionResult handle_lock_screen(const CommandContext& context);
    ExecutionResult handle_open_terminal(const CommandContext& context);
    ExecutionResult handle_switch_workspace(const CommandContext& context, const std::string& workspace);
    ExecutionResult handle_open_application(const CommandContext& context, const std::string& app_name);
    ExecutionResult handle_window_operation(const CommandContext& context, const std::string& operation);
    ExecutionResult handle_volume_control(const CommandContext& context, const std::string& action);
    ExecutionResult handle_screenshot(const CommandContext& context, const std::string& type = "screen");

    // Security and validation
    bool request_user_confirmation(const PersonalMacro& macro, const CommandContext& context);
    void log_security_event(const PersonalMacro& macro, const CommandContext& context, const std::string& reason);

    // State management
    std::atomic<PrivilegeLevel> current_privilege_{PrivilegeLevel::USER};
    std::queue<ExecutionResult> command_history_;
    mutable CommandStats stats_;

    // Callbacks
    CommandCallback command_callback_;
    SecurityValidator security_validator_;

    // Performance optimization
    std::chrono::steady_clock::time_point last_cache_cleanup_;
    void cleanup_caches();

    // Pattern learning database
    struct PatternLearning {
        std::string original_pattern;
        std::vector<std::string> user_variations;
        float success_rate = 0.0f;
        size_t usage_count = 0;
    };
    std::unordered_map<std::string, PatternLearning> pattern_learning_;
};

// Implementation
inline PersonalVoiceCommands::PersonalVoiceCommands(const Config& config)
    : config_(config), last_cache_cleanup_(std::chrono::steady_clock::now()) {
}

inline PersonalVoiceCommands::~PersonalVoiceCommands() {
    save_config();
    print_usage_report();
}

inline bool PersonalVoiceCommands::initialize() {
    std::cout << "[INFO] Initializing Personal Voice Commands Engine\n";

    // Load existing configuration
    if (!load_config()) {
        std::cout << "[WARN] Could not load existing config, using defaults\n";
    }

    // Register built-in command sets
    register_system_commands();
    register_application_commands();
    register_window_management_commands();
    register_productivity_commands();

    std::cout << "[INFO] Personal Voice Commands initialized with "
              << personal_macros_.size() << " commands\n";
    return true;
}

inline PersonalVoiceCommands::ExecutionResult PersonalVoiceCommands::process_voice_input(
    const std::string& text, float confidence) {

    auto start_time = std::chrono::high_resolution_clock::now();

    ExecutionResult result;
    result.confidence = confidence;

    CommandContext context(text, confidence);

    // Find best matching command using DTW
    std::string best_match;
    float best_similarity = 0.0f;
    PersonalMacro* matched_macro = nullptr;

    {
        std::lock_guard<std::mutex> lock(macros_mutex_);
        for (const auto& [name, macro] : personal_macros_) {
            if (!macro->enabled) continue;

            for (const auto& pattern : macro->voice_patterns) {
                float similarity = compute_dtw_similarity(text, pattern);

                // Apply confidence boost for frequently used commands
                if (macro->execution_count > 10) {
                    similarity += config_.confidence_boost;
                }

                if (similarity > best_similarity && similarity >= config_.dtw_threshold) {
                    best_similarity = similarity;
                    best_match = name;
                    matched_macro = macro.get();
                }
            }
        }
    }

    if (!matched_macro) {
        result.response = "No matching personal command found";
        result.error_message = "DTW similarity below threshold (" + std::to_string(config_.dtw_threshold) + ")";
        return result;
    }

    result.command_name = best_match;
    result.confidence = std::min(confidence * best_similarity, 1.0f);

    // Security validation
    context.required_privilege = matched_macro->privilege;
    context.requires_confirmation = matched_macro->privilege >= PrivilegeLevel::SYSTEM;

    if (config_.enable_security_validation) {
        if (!validate_command_security(*matched_macro, context)) {
            result.response = "Command rejected by security validation";
            result.error_message = "Insufficient privileges or security policy violation";
            stats_.security_rejections++;
            return result;
        }
    }

    // Request confirmation for system-level commands
    if (context.requires_confirmation && config_.enable_confirmation_for_system) {
        if (!request_user_confirmation(*matched_macro, context)) {
            result.response = "Command cancelled by user";
            return result;
        }
    }

    // Execute the matched macro
    try {
        // Execute shell commands
        if (!matched_macro->commands.empty()) {
            auto shell_result = execute_shell_commands(*matched_macro, context);
            if (!shell_result.success) {
                result = shell_result;
            } else {
                result.output += shell_result.output;
                result.success = true;
            }
        }

        // Launch applications
        if (!matched_macro->applications.empty()) {
            auto app_result = execute_application_launch(*matched_macro, context);
            if (!app_result.success && result.success) {
                result = app_result;
            } else if (app_result.success) {
                result.output += app_result.output;
                result.success = true;
            }
        }

        // Send keystrokes
        if (!matched_macro->keystrokes.empty()) {
            auto key_result = execute_keystroke_sequence(*matched_macro, context);
            if (!key_result.success && result.success) {
                result = key_result;
            } else if (key_result.success) {
                result.output += key_result.output;
                result.success = true;
            }
        }

        if (result.success) {
            result.response = "Personal command executed: " + matched_macro->name;
            matched_macro->execution_count++;
            matched_macro->last_used = std::chrono::steady_clock::now();

            // Update statistics
            stats_.total_commands_executed++;
            stats_.successful_executions++;
            stats_.command_usage_counts[matched_macro->name]++;

            // Learn from successful execution
            if (config_.enable_learning) {
                learn_from_execution(matched_macro->name, true, result.confidence);
            }
        }

    } catch (const std::exception& e) {
        result.success = false;
        result.response = "Command execution failed";
        result.error_message = e.what();
        stats_.failed_executions++;

        if (config_.enable_learning) {
            learn_from_execution(matched_macro->name, false, result.confidence);
        }
    }

    // Calculate execution time
    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Update statistics
    stats_.average_confidence = (stats_.average_confidence * (stats_.total_commands_executed - 1) +
                                result.confidence) / stats_.total_commands_executed;
    stats_.average_execution_time_ms = (stats_.average_execution_time_ms * (stats_.total_commands_executed - 1) +
                                       result.execution_time_ms) / stats_.total_commands_executed;

    // Add to command history
    command_history_.push(result);
    if (command_history_.size() > config_.max_command_history) {
        command_history_.pop();
    }

    // Call callback if set
    if (command_callback_) {
        command_callback_(result);
    }

    // Periodic cache cleanup
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::minutes>(now - last_cache_cleanup_).count() > 5) {
        cleanup_caches();
        last_cache_cleanup_ = now;
    }

    return result;
}

inline float PersonalVoiceCommands::compute_dtw_similarity(const std::string& input, const std::string& pattern) {
    // Check cache first
    std::string cache_key = input + "|" + pattern;

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = dtw_cache_.find(cache_key);
        if (it != dtw_cache_.end()) {
            auto age = std::chrono::steady_clock::now() - it->second.timestamp;
            if (std::chrono::duration_cast<std::chrono::milliseconds>(age).count() < static_cast<long>(config_.pattern_cache_ms)) {
                return it->second.similarity;
            } else {
                dtw_cache_.erase(it);
            }
        }
    }

    // Extract phonetic features for both input and pattern
    auto input_features = extract_phonetic_features(input);
    auto pattern_features = extract_phonetic_features(pattern);

    if (input_features.empty() || pattern_features.empty()) {
        return 0.0f;
    }

    // Compute DTW distance
    size_t n = input_features.size();
    size_t m = pattern_features.size();

    // Initialize DTW matrix
    std::vector<std::vector<float>> dtw(n + 1, std::vector<float>(m + 1, INFINITY));
    dtw[0][0] = 0.0f;

    // Fill DTW matrix
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            float cost = std::abs(input_features[i-1] - pattern_features[j-1]);
            dtw[i][j] = cost + std::min({
                dtw[i-1][j],      // insertion
                dtw[i][j-1],      // deletion
                dtw[i-1][j-1]     // substitution
            });
        }
    }

    // Normalize by path length and convert to similarity
    float distance = dtw[n][m] / (n + m);
    float similarity = 1.0f / (1.0f + distance);

    // Cache the result
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (dtw_cache_.size() < config_.dtw_cache_size) {
            dtw_cache_[cache_key] = {similarity, std::chrono::steady_clock::now()};
        }
    }

    return similarity;
}

inline std::vector<float> PersonalVoiceCommands::extract_phonetic_features(const std::string& text) {
    std::vector<float> features;

    // Convert to lowercase for consistency
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);

    // Extract basic phonetic features
    for (char c : lower_text) {
        if (std::isalpha(c)) {
            // Map characters to phonetic features (simplified)
            float feature = 0.0f;

            // Vowel vs consonant
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
                feature += 1.0f;  // Vowel marker
            }

            // Character position influence
            feature += (c - 'a') / 25.0f;  // Normalized position in alphabet

            // Add phonetic similarity groups
            if (c == 'b' || c == 'p') feature += 0.1f;  // Bilabial
            if (c == 'd' || c == 't') feature += 0.2f;  // Alveolar
            if (c == 'g' || c == 'k') feature += 0.3f;  // Velar
            if (c == 'f' || c == 'v') feature += 0.4f;  // Labiodental
            if (c == 's' || c == 'z') feature += 0.5f;  // Sibilant
            if (c == 'm' || c == 'n') feature += 0.6f;  // Nasal
            if (c == 'l' || c == 'r') feature += 0.7f;  // Liquid

            features.push_back(feature);
        } else if (std::isspace(c)) {
            features.push_back(0.0f);  // Word boundary
        }
    }

    return features;
}

// Additional method implementations continue...
// [Note: This would continue with all the method implementations for the full class]

}  // namespace vtt