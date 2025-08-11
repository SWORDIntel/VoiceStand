#pragma once

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <regex>
#include <memory>
#include <chrono>

namespace vtt {

// Voice command system for controlling application through speech
class VoiceCommands {
public:
    // Command handler function type
    using CommandHandler = std::function<void(const std::vector<std::string>& args)>;
    
    // Command definition
    struct Command {
        std::string name;
        std::vector<std::string> patterns;  // Regex patterns to match
        CommandHandler handler;
        std::string description;
        bool requires_confirmation = false;
        int min_confidence = 80;  // Minimum confidence percentage
        
        Command(const std::string& n, const std::vector<std::string>& p, 
                CommandHandler h, const std::string& d = "")
            : name(n), patterns(p), handler(h), description(d) {}
    };
    
    // Command execution result
    struct ExecutionResult {
        bool success = false;
        std::string command_name;
        std::string response;
        std::chrono::steady_clock::time_point timestamp;
        float confidence = 0.0f;
    };
    
    VoiceCommands();
    ~VoiceCommands() = default;
    
    // Register a new command
    void register_command(const std::string& name,
                         const std::vector<std::string>& patterns,
                         CommandHandler handler,
                         const std::string& description = "",
                         bool requires_confirmation = false);
    
    // Process text and execute matching commands
    ExecutionResult process_text(const std::string& text, float confidence = 1.0f);
    
    // Enable/disable command processing
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }
    
    // Set confirmation callback
    void set_confirmation_callback(std::function<bool(const std::string&)> callback) {
        confirmation_callback_ = callback;
    }
    
    // Get list of available commands
    std::vector<std::string> get_command_list() const;
    
    // Get command help text
    std::string get_help() const;
    
    // Register built-in commands
    void register_builtin_commands();
    
private:
    // Find matching command for text
    std::shared_ptr<Command> find_command(const std::string& text, 
                                         std::vector<std::string>& matches);
    
    // Extract command arguments from matched text
    std::vector<std::string> extract_arguments(const std::string& text,
                                               const std::regex& pattern,
                                               const std::smatch& matches);
    
    // Built-in command handlers
    void handle_help(const std::vector<std::string>& args);
    void handle_start_recording(const std::vector<std::string>& args);
    void handle_stop_recording(const std::vector<std::string>& args);
    void handle_save_transcript(const std::vector<std::string>& args);
    void handle_clear_text(const std::vector<std::string>& args);
    void handle_switch_language(const std::vector<std::string>& args);
    void handle_increase_volume(const std::vector<std::string>& args);
    void handle_decrease_volume(const std::vector<std::string>& args);
    void handle_mute(const std::vector<std::string>& args);
    void handle_settings(const std::vector<std::string>& args);
    
    std::unordered_map<std::string, std::shared_ptr<Command>> commands_;
    bool enabled_ = true;
    std::function<bool(const std::string&)> confirmation_callback_;
    
    // Statistics
    struct Stats {
        size_t total_commands_executed = 0;
        size_t successful_executions = 0;
        size_t failed_executions = 0;
        std::unordered_map<std::string, size_t> command_counts;
    } stats_;
};

// Implementation
inline VoiceCommands::VoiceCommands() {
    register_builtin_commands();
}

inline void VoiceCommands::register_command(const std::string& name,
                                           const std::vector<std::string>& patterns,
                                           CommandHandler handler,
                                           const std::string& description,
                                           bool requires_confirmation) {
    auto cmd = std::make_shared<Command>(name, patterns, handler, description);
    cmd->requires_confirmation = requires_confirmation;
    commands_[name] = cmd;
}

inline VoiceCommands::ExecutionResult VoiceCommands::process_text(
    const std::string& text, float confidence) {
    
    ExecutionResult result;
    result.timestamp = std::chrono::steady_clock::now();
    result.confidence = confidence;
    
    if (!enabled_) {
        result.response = "Voice commands are disabled";
        return result;
    }
    
    // Convert to lowercase for matching
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    // Find matching command
    std::vector<std::string> matches;
    auto command = find_command(lower_text, matches);
    
    if (!command) {
        result.response = "No matching command found";
        return result;
    }
    
    result.command_name = command->name;
    
    // Check confidence threshold
    if (confidence * 100 < command->min_confidence) {
        result.response = "Confidence too low for command execution";
        return result;
    }
    
    // Request confirmation if needed
    if (command->requires_confirmation && confirmation_callback_) {
        std::string confirm_msg = "Execute command: " + command->name + "?";
        if (!confirmation_callback_(confirm_msg)) {
            result.response = "Command cancelled by user";
            return result;
        }
    }
    
    // Extract arguments
    std::regex pattern(command->patterns[0]);  // Use first matching pattern
    std::smatch smatch_result;
    std::regex_search(lower_text, smatch_result, pattern);
    auto args = extract_arguments(lower_text, pattern, smatch_result);
    
    // Execute command
    try {
        command->handler(args);
        result.success = true;
        result.response = "Command executed: " + command->name;
        
        stats_.total_commands_executed++;
        stats_.successful_executions++;
        stats_.command_counts[command->name]++;
    } catch (const std::exception& e) {
        result.response = "Command failed: " + std::string(e.what());
        stats_.failed_executions++;
    }
    
    return result;
}

inline std::shared_ptr<VoiceCommands::Command> VoiceCommands::find_command(
    const std::string& text, std::vector<std::string>& matches) {
    
    for (const auto& [name, command] : commands_) {
        for (const auto& pattern_str : command->patterns) {
            std::regex pattern(pattern_str, std::regex_constants::icase);
            std::smatch match_result;
            
            if (std::regex_search(text, match_result, pattern)) {
                matches.clear();
                for (const auto& match : match_result) {
                    matches.push_back(match.str());
                }
                return command;
            }
        }
    }
    
    return nullptr;
}

inline std::vector<std::string> VoiceCommands::extract_arguments(
    const std::string& text, const std::regex& pattern, const std::smatch& matches) {
    
    std::vector<std::string> args;
    
    // Skip the first match (full string) and collect capture groups
    for (size_t i = 1; i < matches.size(); ++i) {
        if (matches[i].matched) {
            args.push_back(matches[i].str());
        }
    }
    
    return args;
}

inline void VoiceCommands::register_builtin_commands() {
    // Recording control
    register_command("start_recording",
        {"start recording", "begin recording", "record"},
        [this](const auto& args) { handle_start_recording(args); },
        "Start audio recording");
    
    register_command("stop_recording",
        {"stop recording", "end recording", "stop"},
        [this](const auto& args) { handle_stop_recording(args); },
        "Stop audio recording");
    
    // Text management
    register_command("save_transcript",
        {"save transcript", "save text", "export transcript"},
        [this](const auto& args) { handle_save_transcript(args); },
        "Save current transcript to file");
    
    register_command("clear_text",
        {"clear text", "clear transcript", "clear all"},
        [this](const auto& args) { handle_clear_text(args); },
        "Clear all transcribed text",
        true);  // Requires confirmation
    
    // Language control
    register_command("switch_language",
        {"switch to (\\w+)", "change language to (\\w+)", "use (\\w+) language"},
        [this](const auto& args) { handle_switch_language(args); },
        "Switch transcription language");
    
    // Volume control
    register_command("increase_volume",
        {"increase volume", "volume up", "louder"},
        [this](const auto& args) { handle_increase_volume(args); },
        "Increase input volume");
    
    register_command("decrease_volume",
        {"decrease volume", "volume down", "quieter"},
        [this](const auto& args) { handle_decrease_volume(args); },
        "Decrease input volume");
    
    register_command("mute",
        {"mute", "mute microphone", "disable audio"},
        [this](const auto& args) { handle_mute(args); },
        "Mute microphone input");
    
    // Settings
    register_command("open_settings",
        {"open settings", "show settings", "preferences"},
        [this](const auto& args) { handle_settings(args); },
        "Open settings dialog");
    
    // Help
    register_command("help",
        {"help", "show commands", "what can you do"},
        [this](const auto& args) { handle_help(args); },
        "Show available commands");
}

inline std::vector<std::string> VoiceCommands::get_command_list() const {
    std::vector<std::string> list;
    for (const auto& [name, command] : commands_) {
        list.push_back(name + ": " + command->description);
    }
    return list;
}

inline std::string VoiceCommands::get_help() const {
    std::stringstream ss;
    ss << "Available Voice Commands:\n";
    ss << "========================\n\n";
    
    for (const auto& [name, command] : commands_) {
        ss << "• " << name << "\n";
        ss << "  Description: " << command->description << "\n";
        ss << "  Patterns: ";
        for (size_t i = 0; i < command->patterns.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << "\"" << command->patterns[i] << "\"";
        }
        ss << "\n";
        if (command->requires_confirmation) {
            ss << "  (Requires confirmation)\n";
        }
        ss << "\n";
    }
    
    return ss.str();
}

// Built-in command implementations
inline void VoiceCommands::handle_help(const std::vector<std::string>& args) {
    std::cout << get_help() << std::endl;
}

inline void VoiceCommands::handle_start_recording(const std::vector<std::string>& args) {
    std::cout << "[COMMAND] Starting recording..." << std::endl;
    // Implementation would trigger actual recording
}

inline void VoiceCommands::handle_stop_recording(const std::vector<std::string>& args) {
    std::cout << "[COMMAND] Stopping recording..." << std::endl;
    // Implementation would stop actual recording
}

inline void VoiceCommands::handle_save_transcript(const std::vector<std::string>& args) {
    std::cout << "[COMMAND] Saving transcript..." << std::endl;
    // Implementation would save to file
}

inline void VoiceCommands::handle_clear_text(const std::vector<std::string>& args) {
    std::cout << "[COMMAND] Clearing text..." << std::endl;
    // Implementation would clear transcript buffer
}

inline void VoiceCommands::handle_switch_language(const std::vector<std::string>& args) {
    if (!args.empty()) {
        std::cout << "[COMMAND] Switching language to: " << args[0] << std::endl;
        // Implementation would change whisper language setting
    }
}

inline void VoiceCommands::handle_increase_volume(const std::vector<std::string>& args) {
    std::cout << "[COMMAND] Increasing volume..." << std::endl;
    // Implementation would adjust audio input gain
}

inline void VoiceCommands::handle_decrease_volume(const std::vector<std::string>& args) {
    std::cout << "[COMMAND] Decreasing volume..." << std::endl;
    // Implementation would adjust audio input gain
}

inline void VoiceCommands::handle_mute(const std::vector<std::string>& args) {
    std::cout << "[COMMAND] Muting microphone..." << std::endl;
    // Implementation would mute audio input
}

inline void VoiceCommands::handle_settings(const std::vector<std::string>& args) {
    std::cout << "[COMMAND] Opening settings..." << std::endl;
    // Implementation would open settings dialog
}

}  // namespace vtt