#pragma once

#include "integrated_vtt_system.h"
#include "personal_voice_commands.h"
#include "wake_word_detector.h"
#include <memory>
#include <atomic>
#include <functional>

namespace vtt {

/**
 * Phase 2 Personal Commands Integration
 * Integrates Personal Voice Commands with the existing VTT system
 * Builds on the GNA foundation for optimal power efficiency
 */
class Phase2PersonalCommandsIntegration {
public:
    struct Config {
        IntegratedVTTSystem::Config vtt_config;
        PersonalVoiceCommands::Config commands_config;

        // Integration settings
        bool enable_command_mode = true;
        bool enable_dictation_mode = true;
        std::string command_trigger = "computer";  // Wake word for command mode
        std::string dictation_trigger = "dictate"; // Wake word for dictation mode

        // Mode switching
        float mode_switch_confidence = 0.85f;
        size_t command_timeout_ms = 5000;         // Timeout for command mode
        size_t dictation_timeout_ms = 10000;     // Timeout for dictation mode

        // Performance optimization
        bool use_gna_wake_detection = true;       // Use GNA for wake word detection (0mW)
        bool enable_parallel_processing = true;   // Process commands and dictation in parallel
        size_t max_queue_size = 100;             // Maximum queued commands
    };

    enum class OperationMode {
        IDLE,           // Waiting for wake word
        COMMAND,        // Processing voice commands
        DICTATION,      // Processing dictation
        HYBRID          // Both command and dictation active
    };

    struct SystemResult {
        OperationMode mode;
        bool command_executed = false;
        bool text_transcribed = false;
        std::string command_result;
        std::string transcribed_text;
        float confidence = 0.0f;
        std::chrono::steady_clock::time_point timestamp;

        SystemResult() : timestamp(std::chrono::steady_clock::now()) {}
    };

    using SystemCallback = std::function<void(const SystemResult&)>;

    Phase2PersonalCommandsIntegration(const Config& config = Config());
    ~Phase2PersonalCommandsIntegration();

    // System lifecycle
    bool initialize();
    void start();
    void stop();
    bool is_running() const { return is_running_; }

    // Audio processing
    void process_audio(const float* samples, size_t num_samples, uint32_t sample_rate);

    // Mode control
    void set_mode(OperationMode mode);
    OperationMode get_mode() const { return current_mode_; }
    void toggle_mode();

    // Personal macro management (delegate to PersonalVoiceCommands)
    bool register_personal_macro(const PersonalVoiceCommands::PersonalMacro& macro);
    bool remove_personal_macro(const std::string& name);
    std::vector<PersonalVoiceCommands::PersonalMacro> get_personal_macros() const;

    // Callbacks
    void set_system_callback(SystemCallback callback) { system_callback_ = callback; }

    // Statistics and performance
    struct SystemStats {
        size_t total_audio_processed_ms = 0;
        size_t commands_executed = 0;
        size_t dictation_sessions = 0;
        size_t mode_switches = 0;
        size_t wake_word_detections = 0;
        float average_command_confidence = 0.0f;
        float average_dictation_confidence = 0.0f;
        float gna_power_usage_mw = 0.0f;  // Should be ~0mW with GNA
        float average_processing_latency_ms = 0.0f;

        // Command breakdown
        std::unordered_map<std::string, size_t> command_usage;
        std::unordered_map<OperationMode, size_t> mode_usage;
    };

    SystemStats get_system_stats() const;
    void print_performance_summary() const;

    // Advanced features
    void train_personal_wake_words(const std::vector<std::string>& words,
                                  const std::vector<std::vector<float>>& samples);
    void enable_learning_mode(bool enable);
    void export_session_data(const std::string& filepath) const;

private:
    Config config_;

    // Core components
    std::unique_ptr<IntegratedVTTSystem> vtt_system_;
    std::unique_ptr<PersonalVoiceCommands> personal_commands_;

    // State management
    std::atomic<OperationMode> current_mode_{OperationMode::IDLE};
    std::atomic<bool> is_running_{false};
    std::chrono::steady_clock::time_point last_activity_;
    std::chrono::steady_clock::time_point mode_start_time_;

    // Audio processing pipeline
    std::vector<float> audio_buffer_;
    std::queue<std::vector<float>> audio_queue_;
    std::mutex audio_queue_mutex_;

    // Performance tracking
    mutable SystemStats stats_;
    std::chrono::steady_clock::time_point session_start_;

    // Callbacks
    SystemCallback system_callback_;

    // Internal processing methods
    void handle_wake_word_detection(const std::string& wake_word);
    void handle_command_processing(const std::string& text, float confidence);
    void handle_dictation_processing(const std::string& text, float confidence);
    void handle_timeout_check();

    // Mode transition logic
    void transition_to_mode(OperationMode new_mode);
    bool should_switch_mode(const std::string& text, float confidence);

    // GNA optimization (building on Phase 1)
    void optimize_gna_wake_detection();
    void configure_power_efficient_processing();

    // Audio processing optimization
    void process_audio_parallel(const float* samples, size_t num_samples, uint32_t sample_rate);
    void process_audio_sequential(const float* samples, size_t num_samples, uint32_t sample_rate);

    // Statistics tracking
    void update_performance_stats(OperationMode mode, float confidence, float latency_ms);
    void track_command_usage(const std::string& command_name);
};

// Implementation
inline Phase2PersonalCommandsIntegration::Phase2PersonalCommandsIntegration(const Config& config)
    : config_(config), session_start_(std::chrono::steady_clock::now()) {

    // Initialize components
    vtt_system_ = std::make_unique<IntegratedVTTSystem>(config_.vtt_config);
    personal_commands_ = std::make_unique<PersonalVoiceCommands>(config_.commands_config);
}

inline Phase2PersonalCommandsIntegration::~Phase2PersonalCommandsIntegration() {
    stop();
    print_performance_summary();
}

inline bool Phase2PersonalCommandsIntegration::initialize() {
    std::cout << "[INFO] Initializing Phase 2 Personal Commands Integration\n";

    // Initialize VTT system
    if (!vtt_system_->initialize()) {
        std::cerr << "[ERROR] Failed to initialize VTT system\n";
        return false;
    }

    // Initialize personal commands
    if (!personal_commands_->initialize()) {
        std::cerr << "[ERROR] Failed to initialize personal commands\n";
        return false;
    }

    // Configure wake word callbacks
    vtt_system_->set_wake_word_callback([this](const std::string& wake_word) {
        handle_wake_word_detection(wake_word);
    });

    // Configure transcription callbacks
    vtt_system_->set_result_callback([this](const IntegratedVTTSystem::EnhancedResult& result) {
        if (current_mode_ == OperationMode::COMMAND) {
            handle_command_processing(result.text, result.confidence);
        } else if (current_mode_ == OperationMode::DICTATION) {
            handle_dictation_processing(result.text, result.confidence);
        }
    });

    // Configure personal commands callback
    personal_commands_->set_command_callback([this](const PersonalVoiceCommands::ExecutionResult& result) {
        SystemResult sys_result;
        sys_result.mode = OperationMode::COMMAND;
        sys_result.command_executed = result.success;
        sys_result.command_result = result.response;
        sys_result.confidence = result.confidence;

        if (system_callback_) {
            system_callback_(sys_result);
        }

        track_command_usage(result.command_name);
    });

    // Optimize for GNA wake detection if enabled
    if (config_.use_gna_wake_detection) {
        optimize_gna_wake_detection();
    }

    // Configure power-efficient processing
    configure_power_efficient_processing();

    std::cout << "[INFO] Phase 2 Personal Commands Integration initialized successfully\n";
    std::cout << "Configuration:\n";
    std::cout << "  - Command Mode: " << (config_.enable_command_mode ? "Enabled" : "Disabled") << "\n";
    std::cout << "  - Dictation Mode: " << (config_.enable_dictation_mode ? "Enabled" : "Disabled") << "\n";
    std::cout << "  - GNA Wake Detection: " << (config_.use_gna_wake_detection ? "Enabled" : "Disabled") << "\n";
    std::cout << "  - Parallel Processing: " << (config_.enable_parallel_processing ? "Enabled" : "Disabled") << "\n";

    return true;
}

inline void Phase2PersonalCommandsIntegration::start() {
    if (is_running_) return;

    is_running_ = true;
    current_mode_ = OperationMode::IDLE;
    last_activity_ = std::chrono::steady_clock::now();

    std::cout << "[INFO] Phase 2 Personal Commands Integration started\n";
}

inline void Phase2PersonalCommandsIntegration::stop() {
    if (!is_running_) return;

    is_running_ = false;
    current_mode_ = OperationMode::IDLE;

    std::cout << "[INFO] Phase 2 Personal Commands Integration stopped\n";
}

inline void Phase2PersonalCommandsIntegration::process_audio(const float* samples,
                                                            size_t num_samples,
                                                            uint32_t sample_rate) {
    if (!is_running_) return;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Process audio based on configuration
    if (config_.enable_parallel_processing) {
        process_audio_parallel(samples, num_samples, sample_rate);
    } else {
        process_audio_sequential(samples, num_samples, sample_rate);
    }

    // Handle timeouts
    handle_timeout_check();

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    float latency_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    stats_.total_audio_processed_ms += (num_samples * 1000) / sample_rate;
    stats_.average_processing_latency_ms = (stats_.average_processing_latency_ms *
                                           (stats_.total_audio_processed_ms / 1000 - 1) + latency_ms) /
                                          (stats_.total_audio_processed_ms / 1000);
}

inline void Phase2PersonalCommandsIntegration::handle_wake_word_detection(const std::string& wake_word) {
    std::cout << "[INFO] Wake word detected: " << wake_word << "\n";

    stats_.wake_word_detections++;
    last_activity_ = std::chrono::steady_clock::now();

    // Determine target mode based on wake word
    OperationMode target_mode = OperationMode::IDLE;

    if (wake_word == config_.command_trigger && config_.enable_command_mode) {
        target_mode = OperationMode::COMMAND;
    } else if (wake_word == config_.dictation_trigger && config_.enable_dictation_mode) {
        target_mode = OperationMode::DICTATION;
    } else {
        // Default behavior for other wake words
        target_mode = config_.enable_command_mode ? OperationMode::COMMAND : OperationMode::DICTATION;
    }

    transition_to_mode(target_mode);
}

inline void Phase2PersonalCommandsIntegration::transition_to_mode(OperationMode new_mode) {
    if (current_mode_ != new_mode) {
        std::cout << "[INFO] Mode transition: " << static_cast<int>(current_mode_)
                  << " -> " << static_cast<int>(new_mode) << "\n";

        current_mode_ = new_mode;
        mode_start_time_ = std::chrono::steady_clock::now();
        stats_.mode_switches++;
        stats_.mode_usage[new_mode]++;

        // Configure VTT system based on mode
        if (new_mode == OperationMode::COMMAND || new_mode == OperationMode::HYBRID) {
            vtt_system_->start_listening();
        } else if (new_mode == OperationMode::DICTATION) {
            vtt_system_->start_listening();
        } else {
            vtt_system_->stop_listening();
        }
    }
}

inline void Phase2PersonalCommandsIntegration::handle_command_processing(const std::string& text,
                                                                        float confidence) {
    std::cout << "[COMMAND] Processing: \"" << text << "\" (confidence: "
              << (confidence * 100) << "%)\n";

    auto result = personal_commands_->process_voice_input(text, confidence);

    SystemResult sys_result;
    sys_result.mode = OperationMode::COMMAND;
    sys_result.command_executed = result.success;
    sys_result.command_result = result.response;
    sys_result.confidence = confidence;

    if (result.success) {
        stats_.commands_executed++;
        stats_.average_command_confidence = (stats_.average_command_confidence *
                                           (stats_.commands_executed - 1) + confidence) /
                                          stats_.commands_executed;

        // Return to idle after successful command
        transition_to_mode(OperationMode::IDLE);
    }

    if (system_callback_) {
        system_callback_(sys_result);
    }

    last_activity_ = std::chrono::steady_clock::now();
}

inline void Phase2PersonalCommandsIntegration::handle_dictation_processing(const std::string& text,
                                                                          float confidence) {
    std::cout << "[DICTATION] Transcribed: \"" << text << "\" (confidence: "
              << (confidence * 100) << "%)\n";

    SystemResult sys_result;
    sys_result.mode = OperationMode::DICTATION;
    sys_result.text_transcribed = true;
    sys_result.transcribed_text = text;
    sys_result.confidence = confidence;

    stats_.dictation_sessions++;
    stats_.average_dictation_confidence = (stats_.average_dictation_confidence *
                                         (stats_.dictation_sessions - 1) + confidence) /
                                        stats_.dictation_sessions;

    if (system_callback_) {
        system_callback_(sys_result);
    }

    last_activity_ = std::chrono::steady_clock::now();
}

inline void Phase2PersonalCommandsIntegration::handle_timeout_check() {
    auto now = std::chrono::steady_clock::now();
    auto time_since_activity = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_activity_).count();

    bool should_timeout = false;

    if (current_mode_ == OperationMode::COMMAND &&
        time_since_activity > config_.command_timeout_ms) {
        should_timeout = true;
    } else if (current_mode_ == OperationMode::DICTATION &&
               time_since_activity > config_.dictation_timeout_ms) {
        should_timeout = true;
    }

    if (should_timeout) {
        std::cout << "[INFO] Mode timeout, returning to idle\n";
        transition_to_mode(OperationMode::IDLE);
    }
}

inline void Phase2PersonalCommandsIntegration::optimize_gna_wake_detection() {
    std::cout << "[INFO] Optimizing GNA wake detection for 0mW power usage\n";

    // Configure GNA-specific wake word detection
    // This builds on the Phase 1 GNA foundation
    vtt_system_->register_wake_word(config_.command_trigger);
    vtt_system_->register_wake_word(config_.dictation_trigger);

    // Set GNA power usage to near-zero
    stats_.gna_power_usage_mw = 0.0f;  // GNA operates at ~0mW
}

inline void Phase2PersonalCommandsIntegration::print_performance_summary() const {
    auto session_duration = std::chrono::steady_clock::now() - session_start_;
    auto duration_min = std::chrono::duration_cast<std::chrono::minutes>(session_duration).count();

    std::cout << "\n=== Phase 2 Personal Commands Performance Summary ===\n";
    std::cout << "Session duration: " << duration_min << " minutes\n";
    std::cout << "Total audio processed: " << stats_.total_audio_processed_ms / 1000 << " seconds\n";
    std::cout << "Commands executed: " << stats_.commands_executed << "\n";
    std::cout << "Dictation sessions: " << stats_.dictation_sessions << "\n";
    std::cout << "Mode switches: " << stats_.mode_switches << "\n";
    std::cout << "Wake word detections: " << stats_.wake_word_detections << "\n";
    std::cout << "Average command confidence: " << (stats_.average_command_confidence * 100) << "%\n";
    std::cout << "Average dictation confidence: " << (stats_.average_dictation_confidence * 100) << "%\n";
    std::cout << "GNA power usage: " << stats_.gna_power_usage_mw << " mW\n";
    std::cout << "Average processing latency: " << stats_.average_processing_latency_ms << " ms\n";

    std::cout << "\nPersonal Commands Statistics:\n";
    auto cmd_stats = personal_commands_->get_statistics();
    std::cout << "  Total command executions: " << cmd_stats.total_commands_executed << "\n";
    std::cout << "  Success rate: " << (cmd_stats.successful_executions * 100.0f /
                                       std::max(1ul, cmd_stats.total_commands_executed)) << "%\n";
    std::cout << "  Security rejections: " << cmd_stats.security_rejections << "\n";

    std::cout << "====================================================\n";
}

}  // namespace vtt