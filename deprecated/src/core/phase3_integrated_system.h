#pragma once

#include "integrated_vtt_system.h"
#include "voice_commands.h"
#include "auto_correction.h"
#include "context_aware_processor.h"
#include "meeting_mode.h"
#include "offline_translation.h"
#include <memory>
#include <string>

namespace vtt {

// Phase 3 Integrated VTT System with all advanced features
class Phase3IntegratedSystem : public IntegratedVTTSystem {
public:
    struct Phase3Config : public IntegratedVTTSystem::Config {
        // Voice commands
        bool enable_voice_commands = true;
        float voice_command_confidence_threshold = 0.8f;
        
        // Auto-correction
        bool enable_auto_correction = true;
        bool learn_from_corrections = true;
        
        // Context awareness
        bool enable_context_awareness = true;
        ContextAwareProcessor::ContextType default_context = 
            ContextAwareProcessor::ContextType::GENERAL;
        
        // Meeting mode
        bool enable_meeting_mode = false;
        MeetingMode::Config meeting_config;
        
        // Translation
        bool enable_translation = false;
        OfflineTranslation::Language default_target_language = 
            OfflineTranslation::Language::ENGLISH;
    };
    
    Phase3IntegratedSystem(const Phase3Config& config = Phase3Config());
    ~Phase3IntegratedSystem();
    
    // Initialize Phase 3 features
    bool initialize_phase3();
    
    // Override process_audio to include Phase 3 features
    void process_audio_enhanced(const float* samples, size_t num_samples, 
                               uint32_t sample_rate);
    
    // Voice command control
    void enable_voice_commands(bool enable);
    void register_custom_command(const std::string& name,
                                const std::vector<std::string>& patterns,
                                VoiceCommands::CommandHandler handler);
    
    // Auto-correction control
    void enable_auto_correction(bool enable);
    void teach_correction(const std::string& original, const std::string& corrected);
    
    // Context control
    void set_context_type(ContextAwareProcessor::ContextType type);
    void add_custom_vocabulary(const std::string& domain,
                              const std::vector<std::string>& terms);
    
    // Meeting mode control
    void start_meeting(const std::string& title = "");
    MeetingMode::MeetingSummary end_meeting();
    void add_meeting_participant(const std::string& name, const std::string& role = "");
    
    // Translation control
    void enable_translation(bool enable);
    void set_target_language(OfflineTranslation::Language lang);
    std::string translate_last_transcription();
    
    // Get Phase 3 statistics
    struct Phase3Stats {
        VoiceCommands::ExecutionResult last_command;
        AutoCorrection::Stats correction_stats;
        ContextAwareProcessor::Stats context_stats;
        MeetingMode::Stats meeting_stats;
        OfflineTranslation::Stats translation_stats;
        
        Json::Value to_json() const {
            Json::Value stats;
            
            // Add command stats
            Json::Value cmd;
            cmd["last_command"] = last_command.command_name;
            cmd["success"] = last_command.success;
            stats["voice_commands"] = cmd;
            
            // Add correction stats
            stats["auto_correction"] = correction_stats.to_json();
            
            // Add context stats
            stats["context_awareness"] = context_stats.to_json();
            
            // Add meeting stats
            Json::Value meeting;
            meeting["total_words"] = static_cast<Json::Int64>(meeting_stats.total_words);
            meeting["speaker_changes"] = static_cast<Json::Int64>(meeting_stats.speaker_changes);
            meeting["action_items"] = static_cast<Json::Int64>(meeting_stats.action_items_count);
            meeting["decisions"] = static_cast<Json::Int64>(meeting_stats.decisions_count);
            stats["meeting_mode"] = meeting;
            
            // Add translation stats
            Json::Value trans;
            trans["total_translations"] = static_cast<Json::Int64>(translation_stats.total_translations);
            trans["successful"] = static_cast<Json::Int64>(translation_stats.successful_translations);
            stats["translation"] = trans;
            
            return stats;
        }
    };
    
    Phase3Stats get_phase3_stats() const;
    
    // Save/load Phase 3 models
    bool save_phase3_models(const std::string& directory);
    bool load_phase3_models(const std::string& directory);
    
    // Print comprehensive performance report
    void print_phase3_report() const;
    
private:
    // Process transcription through Phase 3 pipeline
    std::string apply_phase3_processing(const std::string& text);
    
    // Handle voice command in transcription
    bool check_and_execute_command(const std::string& text, float confidence);
    
    // Update all context processors
    void update_contexts(const std::string& text);
    
    Phase3Config phase3_config_;
    
    // Phase 3 components
    std::unique_ptr<VoiceCommands> voice_commands_;
    std::unique_ptr<AutoCorrection> auto_correction_;
    std::unique_ptr<ContextAwareProcessor> context_processor_;
    std::unique_ptr<MeetingMode> meeting_mode_;
    std::unique_ptr<OfflineTranslation> translator_;
    
    // State
    bool phase3_initialized_ = false;
    std::string last_transcription_;
    Phase3Stats phase3_stats_;
};

// Implementation
inline Phase3IntegratedSystem::Phase3IntegratedSystem(const Phase3Config& config)
    : IntegratedVTTSystem(config)
    , phase3_config_(config) {
    
    // Initialize Phase 3 components
    voice_commands_ = std::make_unique<VoiceCommands>();
    auto_correction_ = std::make_unique<AutoCorrection>();
    context_processor_ = std::make_unique<ContextAwareProcessor>();
    meeting_mode_ = std::make_unique<MeetingMode>(config.meeting_config);
    translator_ = std::make_unique<OfflineTranslation>();
}

inline Phase3IntegratedSystem::~Phase3IntegratedSystem() {
    if (phase3_initialized_) {
        save_phase3_models("phase3_models");
    }
    print_phase3_report();
}

inline bool Phase3IntegratedSystem::initialize_phase3() {
    std::cout << "\n=== Initializing Phase 3 Features ===\n";
    
    // Initialize base system first
    if (!initialize()) {
        std::cerr << "[ERROR] Failed to initialize base system\n";
        return false;
    }
    
    // Set up voice commands
    if (phase3_config_.enable_voice_commands) {
        voice_commands_->set_enabled(true);
        
        // Register VTT-specific commands
        voice_commands_->register_command("start_transcription",
            {"start transcription", "begin transcription", "start listening"},
            [this](const auto& args) { start_listening(); },
            "Start voice transcription");
        
        voice_commands_->register_command("stop_transcription",
            {"stop transcription", "end transcription", "stop listening"},
            [this](const auto& args) { stop_listening(); },
            "Stop voice transcription");
        
        voice_commands_->register_command("switch_language",
            {"switch to (\\w+)", "change language to (\\w+)"},
            [this](const auto& args) {
                if (!args.empty()) {
                    auto lang = OfflineTranslation::parse_language_code(args[0]);
                    set_target_language(lang);
                }
            },
            "Switch target language for translation");
        
        std::cout << "  ✓ Voice Commands initialized\n";
    }
    
    // Set up auto-correction
    if (phase3_config_.enable_auto_correction) {
        auto_correction_->set_enabled(true);
        auto_correction_->load_model("auto_corrections.json");
        std::cout << "  ✓ Auto-Correction initialized\n";
    }
    
    // Set up context awareness
    if (phase3_config_.enable_context_awareness) {
        context_processor_->set_context_type(phase3_config_.default_context);
        std::cout << "  ✓ Context-Aware Processing initialized\n";
    }
    
    // Set up translation
    if (phase3_config_.enable_translation) {
        translator_->set_enabled(true);
        std::cout << "  ✓ Offline Translation initialized\n";
    }
    
    phase3_initialized_ = true;
    
    std::cout << "\n=== Phase 3 Features Ready ===\n";
    std::cout << "Advanced capabilities enabled:\n";
    std::cout << "  • Voice Commands: " << (phase3_config_.enable_voice_commands ? "ON" : "OFF") << "\n";
    std::cout << "  • Auto-Correction: " << (phase3_config_.enable_auto_correction ? "ON" : "OFF") << "\n";
    std::cout << "  • Context Awareness: " << (phase3_config_.enable_context_awareness ? "ON" : "OFF") << "\n";
    std::cout << "  • Meeting Mode: " << (phase3_config_.enable_meeting_mode ? "ON" : "OFF") << "\n";
    std::cout << "  • Translation: " << (phase3_config_.enable_translation ? "ON" : "OFF") << "\n";
    std::cout << "\n";
    
    return true;
}

inline void Phase3IntegratedSystem::process_audio_enhanced(const float* samples, 
                                                          size_t num_samples,
                                                          uint32_t sample_rate) {
    // Process through base system first
    process_audio(samples, num_samples, sample_rate);
    
    // Get the latest transcription
    // (In production, this would be passed through a callback)
    std::string transcription = last_transcription_;
    
    if (!transcription.empty()) {
        // Apply Phase 3 processing
        transcription = apply_phase3_processing(transcription);
        
        // Check for voice commands
        if (phase3_config_.enable_voice_commands) {
            check_and_execute_command(transcription, 0.9f);
        }
        
        // Process in meeting mode if active
        if (phase3_config_.enable_meeting_mode && meeting_mode_) {
            meeting_mode_->process_audio_segment(samples, num_samples, 
                                                sample_rate, transcription);
        }
        
        // Update last transcription
        last_transcription_ = transcription;
    }
}

inline std::string Phase3IntegratedSystem::apply_phase3_processing(const std::string& text) {
    std::string processed = text;
    
    // Apply context-aware processing
    if (phase3_config_.enable_context_awareness && context_processor_) {
        processed = context_processor_->process_with_context(processed);
        update_contexts(processed);
    }
    
    // Apply auto-correction
    if (phase3_config_.enable_auto_correction && auto_correction_) {
        processed = auto_correction_->apply_corrections(processed);
    }
    
    // Apply translation if enabled
    if (phase3_config_.enable_translation && translator_) {
        auto detected_lang = translator_->detect_language(processed);
        if (detected_lang != phase3_config_.default_target_language) {
            processed = translator_->translate(processed, 
                                              phase3_config_.default_target_language,
                                              detected_lang);
        }
    }
    
    return processed;
}

inline bool Phase3IntegratedSystem::check_and_execute_command(const std::string& text, 
                                                             float confidence) {
    if (!voice_commands_) return false;
    
    auto result = voice_commands_->process_text(text, confidence);
    phase3_stats_.last_command = result;
    
    if (result.success) {
        std::cout << "[COMMAND] Executed: " << result.command_name << "\n";
        return true;
    }
    
    return false;
}

inline void Phase3IntegratedSystem::update_contexts(const std::string& text) {
    if (context_processor_) {
        context_processor_->update_context(text);
    }
}

inline void Phase3IntegratedSystem::enable_voice_commands(bool enable) {
    phase3_config_.enable_voice_commands = enable;
    if (voice_commands_) {
        voice_commands_->set_enabled(enable);
    }
}

inline void Phase3IntegratedSystem::register_custom_command(const std::string& name,
                                                           const std::vector<std::string>& patterns,
                                                           VoiceCommands::CommandHandler handler) {
    if (voice_commands_) {
        voice_commands_->register_command(name, patterns, handler);
    }
}

inline void Phase3IntegratedSystem::enable_auto_correction(bool enable) {
    phase3_config_.enable_auto_correction = enable;
    if (auto_correction_) {
        auto_correction_->set_enabled(enable);
    }
}

inline void Phase3IntegratedSystem::teach_correction(const std::string& original,
                                                    const std::string& corrected) {
    if (auto_correction_ && phase3_config_.learn_from_corrections) {
        auto_correction_->learn_correction(original, corrected);
    }
}

inline void Phase3IntegratedSystem::set_context_type(ContextAwareProcessor::ContextType type) {
    phase3_config_.default_context = type;
    if (context_processor_) {
        context_processor_->set_context_type(type);
    }
}

inline void Phase3IntegratedSystem::add_custom_vocabulary(const std::string& domain,
                                                         const std::vector<std::string>& terms) {
    if (context_processor_) {
        context_processor_->add_domain_vocabulary(domain, terms);
    }
}

inline void Phase3IntegratedSystem::start_meeting(const std::string& title) {
    phase3_config_.enable_meeting_mode = true;
    if (meeting_mode_) {
        meeting_mode_->start_meeting(title);
        // Switch to business context for meetings
        set_context_type(ContextAwareProcessor::ContextType::BUSINESS);
    }
}

inline MeetingMode::MeetingSummary Phase3IntegratedSystem::end_meeting() {
    phase3_config_.enable_meeting_mode = false;
    if (meeting_mode_) {
        auto summary = meeting_mode_->end_meeting();
        // Switch back to general context
        set_context_type(ContextAwareProcessor::ContextType::GENERAL);
        return summary;
    }
    return MeetingMode::MeetingSummary();
}

inline void Phase3IntegratedSystem::add_meeting_participant(const std::string& name,
                                                          const std::string& role) {
    if (meeting_mode_) {
        meeting_mode_->add_participant(name, role);
    }
}

inline void Phase3IntegratedSystem::enable_translation(bool enable) {
    phase3_config_.enable_translation = enable;
    if (translator_) {
        translator_->set_enabled(enable);
    }
}

inline void Phase3IntegratedSystem::set_target_language(OfflineTranslation::Language lang) {
    phase3_config_.default_target_language = lang;
}

inline std::string Phase3IntegratedSystem::translate_last_transcription() {
    if (translator_ && !last_transcription_.empty()) {
        return translator_->translate(last_transcription_,
                                     phase3_config_.default_target_language);
    }
    return last_transcription_;
}

inline Phase3IntegratedSystem::Phase3Stats Phase3IntegratedSystem::get_phase3_stats() const {
    Phase3Stats stats = phase3_stats_;
    
    if (auto_correction_) {
        stats.correction_stats = auto_correction_->get_stats();
    }
    if (context_processor_) {
        stats.context_stats = context_processor_->get_stats();
    }
    if (meeting_mode_) {
        stats.meeting_stats = meeting_mode_->get_stats();
    }
    if (translator_) {
        stats.translation_stats = translator_->get_stats();
    }
    
    return stats;
}

inline bool Phase3IntegratedSystem::save_phase3_models(const std::string& directory) {
    bool success = true;
    
    if (auto_correction_) {
        success &= auto_correction_->save_model(directory + "/auto_corrections.json");
    }
    
    if (context_processor_) {
        success &= context_processor_->save_context_model(directory + "/context_model.json");
    }
    
    return success;
}

inline bool Phase3IntegratedSystem::load_phase3_models(const std::string& directory) {
    bool success = true;
    
    if (auto_correction_) {
        success &= auto_correction_->load_model(directory + "/auto_corrections.json");
    }
    
    if (context_processor_) {
        success &= context_processor_->load_context_model(directory + "/context_model.json");
    }
    
    return success;
}

inline void Phase3IntegratedSystem::print_phase3_report() const {
    std::cout << "\n=== Phase 3 Integrated System Performance Report ===\n";
    
    // Base system report
    print_performance_report();
    
    // Phase 3 specific stats
    auto stats = get_phase3_stats();
    
    std::cout << "\nPhase 3 Features:\n";
    
    // Voice commands
    std::cout << "\nVoice Commands:\n";
    std::cout << "  Last command: " << stats.last_command.command_name << "\n";
    std::cout << "  Success: " << (stats.last_command.success ? "Yes" : "No") << "\n";
    
    // Auto-correction
    std::cout << "\nAuto-Correction:\n";
    std::cout << "  Total corrections: " << stats.correction_stats.total_corrections << "\n";
    std::cout << "  Unique corrections: " << stats.correction_stats.unique_corrections << "\n";
    std::cout << "  Applied: " << stats.correction_stats.corrections_applied << "\n";
    
    // Context awareness
    std::cout << "\nContext Awareness:\n";
    std::cout << "  Processed: " << stats.context_stats.total_processed << "\n";
    std::cout << "  Context switches: " << stats.context_stats.context_switches << "\n";
    
    // Meeting mode
    if (phase3_config_.enable_meeting_mode) {
        std::cout << "\nMeeting Mode:\n";
        std::cout << "  Total words: " << stats.meeting_stats.total_words << "\n";
        std::cout << "  Speaker changes: " << stats.meeting_stats.speaker_changes << "\n";
        std::cout << "  Action items: " << stats.meeting_stats.action_items_count << "\n";
        std::cout << "  Decisions: " << stats.meeting_stats.decisions_count << "\n";
    }
    
    // Translation
    if (phase3_config_.enable_translation) {
        std::cout << "\nTranslation:\n";
        std::cout << "  Total: " << stats.translation_stats.total_translations << "\n";
        std::cout << "  Successful: " << stats.translation_stats.successful_translations << "\n";
        std::cout << "  Dictionary fallbacks: " << stats.translation_stats.dictionary_fallbacks << "\n";
    }
    
    std::cout << "\n=================================================\n";
}

}  // namespace vtt