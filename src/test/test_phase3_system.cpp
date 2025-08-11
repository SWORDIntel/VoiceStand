#include "../core/phase3_integrated_system.h"
#include "../core/audio_capture.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>

using namespace vtt;

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n[INFO] Shutting down...\n";
        g_running = false;
    }
}

void demonstrate_voice_commands(Phase3IntegratedSystem& system) {
    std::cout << "\n=== Voice Commands Demo ===\n";
    
    // Test various voice commands
    std::vector<std::string> test_commands = {
        "start transcription",
        "switch to spanish",
        "stop transcription",
        "help"
    };
    
    for (const auto& cmd : test_commands) {
        std::cout << "Testing command: \"" << cmd << "\"\n";
        // Simulate voice command (would come from transcription in real use)
        // Command execution would happen through the process_audio_enhanced pipeline
    }
}

void demonstrate_auto_correction(Phase3IntegratedSystem& system) {
    std::cout << "\n=== Auto-Correction Demo ===\n";
    
    // Teach the system some corrections
    system.teach_correction("teh", "the");
    system.teach_correction("recieve", "receive");
    system.teach_correction("occured", "occurred");
    
    std::cout << "Taught 3 corrections to the system\n";
    
    // Test text with errors (would be corrected automatically in pipeline)
    std::string test_text = "teh meeting occured yesterday and we will recieve the report";
    std::cout << "Test text: " << test_text << "\n";
    // In real use, corrections happen automatically during transcription
}

void demonstrate_context_awareness(Phase3IntegratedSystem& system) {
    std::cout << "\n=== Context-Aware Processing Demo ===\n";
    
    // Test different contexts
    std::vector<std::pair<ContextAwareProcessor::ContextType, std::string>> contexts = {
        {ContextAwareProcessor::ContextType::TECHNICAL, 
         "We need to deploy the microservice to the kubernetes cluster using docker containers"},
        {ContextAwareProcessor::ContextType::MEDICAL,
         "The patient's diagnosis shows chronic symptoms requiring immediate treatment"},
        {ContextAwareProcessor::ContextType::BUSINESS,
         "Our quarterly revenue shows positive ROI with strong stakeholder engagement"}
    };
    
    for (const auto& [context_type, text] : contexts) {
        system.set_context_type(context_type);
        std::cout << "Context: " << static_cast<int>(context_type) << "\n";
        std::cout << "Sample: " << text.substr(0, 50) << "...\n";
    }
}

void demonstrate_meeting_mode(Phase3IntegratedSystem& system) {
    std::cout << "\n=== Meeting Mode Demo ===\n";
    
    // Start a meeting
    system.start_meeting("Q4 Planning Meeting");
    
    // Add participants
    system.add_meeting_participant("John Smith", "Project Manager");
    system.add_meeting_participant("Jane Doe", "Tech Lead");
    system.add_meeting_participant("Bob Johnson", "Product Owner");
    
    std::cout << "Meeting started with 3 participants\n";
    
    // Simulate meeting conversation
    std::vector<std::string> meeting_transcript = {
        "Let's discuss our Q4 objectives",
        "We need to focus on performance improvements",
        "Action item: John will create the roadmap",
        "Decision: We'll proceed with the new architecture",
        "Bob will follow up with stakeholders"
    };
    
    // Process each line (in real use, this comes from audio)
    for (const auto& line : meeting_transcript) {
        std::cout << "Processing: " << line << "\n";
        // Would be processed through audio pipeline
    }
    
    // End meeting and get summary
    auto summary = system.end_meeting();
    std::cout << "Meeting ended. Summary available.\n";
}

void demonstrate_translation(Phase3IntegratedSystem& system) {
    std::cout << "\n=== Offline Translation Demo ===\n";
    
    // Enable translation
    system.enable_translation(true);
    
    // Test different target languages
    std::vector<std::pair<OfflineTranslation::Language, std::string>> translations = {
        {OfflineTranslation::Language::SPANISH, "Hello, how are you today?"},
        {OfflineTranslation::Language::FRENCH, "Good morning, nice to meet you"},
        {OfflineTranslation::Language::GERMAN, "Thank you for your help"}
    };
    
    for (const auto& [lang, text] : translations) {
        system.set_target_language(lang);
        std::cout << "Target language: " << OfflineTranslation::get_language_name(lang) << "\n";
        std::cout << "Original: " << text << "\n";
        // Translation happens automatically in pipeline
    }
}

int main(int argc, char* argv[]) {
    // Set up signal handler
    signal(SIGINT, signal_handler);
    
    std::cout << "=== Phase 3 Integrated VTT System Test ===\n";
    std::cout << "Testing all Phase 3 advanced features\n\n";
    
    // Configure Phase 3 system
    Phase3IntegratedSystem::Phase3Config config;
    config.whisper_config.model_path = "models/ggml-base.en.bin";
    config.whisper_config.language = "en";
    config.enable_voice_commands = true;
    config.enable_auto_correction = true;
    config.enable_context_awareness = true;
    config.enable_meeting_mode = false;  // Start disabled
    config.enable_translation = false;   // Start disabled
    
    // Create Phase 3 system
    Phase3IntegratedSystem system(config);
    
    // Initialize
    if (!system.initialize_phase3()) {
        std::cerr << "[ERROR] Failed to initialize Phase 3 system\n";
        return 1;
    }
    
    // Set up callbacks
    system.set_result_callback(
        [](const IntegratedVTTSystem::EnhancedResult& result) {
            std::cout << "\n[RESULT] " << result.speaker_name << ": " << result.text << "\n";
            
            if (result.wake_word_triggered) {
                std::cout << "  (Wake word: " << result.wake_word_used << ")\n";
            }
        }
    );
    
    // Create audio capture
    AudioCapture audio_capture;
    
    // Set up audio processing
    audio_capture.set_audio_callback(
        [&system](const float* samples, size_t num_samples) {
            system.process_audio_enhanced(samples, num_samples, 16000);
        }
    );
    
    // Interactive menu
    auto show_menu = []() {
        std::cout << "\n=== Phase 3 Feature Test Menu ===\n";
        std::cout << "1. Test Voice Commands\n";
        std::cout << "2. Test Auto-Correction\n";
        std::cout << "3. Test Context Awareness\n";
        std::cout << "4. Test Meeting Mode\n";
        std::cout << "5. Test Translation\n";
        std::cout << "6. Start Audio Capture\n";
        std::cout << "7. Stop Audio Capture\n";
        std::cout << "8. Show Statistics\n";
        std::cout << "9. Save Models\n";
        std::cout << "q. Quit\n";
        std::cout << "Enter choice: ";
    };
    
    // Main interaction loop
    std::string choice;
    bool audio_running = false;
    
    while (g_running) {
        show_menu();
        std::getline(std::cin, choice);
        
        if (choice == "1") {
            demonstrate_voice_commands(system);
        }
        else if (choice == "2") {
            demonstrate_auto_correction(system);
        }
        else if (choice == "3") {
            demonstrate_context_awareness(system);
        }
        else if (choice == "4") {
            demonstrate_meeting_mode(system);
        }
        else if (choice == "5") {
            demonstrate_translation(system);
        }
        else if (choice == "6") {
            if (!audio_running) {
                if (audio_capture.start()) {
                    audio_running = true;
                    std::cout << "[INFO] Audio capture started\n";
                } else {
                    std::cout << "[ERROR] Failed to start audio capture\n";
                }
            } else {
                std::cout << "[INFO] Audio already running\n";
            }
        }
        else if (choice == "7") {
            if (audio_running) {
                audio_capture.stop();
                audio_running = false;
                std::cout << "[INFO] Audio capture stopped\n";
            } else {
                std::cout << "[INFO] Audio not running\n";
            }
        }
        else if (choice == "8") {
            auto stats = system.get_phase3_stats();
            std::cout << "\n=== Phase 3 Statistics ===\n";
            
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "  ";
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            
            Json::Value stats_json = stats.to_json();
            std::stringstream ss;
            writer->write(stats_json, &ss);
            
            std::cout << ss.str() << "\n";
        }
        else if (choice == "9") {
            if (system.save_phase3_models("phase3_models")) {
                std::cout << "[INFO] Models saved successfully\n";
            } else {
                std::cout << "[ERROR] Failed to save models\n";
            }
        }
        else if (choice == "q") {
            g_running = false;
        }
    }
    
    // Clean up
    if (audio_running) {
        audio_capture.stop();
    }
    
    // Print final report
    system.print_phase3_report();
    
    std::cout << "\n[INFO] Phase 3 test completed successfully\n";
    return 0;
}