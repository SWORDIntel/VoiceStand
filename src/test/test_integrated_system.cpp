#include "../core/integrated_vtt_system.h"
#include "../core/audio_capture.h"
#include "../gui/main_window.h"
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

int main(int argc, char* argv[]) {
    // Set up signal handler
    signal(SIGINT, signal_handler);
    
    std::cout << "=== Integrated VTT System Test ===\n";
    std::cout << "Testing all Phase 1 and Phase 2 features\n\n";
    
    // Configure the integrated system
    IntegratedVTTSystem::Config config;
    config.whisper_config.model_path = "models/ggml-base.en.bin";
    config.whisper_config.language = "en";
    config.whisper_config.use_gpu = false;
    config.enable_wake_words = true;
    config.enable_noise_cancellation = true;
    config.enable_speaker_diarization = true;
    config.enable_punctuation = true;
    config.continuous_listening = false;
    
    // Create integrated system
    IntegratedVTTSystem vtt_system(config);
    
    // Initialize system
    if (!vtt_system.initialize()) {
        std::cerr << "[ERROR] Failed to initialize VTT system\n";
        return 1;
    }
    
    // Set up result callback
    vtt_system.set_result_callback(
        [](const IntegratedVTTSystem::EnhancedResult& result) {
            std::cout << "\n=== Transcription Result ===\n";
            std::cout << "Text: " << result.text << "\n";
            std::cout << "Speaker: " << result.speaker_name << "\n";
            std::cout << "Confidence: " << (result.confidence * 100) << "%\n";
            
            if (result.wake_word_triggered) {
                std::cout << "Wake word detected: " << result.wake_word_used << "\n";
            }
            
            if (result.noise_reduction_db > 0) {
                std::cout << "Noise reduced: " << result.noise_reduction_db << " dB\n";
            }
            
            std::cout << "===========================\n";
        }
    );
    
    // Set up wake word callback
    vtt_system.set_wake_word_callback(
        [&vtt_system](const std::string& word) {
            std::cout << "\n[WAKE] Wake word detected: \"" << word << "\"\n";
            std::cout << "[INFO] Starting transcription...\n";
        }
    );
    
    // Register custom wake words
    vtt_system.register_wake_word("hey computer");
    vtt_system.register_wake_word("okay system");
    vtt_system.register_wake_word("start recording");
    
    // Add speaker profiles (if available)
    std::cout << "[INFO] System initialized with:\n";
    std::cout << "  - Wake words: hey computer, okay system, start recording\n";
    std::cout << "  - Noise cancellation: ENABLED\n";
    std::cout << "  - Speaker diarization: ENABLED\n";
    std::cout << "  - Punctuation restoration: ENABLED\n";
    std::cout << "\n";
    
    // Create audio capture
    AudioCapture audio_capture;
    
    // Set up audio processing callback
    audio_capture.set_audio_callback(
        [&vtt_system](const AudioData& audio_data) {
            vtt_system.process_audio(audio_data.samples, audio_data.num_samples, audio_data.sample_rate);
        }
    );
    
    // Start audio capture
    if (!audio_capture.start()) {
        std::cerr << "[ERROR] Failed to start audio capture\n";
        return 1;
    }
    
    std::cout << "[INFO] Audio capture started\n";
    std::cout << "[INFO] Say a wake word to start transcription\n";
    std::cout << "[INFO] Press Ctrl+C to exit\n\n";
    
    // Test mode menu
    auto show_menu = []() {
        std::cout << "\n--- Test Commands ---\n";
        std::cout << "1. Toggle listening\n";
        std::cout << "2. Show statistics\n";
        std::cout << "3. Export session\n";
        std::cout << "4. List speakers\n";
        std::cout << "5. Test noise cancellation\n";
        std::cout << "q. Quit\n";
        std::cout << "Enter command: ";
    };
    
    // Start input thread for testing
    std::thread input_thread([&]() {
        std::string input;
        while (g_running) {
            show_menu();
            std::getline(std::cin, input);
            
            if (input == "1") {
                vtt_system.toggle_listening();
                std::cout << "[INFO] Listening: " 
                         << (vtt_system.is_listening() ? "ON" : "OFF") << "\n";
            }
            else if (input == "2") {
                const auto& stats = vtt_system.get_stats();
                std::cout << "\n--- System Statistics ---\n";
                std::cout << "Audio processed: " << stats.total_audio_processed_ms / 1000 << " sec\n";
                std::cout << "Transcriptions: " << stats.total_transcriptions << "\n";
                std::cout << "Wake word detections: " << stats.wake_word_detections << "\n";
                std::cout << "Speaker changes: " << stats.speaker_changes << "\n";
                std::cout << "Average confidence: " << (stats.average_confidence * 100) << "%\n";
                std::cout << "Average latency: " << stats.average_latency_ms << " ms\n";
                std::cout << "Noise reduction: " << stats.noise_reduction_avg_db << " dB\n";
            }
            else if (input == "3") {
                std::string session_data = vtt_system.export_session();
                std::cout << "\n--- Session Export ---\n";
                std::cout << session_data << "\n";
            }
            else if (input == "4") {
                auto speakers = vtt_system.get_speaker_list();
                std::cout << "\n--- Detected Speakers ---\n";
                for (const auto& speaker : speakers) {
                    std::cout << "  - " << speaker << "\n";
                }
                if (speakers.empty()) {
                    std::cout << "  No speakers detected yet\n";
                }
            }
            else if (input == "5") {
                std::cout << "[INFO] Generating test noise...\n";
                // Generate noisy test signal
                std::vector<float> noisy_signal(16000);  // 1 second
                for (size_t i = 0; i < noisy_signal.size(); ++i) {
                    // Speech-like signal with noise
                    float speech = 0.3f * std::sin(2 * M_PI * 200 * i / 16000.0f);
                    float noise = 0.1f * ((rand() / float(RAND_MAX)) - 0.5f);
                    noisy_signal[i] = speech + noise;
                }
                
                std::cout << "[INFO] Processing noisy signal...\n";
                vtt_system.process_audio(noisy_signal.data(), noisy_signal.size(), 16000);
            }
            else if (input == "q") {
                g_running = false;
            }
        }
    });
    
    // Main loop
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Clean up
    audio_capture.stop();
    if (input_thread.joinable()) {
        input_thread.join();
    }
    
    // Print final report
    vtt_system.print_performance_report();
    
    std::cout << "\n[INFO] Test completed successfully\n";
    return 0;
}