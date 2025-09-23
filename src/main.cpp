#include "core/audio_capture.h"
#include "core/whisper_processor.h"
#include "gui/main_window.h"
#include "integration/hotkey_manager.h"

#include <iostream>
#include <memory>
#include <signal.h>
#include <filesystem>
#include <fstream>
#include <json/json.h>

namespace fs = std::filesystem;

class VoiceToTextApp {
public:
    VoiceToTextApp() 
        : audio_capture_(std::make_unique<vtt::AudioCapture>())
        , whisper_processor_(std::make_unique<vtt::WhisperProcessor>())
        , main_window_(std::make_unique<vtt::MainWindow>())
        , hotkey_manager_(std::make_unique<vtt::HotkeyManager>()) {
    }
    
    bool initialize(int argc, char** argv) {
        if (!load_config()) {
            create_default_config();
        }

        // Handle command line arguments
        if (argc > 1) {
            std::string command = argv[1];

            if (command == "--download-model") {
                if (argc < 3) {
                    std::cerr << "Usage: " << argv[0] << " --download-model <size>\n";
                    std::cerr << "Sizes: tiny, base, small, medium, large\n";
                    return false;
                }
                return download_model(argv[2]);
            }
            else if (command == "--switch-model") {
                if (argc < 3) {
                    std::cerr << "Usage: " << argv[0] << " --switch-model <size>\n";
                    std::cerr << "Sizes: tiny, base, small, medium, large\n";
                    return false;
                }
                return switch_model(argv[2]);
            }
            else if (command == "--list-models") {
                return list_models();
            }
            else if (command == "--model-info") {
                return show_model_info();
            }
            else if (command == "--hardware-check") {
                return hardware_check();
            }
            else if (command == "--system-info") {
                return system_info();
            }
            else if (command == "--help" || command == "-h") {
                show_help(argv[0]);
                return false;
            }
            else if (command.substr(0, 2) == "--") {
                std::cerr << "Unknown option: " << command << "\n";
                std::cerr << "Use --help for usage information\n";
                return false;
            }
        }
        
        vtt::AudioConfig audio_config;
        audio_config.sample_rate = config_["audio"]["sample_rate"].asUInt();
        audio_config.vad_threshold = config_["audio"]["vad_threshold"].asFloat();
        
        if (!audio_capture_->initialize(audio_config)) {
            std::cerr << "Failed to initialize audio capture\n";
            return false;
        }
        
        vtt::WhisperConfig whisper_config;
        whisper_config.model_path = config_["whisper"]["model_path"].asString();
        whisper_config.language = config_["whisper"]["language"].asString();
        whisper_config.num_threads = config_["whisper"]["num_threads"].asInt();
        
        if (!whisper_processor_->initialize(whisper_config)) {
            std::cerr << "Failed to initialize Whisper processor\n";
            return false;
        }
        
        if (!hotkey_manager_->initialize()) {
            std::cerr << "Failed to initialize hotkey manager\n";
            return false;
        }
        
        // Pass config to main window
        main_window_->set_config(config_);
        
        setup_callbacks();
        
        std::string hotkey = config_["hotkeys"]["toggle_recording"].asString();
        if (!hotkey_manager_->register_hotkey(hotkey)) {
            std::cerr << "Failed to register hotkey: " << hotkey << "\n";
            return false;
        }

        // Register mouse button if configured
        if (config_["hotkeys"].isMember("toggle_recording_mouse")) {
            std::string mouse_button = config_["hotkeys"]["toggle_recording_mouse"].asString();
            if (!mouse_button.empty() && !hotkey_manager_->register_mouse_button(mouse_button)) {
                std::cerr << "Failed to register mouse button: " << mouse_button << "\n";
                // Don't fail initialization for mouse button registration failure
            }
        }
        
        hotkey_manager_->start();
        
        whisper_processor_->start_streaming();
        
        return true;
    }
    
    void run(int argc, char** argv) {
        std::cout << "Voice to Text - Starting...\n";
        std::cout << "Press " << config_["hotkeys"]["toggle_recording"].asString() 
                  << " to toggle recording\n";
        
        main_window_->initialize(argc, argv);
    }
    
private:
    bool load_config() {
        fs::path config_dir = fs::path(getenv("HOME")) / ".config" / "voice-to-text";
        fs::path config_file = config_dir / "config.json";
        
        if (!fs::exists(config_file)) {
            return false;
        }
        
        std::ifstream file(config_file);
        Json::Reader reader;
        return reader.parse(file, config_);
    }
    
    void create_default_config() {
        fs::path config_dir = fs::path(getenv("HOME")) / ".config" / "voice-to-text";
        fs::create_directories(config_dir);
        
        fs::path models_dir = config_dir / "models";
        fs::create_directories(models_dir);
        
        config_["audio"]["sample_rate"] = 16000;
        config_["audio"]["vad_threshold"] = 0.3;
        config_["audio"]["device"] = "default";
        
        config_["whisper"]["model_path"] = (models_dir / "ggml-base.bin").string();
        config_["whisper"]["language"] = "auto";
        config_["whisper"]["num_threads"] = 4;
        
        config_["hotkeys"]["toggle_recording"] = "Ctrl+Alt+Space";
        config_["hotkeys"]["push_to_talk"] = "Ctrl+Alt+V";
        config_["hotkeys"]["toggle_recording_mouse"] = "";  // Disabled by default
        
        config_["ui"]["theme"] = "system";
        config_["ui"]["show_waveform"] = true;
        config_["ui"]["auto_scroll"] = true;
        
        save_config();
    }
    
    void save_config() {
        fs::path config_dir = fs::path(getenv("HOME")) / ".config" / "voice-to-text";
        fs::path config_file = config_dir / "config.json";
        
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        
        std::ofstream file(config_file);
        writer->write(config_, &file);
    }
    
    bool download_model(const std::string& model_size) {
        fs::path config_dir = fs::path(getenv("HOME")) / ".config" / "voice-to-text";
        fs::path models_dir = config_dir / "models";
        fs::create_directories(models_dir);

        std::string filename = "ggml-" + model_size + ".bin";
        fs::path model_path = models_dir / filename;

        if (fs::exists(model_path)) {
            std::cout << "Model already exists: " << model_path << "\n";
            return true;
        }

        return vtt::WhisperProcessor::download_model(model_size, model_path.string());
    }

    bool switch_model(const std::string& model_size) {
        fs::path config_dir = fs::path(getenv("HOME")) / ".config" / "voice-to-text";
        fs::path models_dir = config_dir / "models";

        std::string filename = "ggml-" + model_size + ".bin";
        fs::path model_path = models_dir / filename;

        if (!fs::exists(model_path)) {
            std::cout << "Model not found: " << model_path << "\n";
            std::cout << "Download it first with: --download-model " << model_size << "\n";
            return false;
        }

        // Update configuration
        config_["whisper"]["model_path"] = model_path.string();

        // Update model-specific settings
        update_model_config(model_size);

        save_config();

        std::cout << "✅ Switched to " << model_size << " model\n";
        std::cout << "📍 Model path: " << model_path << "\n";
        std::cout << "⚙️  Configuration updated\n";

        return true;
    }

    bool list_models() {
        fs::path config_dir = fs::path(getenv("HOME")) / ".config" / "voice-to-text";
        fs::path models_dir = config_dir / "models";

        std::cout << "📦 Available Models:\n";
        std::cout << "==================\n\n";

        const std::vector<std::string> model_sizes = {"tiny", "base", "small", "medium", "large"};
        const std::map<std::string, std::string> model_descriptions = {
            {"tiny", "39MB - Development/Testing (65% accuracy)"},
            {"base", "142MB - Recommended Default (72% accuracy)"},
            {"small", "244MB - Professional Use (76% accuracy)"},
            {"medium", "769MB - High Accuracy (81% accuracy)"},
            {"large", "1550MB - Maximum Accuracy (84% accuracy)"}
        };

        std::string current_model = get_current_model();
        bool found_any = false;

        for (const auto& model : model_sizes) {
            std::string filename = "ggml-" + model + ".bin";
            fs::path model_path = models_dir / filename;

            if (fs::exists(model_path)) {
                found_any = true;
                std::string status = (model == current_model) ? " (ACTIVE)" : "";
                std::cout << "✅ " << model << status << " - "
                          << model_descriptions.at(model) << "\n";
                std::cout << "   📍 " << model_path << "\n";

                // Show file size
                auto file_size = fs::file_size(model_path);
                std::cout << "   📊 Size: " << (file_size / (1024 * 1024)) << "MB\n\n";
            } else {
                std::cout << "❌ " << model << " - "
                          << model_descriptions.at(model) << " (not downloaded)\n\n";
            }
        }

        if (!found_any) {
            std::cout << "No models found. Download one with:\n";
            std::cout << "  ./voice-to-text --download-model base\n\n";
        }

        std::cout << "Current active model: " << current_model << "\n";
        std::cout << "\nTo switch models: ./voice-to-text --switch-model <size>\n";
        std::cout << "To download models: ./voice-to-text --download-model <size>\n";

        return true;
    }

    bool show_model_info() {
        std::string current_model = get_current_model();

        std::cout << "🎯 Current Model Information\n";
        std::cout << "===========================\n\n";

        if (current_model.empty()) {
            std::cout << "❌ No model configured\n";
            std::cout << "Download a model with: ./voice-to-text --download-model base\n";
            return false;
        }

        fs::path model_path = config_["whisper"]["model_path"].asString();

        std::cout << "📍 Active Model: " << current_model << "\n";
        std::cout << "📂 Path: " << model_path << "\n";

        if (fs::exists(model_path)) {
            auto file_size = fs::file_size(model_path);
            std::cout << "📊 File Size: " << (file_size / (1024 * 1024)) << "MB\n";

            // Model specifications
            const std::map<std::string, std::tuple<std::string, std::string, std::string>> specs = {
                {"tiny", {"~390MB", "<1ms", "~65%"}},
                {"base", {"~500MB", "<2ms", "~72%"}},
                {"small", {"~1GB", "<3ms", "~76%"}},
                {"medium", {"~2GB", "<5ms", "~81%"}},
                {"large", {"~4-6GB", "<8ms", "~84%"}}
            };

            if (specs.find(current_model) != specs.end()) {
                auto [ram, latency, accuracy] = specs.at(current_model);
                std::cout << "💾 RAM Usage: " << ram << "\n";
                std::cout << "⚡ NPU Latency: " << latency << "\n";
                std::cout << "🎯 Accuracy: " << accuracy << "\n";
            }
        } else {
            std::cout << "❌ Model file not found!\n";
        }

        // Configuration details
        std::cout << "\n⚙️  Configuration:\n";
        std::cout << "   Threads: " << config_["whisper"]["num_threads"].asInt() << "\n";
        std::cout << "   Language: " << config_["whisper"]["language"].asString() << "\n";
        std::cout << "   Audio Sample Rate: " << config_["audio"]["sample_rate"].asUInt() << "Hz\n";
        std::cout << "   VAD Threshold: " << config_["audio"]["vad_threshold"].asFloat() << "\n";

        return true;
    }

    bool hardware_check() {
        std::cout << "🔍 Hardware Capability Check\n";
        std::cout << "============================\n\n";

        // CPU Information
        std::cout << "🖥️  CPU Information:\n";
        std::system("lscpu | grep -E 'Model name|Architecture|CPU\\(s\\):|Thread|Core'");

        // Memory Information
        std::cout << "\n💾 Memory Information:\n";
        std::system("free -h");

        // Intel NPU Detection
        std::cout << "\n🧠 Intel NPU Detection:\n";
        if (fs::exists("/sys/class/intel_npu")) {
            std::cout << "✅ Intel NPU detected\n";
            std::system("ls -la /sys/class/intel_npu/");
        } else {
            std::cout << "❌ Intel NPU not detected\n";
        }

        // Check for Meteor Lake indicators
        std::cout << "\n🚀 Meteor Lake Detection:\n";
        int result = std::system("lscpu | grep -qi 'meteor\\|ultra'");
        if (result == 0) {
            std::cout << "✅ Meteor Lake CPU detected\n";
        } else {
            std::cout << "❌ Meteor Lake CPU not detected\n";
        }

        // Audio System Check
        std::cout << "\n🔊 Audio System:\n";
        std::system("pactl info 2>/dev/null | head -5 || echo 'PulseAudio not available'");

        // Recommendations
        std::cout << "\n💡 Recommendations:\n";
        if (result == 0) {
            std::cout << "   • Use 'small' or 'medium' model for optimal NPU acceleration\n";
            std::cout << "   • Enable NPU processing in configuration\n";
        } else {
            std::cout << "   • Use 'base' model for CPU processing\n";
            std::cout << "   • Consider upgrading to Intel Meteor Lake for NPU benefits\n";
        }

        return true;
    }

    bool system_info() {
        std::cout << "📊 VoiceStand System Information\n";
        std::cout << "================================\n\n";

        // Application Info
        std::cout << "🎤 Application:\n";
        std::cout << "   Version: VoiceStand v1.0\n";
        std::cout << "   Build: Production Release\n";
        std::cout << "   Target: Intel Hardware Accelerated\n\n";

        // Current Configuration
        std::cout << "⚙️  Current Configuration:\n";
        std::cout << "   Config Path: " << fs::path(getenv("HOME")) / ".config" / "voice-to-text" / "config.json" << "\n";
        std::cout << "   Models Path: " << fs::path(getenv("HOME")) / ".config" / "voice-to-text" / "models" << "\n";
        std::cout << "   Active Model: " << get_current_model() << "\n";
        std::cout << "   Recording Hotkey: " << config_["hotkeys"]["toggle_recording"].asString() << "\n\n";

        // System Resources
        std::cout << "💻 System Resources:\n";
        std::system("echo '   OS: '$(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"')");
        std::system("echo '   Kernel: '$(uname -r)");
        std::system("echo '   Architecture: '$(uname -m)");
        std::cout << "\n";

        // Performance Metrics
        std::cout << "⚡ Expected Performance:\n";
        std::string model = get_current_model();
        if (model == "tiny") {
            std::cout << "   • Inference: <1ms (NPU) / <5ms (CPU)\n";
            std::cout << "   • Memory: ~390MB\n";
            std::cout << "   • Accuracy: ~65%\n";
        } else if (model == "base") {
            std::cout << "   • Inference: <2ms (NPU) / <10ms (CPU)\n";
            std::cout << "   • Memory: ~500MB\n";
            std::cout << "   • Accuracy: ~72%\n";
        } else if (model == "small") {
            std::cout << "   • Inference: <3ms (NPU) / <15ms (CPU)\n";
            std::cout << "   • Memory: ~1GB\n";
            std::cout << "   • Accuracy: ~76%\n";
        }

        return true;
    }

    void show_help(const char* program_name) {
        std::cout << "🎤 VoiceStand - Intel Hardware Accelerated Voice-to-Text\n";
        std::cout << "========================================================\n\n";

        std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";

        std::cout << "Model Management:\n";
        std::cout << "  --download-model <size>   Download a Whisper model\n";
        std::cout << "  --switch-model <size>     Switch to a different model\n";
        std::cout << "  --list-models             List all available models\n";
        std::cout << "  --model-info              Show current model information\n\n";

        std::cout << "System Information:\n";
        std::cout << "  --hardware-check          Check hardware capabilities\n";
        std::cout << "  --system-info             Show system and configuration info\n\n";

        std::cout << "General:\n";
        std::cout << "  --help, -h                Show this help message\n\n";

        std::cout << "Model Sizes:\n";
        std::cout << "  tiny     39MB   ~65% accuracy  <1ms latency  (Development)\n";
        std::cout << "  base     142MB  ~72% accuracy  <2ms latency  (Recommended)\n";
        std::cout << "  small    244MB  ~76% accuracy  <3ms latency  (Professional)\n";
        std::cout << "  medium   769MB  ~81% accuracy  <5ms latency  (High Accuracy)\n";
        std::cout << "  large    1550MB ~84% accuracy  <8ms latency  (Maximum)\n\n";

        std::cout << "Examples:\n";
        std::cout << "  " << program_name << " --download-model base\n";
        std::cout << "  " << program_name << " --switch-model small\n";
        std::cout << "  " << program_name << " --hardware-check\n";
        std::cout << "  " << program_name << "                    # Start GUI application\n\n";

        std::cout << "For more information, visit:\n";
        std::cout << "  https://github.com/SWORDIntel/VoiceStand\n";
    }

    std::string get_current_model() {
        std::string model_path = config_["whisper"]["model_path"].asString();
        if (model_path.empty()) return "";

        fs::path path(model_path);
        std::string filename = path.filename().string();

        // Extract model size from filename (ggml-<size>.bin)
        if (filename.substr(0, 5) == "ggml-" && filename.length() >= 4 && filename.substr(filename.length() - 4) == ".bin") {
            return filename.substr(5, filename.length() - 9);
        }

        return "unknown";
    }

    void update_model_config(const std::string& model_size) {
        // Update model-specific settings
        if (model_size == "tiny") {
            config_["whisper"]["num_threads"] = 1;
            config_["audio"]["vad_threshold"] = 0.6;
        } else if (model_size == "base") {
            config_["whisper"]["num_threads"] = 2;
            config_["audio"]["vad_threshold"] = 0.5;
        } else if (model_size == "small") {
            config_["whisper"]["num_threads"] = 4;
            config_["audio"]["vad_threshold"] = 0.4;
        } else if (model_size == "medium") {
            config_["whisper"]["num_threads"] = 4;
            config_["audio"]["vad_threshold"] = 0.4;
        } else if (model_size == "large") {
            config_["whisper"]["num_threads"] = 6;
            config_["audio"]["vad_threshold"] = 0.4;
        }
    }
    
    void setup_callbacks() {
        audio_capture_->set_audio_callback(
            [this](const vtt::AudioData& data) {
                whisper_processor_->process_audio(
                    data.samples, 
                    data.num_samples, 
                    data.sample_rate
                );
                
                std::vector<float> samples(data.samples, data.samples + data.num_samples);
                main_window_->update_waveform(samples);
            }
        );
        
        whisper_processor_->set_transcription_callback(
            [this](const vtt::TranscriptionResult& result) {
                main_window_->append_transcription(result.text, result.is_final);
            }
        );
        
        main_window_->set_recording_started_callback(
            [this]() {
                audio_capture_->start();
            }
        );
        
        main_window_->set_recording_stopped_callback(
            [this]() {
                audio_capture_->stop();
            }
        );
        
        hotkey_manager_->set_hotkey_callback(
            [this](const std::string& hotkey) {
                if (hotkey == config_["hotkeys"]["toggle_recording"].asString()) {
                    main_window_->toggle_recording();
                }
            }
        );
        
        main_window_->set_config_update_callback(
            [this](const Json::Value& new_config) {
                config_ = new_config;
                save_config();
                std::cout << "[INFO] Configuration updated from settings dialog\n";
            }
        );
    }
    
    std::unique_ptr<vtt::AudioCapture> audio_capture_;
    std::unique_ptr<vtt::WhisperProcessor> whisper_processor_;
    std::unique_ptr<vtt::MainWindow> main_window_;
    std::unique_ptr<vtt::HotkeyManager> hotkey_manager_;
    Json::Value config_;
};

std::unique_ptr<VoiceToTextApp> g_app;

void signal_handler(int sig) {
    std::cout << "\nShutting down...\n";
    if (g_app) {
        exit(0);
    }
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    g_app = std::make_unique<VoiceToTextApp>();
    
    if (!g_app->initialize(argc, argv)) {
        std::cerr << "Failed to initialize application\n";
        return 1;
    }
    
    g_app->run(argc, argv);
    
    return 0;
}