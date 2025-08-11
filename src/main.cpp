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
        
        if (argc > 1 && std::string(argv[1]) == "--download-model") {
            if (argc < 3) {
                std::cerr << "Usage: " << argv[0] << " --download-model <size>\n";
                std::cerr << "Sizes: tiny, base, small, medium, large\n";
                return false;
            }
            return download_model(argv[2]);
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