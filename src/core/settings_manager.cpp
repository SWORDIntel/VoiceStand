#include "settings_manager.h"
#include <fstream>
#include <iostream>

namespace vtt {

SettingsManager::SettingsManager() {
    config_dir_ = get_config_directory();
}

SettingsManager& SettingsManager::instance() {
    static SettingsManager inst;
    return inst;
}

bool SettingsManager::load_settings() {
    std::string settings_path = config_dir_ + "/settings.json";
    
    std::ifstream file(settings_path);
    if (!file.is_open()) {
        std::cerr << "[INFO] Settings file not found, using defaults\n";
        save_settings(); // Create default settings file
        return true;
    }
    
    Json::Reader reader;
    if (!reader.parse(file, settings_)) {
        std::cerr << "[ERROR] Failed to parse settings file\n";
        return false;
    }
    
    std::cout << "[INFO] Settings loaded from " << settings_path << "\n";
    return true;
}

bool SettingsManager::save_settings() {
    std::string settings_path = config_dir_ + "/settings.json";
    
    // Ensure directory exists
    std::filesystem::create_directories(config_dir_);
    
    // Set default values if not present
    if (!settings_.isMember("audio")) {
        settings_["audio"]["device"] = "default";
        settings_["audio"]["sample_rate"] = 16000;
        settings_["audio"]["channels"] = 1;
        settings_["audio"]["vad_threshold"] = 0.5;
    }
    
    if (!settings_.isMember("whisper")) {
        settings_["whisper"]["model_path"] = "models/ggml-base.en.bin";
        settings_["whisper"]["language"] = "en";
        settings_["whisper"]["use_gpu"] = false;
        settings_["whisper"]["beam_size"] = 5;
        settings_["whisper"]["n_threads"] = 4;
    }
    
    if (!settings_.isMember("output")) {
        settings_["output"]["format"] = "text";
        settings_["output"]["timestamps"] = false;
        settings_["output"]["auto_punctuation"] = true;
        settings_["output"]["speaker_diarization"] = false;
    }
    
    if (!settings_.isMember("hotkeys")) {
        settings_["hotkeys"]["push_to_talk"] = "Ctrl+Space";
        settings_["hotkeys"]["toggle_listening"] = "Ctrl+Shift+L";
        settings_["hotkeys"]["clear_text"] = "Ctrl+Shift+C";
    }
    
    if (!settings_.isMember("ui")) {
        settings_["ui"]["theme"] = "dark";
        settings_["ui"]["window_opacity"] = 0.95;
        settings_["ui"]["always_on_top"] = false;
        settings_["ui"]["start_minimized"] = false;
    }
    
    // Write to file
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    
    std::ofstream file(settings_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open settings file for writing\n";
        return false;
    }
    
    writer->write(settings_, &file);
    std::cout << "[INFO] Settings saved to " << settings_path << "\n";
    return true;
}

std::string SettingsManager::get_config_directory() {
    const char* home = std::getenv("HOME");
    if (!home) {
        home = std::getenv("USERPROFILE"); // Windows fallback
    }
    
    if (home) {
        return std::string(home) + "/.config/voice-to-text";
    }
    
    return "."; // Current directory as fallback
}

} // namespace vtt