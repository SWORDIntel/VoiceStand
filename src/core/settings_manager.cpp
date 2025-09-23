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
        settings_["hotkeys"]["toggle_recording_mouse"] = "";  // Disabled by default
        settings_["hotkeys"]["push_to_talk_mouse"] = "";      // Optional mouse PTT
    }
    
    if (!settings_.isMember("ui")) {
        settings_["ui"]["theme"] = "dark";
        settings_["ui"]["window_opacity"] = 0.95;
        settings_["ui"]["always_on_top"] = false;
        settings_["ui"]["start_minimized"] = false;
    }

    if (!settings_.isMember("learning_api")) {
        settings_["learning_api"]["enabled"] = false;
        settings_["learning_api"]["base_url"] = "http://localhost:5000";
        settings_["learning_api"]["api_version"] = "v1";
        settings_["learning_api"]["api_key"] = "";
        settings_["learning_api"]["connection_timeout_seconds"] = 10;
        settings_["learning_api"]["request_timeout_seconds"] = 30;
        settings_["learning_api"]["max_retries"] = 3;
        settings_["learning_api"]["enable_async_processing"] = true;
        settings_["learning_api"]["max_concurrent_requests"] = 5;
        settings_["learning_api"]["enable_batching"] = true;
        settings_["learning_api"]["batch_size"] = 10;
        settings_["learning_api"]["enable_offline_queue"] = true;
        settings_["learning_api"]["offline_queue_max_size"] = 10000;
    }

    if (!settings_.isMember("ensemble")) {
        settings_["ensemble"]["enabled"] = false;
        settings_["ensemble"]["learning_mode"] = "BALANCED";  // CONSERVATIVE, BALANCED, EXPERIMENTAL, ADAPTIVE
        settings_["ensemble"]["accuracy_target"] = 0.95;
        settings_["ensemble"]["confidence_threshold"] = 0.85;
        settings_["ensemble"]["min_ensemble_size"] = 3;
        settings_["ensemble"]["max_ensemble_size"] = 5;
        settings_["ensemble"]["enable_uk_dialect_optimization"] = true;
        settings_["ensemble"]["enable_model_caching"] = true;
        settings_["ensemble"]["enable_parallel_inference"] = true;
        settings_["ensemble"]["max_parallel_models"] = 3;
        settings_["ensemble"]["learning_update_interval_seconds"] = 300;
    }

    if (!settings_.isMember("adaptive_learning")) {
        settings_["adaptive_learning"]["enabled"] = false;
        settings_["adaptive_learning"]["database_url"] = "postgresql://localhost/voicestand_learning";
        settings_["adaptive_learning"]["pattern_confidence_threshold"] = 0.7;
        settings_["adaptive_learning"]["uk_pattern_bonus"] = 0.1;
        settings_["adaptive_learning"]["max_history_size"] = 10000;
        settings_["adaptive_learning"]["min_pattern_usage"] = 5;
        settings_["adaptive_learning"]["enable_uk_english_learning"] = true;
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