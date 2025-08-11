#pragma once

#include <json/json.h>
#include <string>
#include <filesystem>

namespace vtt {

class SettingsManager {
public:
    static SettingsManager& instance();
    
    // Load settings from file
    bool load_settings();
    
    // Save settings to file
    bool save_settings();
    
    // Get setting value
    template<typename T>
    T get(const std::string& path, const T& default_value) const {
        Json::Path json_path(path);
        const Json::Value* value = json_path.resolve(settings_);
        if (value && !value->isNull()) {
            if constexpr (std::is_same_v<T, bool>) {
                return value->asBool();
            } else if constexpr (std::is_same_v<T, int>) {
                return value->asInt();
            } else if constexpr (std::is_same_v<T, float>) {
                return value->asFloat();
            } else if constexpr (std::is_same_v<T, std::string>) {
                return value->asString();
            }
        }
        return default_value;
    }
    
    // Set setting value
    template<typename T>
    void set(const std::string& path, const T& value) {
        Json::Path json_path(path);
        Json::Value& target = const_cast<Json::Value&>(*json_path.resolve(settings_));
        if constexpr (std::is_same_v<T, bool>) {
            target = value;
        } else if constexpr (std::is_same_v<T, int>) {
            target = value;
        } else if constexpr (std::is_same_v<T, float>) {
            target = value;
        } else if constexpr (std::is_same_v<T, std::string>) {
            target = value;
        }
    }
    
    // Get the entire settings object
    const Json::Value& get_settings() const { return settings_; }
    
    // Get config directory
    std::string get_config_directory();
    
private:
    SettingsManager();
    SettingsManager(const SettingsManager&) = delete;
    SettingsManager& operator=(const SettingsManager&) = delete;
    
    Json::Value settings_;
    std::string config_dir_;
};

} // namespace vtt