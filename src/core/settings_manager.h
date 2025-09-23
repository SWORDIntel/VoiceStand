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
        // Parse the path
        std::vector<std::string> path_parts;
        std::string current_part;

        // Simple path parsing (split by '.')
        for (char c : path) {
            if (c == '.') {
                if (!current_part.empty()) {
                    path_parts.push_back(current_part);
                    current_part.clear();
                }
            } else {
                current_part += c;
            }
        }
        if (!current_part.empty()) {
            path_parts.push_back(current_part);
        }

        if (path_parts.empty()) {
            return default_value; // Invalid path
        }

        // Navigate to the target value
        const Json::Value* current = &settings_;
        for (const std::string& part : path_parts) {
            if (!current->isObject() || !current->isMember(part)) {
                return default_value; // Path doesn't exist
            }
            current = &((*current)[part]);
        }

        // Extract value with proper type conversion
        if (current && !current->isNull()) {
            if constexpr (std::is_same_v<T, bool>) {
                return current->isBool() ? current->asBool() : default_value;
            } else if constexpr (std::is_same_v<T, int>) {
                return current->isInt() ? current->asInt() : default_value;
            } else if constexpr (std::is_same_v<T, float>) {
                return current->isNumeric() ? current->asFloat() : default_value;
            } else if constexpr (std::is_same_v<T, std::string>) {
                return current->isString() ? current->asString() : default_value;
            }
        }
        return default_value;
    }
    
    // Set setting value
    template<typename T>
    void set(const std::string& path, const T& value) {
        // Parse the path and create nested structure if needed
        std::vector<std::string> path_parts;
        std::string current_part;

        // Simple path parsing (split by '.')
        for (char c : path) {
            if (c == '.') {
                if (!current_part.empty()) {
                    path_parts.push_back(current_part);
                    current_part.clear();
                }
            } else {
                current_part += c;
            }
        }
        if (!current_part.empty()) {
            path_parts.push_back(current_part);
        }

        if (path_parts.empty()) {
            return; // Invalid path
        }

        // Navigate to the target, creating objects as needed
        Json::Value* current = &settings_;
        for (size_t i = 0; i < path_parts.size() - 1; ++i) {
            const std::string& part = path_parts[i];
            if (!current->isObject()) {
                *current = Json::Value(Json::objectValue);
            }
            if (!current->isMember(part)) {
                (*current)[part] = Json::Value(Json::objectValue);
            }
            current = &((*current)[part]);
        }

        // Set the final value
        const std::string& final_key = path_parts.back();
        if (!current->isObject()) {
            *current = Json::Value(Json::objectValue);
        }

        // Set the value with proper type conversion
        if constexpr (std::is_same_v<T, bool>) {
            (*current)[final_key] = value;
        } else if constexpr (std::is_same_v<T, int>) {
            (*current)[final_key] = value;
        } else if constexpr (std::is_same_v<T, float>) {
            (*current)[final_key] = static_cast<double>(value);
        } else if constexpr (std::is_same_v<T, std::string>) {
            (*current)[final_key] = value;
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