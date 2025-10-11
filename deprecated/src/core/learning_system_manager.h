#pragma once

#include "learning_api_client.h"
#include "ensemble_whisper_processor.h"
#include "adaptive_learning_system.h"
#include "audio_feature_serializer.h"
#include "settings_manager.h"
#include <memory>
#include <string>

namespace vtt {

// High-level manager that coordinates all learning system components
class LearningSystemManager {
public:
    struct LearningSystemConfig {
        bool enabled = false;

        // API client configuration
        LearningApiConfig api_config;

        // Ensemble processor configuration
        EnsembleWhisperProcessor::EnsembleConfig ensemble_config;

        // Adaptive learning configuration
        std::string adaptive_learning_db_url = "postgresql://localhost/voicestand_learning";

        // Feature extraction configuration
        bool enable_advanced_features = true;
        bool enable_real_time_learning = true;
    };

    LearningSystemManager();
    explicit LearningSystemManager(const LearningSystemConfig& config);
    ~LearningSystemManager();

    // Initialization and lifecycle
    bool initialize();
    void cleanup();
    bool is_initialized() const { return initialized_; }

    // Component access
    std::shared_ptr<LearningApiClient> get_api_client() const { return api_client_; }
    std::shared_ptr<EnsembleWhisperProcessor> get_ensemble_processor() const { return ensemble_processor_; }
    std::shared_ptr<AdaptiveLearningSystem> get_adaptive_learning() const { return adaptive_learning_; }
    std::shared_ptr<AudioFeatureSerializer> get_feature_serializer() const { return feature_serializer_; }
    std::shared_ptr<LearningSystemIntegration> get_integration() const { return integration_; }

    // Configuration management
    void update_config(const LearningSystemConfig& new_config);
    const LearningSystemConfig& get_config() const { return config_; }

    // Load configuration from settings manager
    static LearningSystemConfig load_config_from_settings(const SettingsManager& settings);

    // Integration with existing processors
    void integrate_with_whisper_processor(WhisperProcessor* processor);

    // Manual learning operations
    void submit_correction(const std::string& recognized_text, const std::string& correct_text);
    void request_model_updates();
    void update_speaker_profile(const std::string& speaker_id, const std::vector<std::string>& sample_texts);

    // Statistics and monitoring
    struct LearningStats {
        size_t total_recognitions_submitted = 0;
        size_t total_corrections_submitted = 0;
        size_t total_model_updates_received = 0;
        double current_ensemble_accuracy = 0.0;
        bool api_online = false;
        std::chrono::steady_clock::time_point last_api_contact;
        size_t pending_api_requests = 0;
        size_t offline_requests_queued = 0;
    };

    LearningStats get_learning_stats() const;

    // Control functions
    void enable_learning_system(bool enable);
    void enable_real_time_learning(bool enable);
    void enable_ensemble_processing(bool enable);

private:
    LearningSystemConfig config_;
    bool initialized_ = false;

    // Core components
    std::shared_ptr<LearningApiClient> api_client_;
    std::shared_ptr<EnsembleWhisperProcessor> ensemble_processor_;
    std::shared_ptr<AdaptiveLearningSystem> adaptive_learning_;
    std::shared_ptr<AudioFeatureSerializer> feature_serializer_;
    std::shared_ptr<LearningSystemIntegration> integration_;

    // Internal state
    mutable std::mutex stats_mutex_;
    LearningStats current_stats_;

    // Helper methods
    void setup_integration_callbacks();
    void update_stats();
};

// Factory function for easy creation with settings
std::shared_ptr<LearningSystemManager> create_learning_system_from_settings(const SettingsManager& settings);

// Utility functions for configuration conversion
LearningApiConfig convert_settings_to_api_config(const Json::Value& settings);
EnsembleWhisperProcessor::EnsembleConfig convert_settings_to_ensemble_config(const Json::Value& settings);

} // namespace vtt