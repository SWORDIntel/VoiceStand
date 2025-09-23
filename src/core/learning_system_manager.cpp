#include "learning_system_manager.h"
#include <iostream>

namespace vtt {

LearningSystemManager::LearningSystemManager()
    : LearningSystemManager(LearningSystemConfig{}) {
}

LearningSystemManager::LearningSystemManager(const LearningSystemConfig& config)
    : config_(config) {
}

LearningSystemManager::~LearningSystemManager() {
    cleanup();
}

bool LearningSystemManager::initialize() {
    if (initialized_) {
        return true;
    }

    if (!config_.enabled) {
        std::cout << "Learning system disabled in configuration" << std::endl;
        initialized_ = true;  // Mark as initialized but inactive
        return true;
    }

    try {
        // Initialize API client
        api_client_ = std::make_shared<LearningApiClient>(config_.api_config);
        if (!api_client_->initialize()) {
            std::cerr << "Failed to initialize learning API client" << std::endl;
            return false;
        }

        // Initialize feature serializer
        feature_serializer_ = std::make_shared<AudioFeatureSerializer>();

        // Initialize adaptive learning system
        adaptive_learning_ = std::make_shared<AdaptiveLearningSystem>(config_.adaptive_learning_db_url);
        if (!adaptive_learning_->initialize()) {
            std::cout << "Warning: Adaptive learning system initialization failed, continuing without it" << std::endl;
        }

        // Initialize ensemble processor
        ensemble_processor_ = std::make_shared<EnsembleWhisperProcessor>(config_.ensemble_config);
        if (!ensemble_processor_->initialize()) {
            std::cout << "Warning: Ensemble processor initialization failed, using single model" << std::endl;
            ensemble_processor_.reset();
        }

        // Create integration layer
        integration_ = std::make_shared<LearningSystemIntegration>(api_client_);

        // Set up component integrations
        if (ensemble_processor_) {
            ensemble_processor_->set_learning_integration(integration_);
        }

        if (adaptive_learning_) {
            adaptive_learning_->set_learning_integration(integration_);
        }

        // Set up integration callbacks
        setup_integration_callbacks();

        // Start real-time learning if enabled
        if (config_.enable_real_time_learning) {
            if (adaptive_learning_) {
                adaptive_learning_->start_real_time_learning();
            }
            if (ensemble_processor_) {
                ensemble_processor_->start_continuous_learning();
            }
        }

        initialized_ = true;
        std::cout << "Learning system manager initialized successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing learning system manager: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void LearningSystemManager::cleanup() {
    if (!initialized_) {
        return;
    }

    // Stop real-time learning
    if (adaptive_learning_) {
        adaptive_learning_->stop_real_time_learning();
    }

    if (ensemble_processor_) {
        ensemble_processor_->stop_continuous_learning();
    }

    // Cleanup components
    if (ensemble_processor_) {
        ensemble_processor_->cleanup();
        ensemble_processor_.reset();
    }

    if (adaptive_learning_) {
        adaptive_learning_->cleanup();
        adaptive_learning_.reset();
    }

    if (api_client_) {
        api_client_->cleanup();
        api_client_.reset();
    }

    feature_serializer_.reset();
    integration_.reset();

    initialized_ = false;
    std::cout << "Learning system manager cleaned up" << std::endl;
}

LearningSystemManager::LearningSystemConfig LearningSystemManager::load_config_from_settings(const SettingsManager& settings) {
    LearningSystemConfig config;

    // Load API configuration
    config.enabled = settings.get<bool>("learning_api.enabled", false);

    if (config.enabled) {
        config.api_config = convert_settings_to_api_config(settings.get_settings()["learning_api"]);

        // Load ensemble configuration if ensemble is enabled
        bool ensemble_enabled = settings.get<bool>("ensemble.enabled", false);
        if (ensemble_enabled) {
            config.ensemble_config = convert_settings_to_ensemble_config(settings.get_settings()["ensemble"]);
        }

        // Load adaptive learning configuration
        config.adaptive_learning_db_url = settings.get<std::string>("adaptive_learning.database_url",
                                                                   "postgresql://localhost/voicestand_learning");
        config.enable_real_time_learning = settings.get<bool>("adaptive_learning.enabled", false);
    }

    return config;
}

void LearningSystemManager::integrate_with_whisper_processor(WhisperProcessor* processor) {
    if (!initialized_ || !integration_ || !processor) {
        return;
    }

    integration_->integrate_with_whisper_processor(processor);
    processor->set_learning_integration(integration_);

    std::cout << "Learning system integrated with WhisperProcessor" << std::endl;
}

void LearningSystemManager::submit_correction(const std::string& recognized_text, const std::string& correct_text) {
    if (!initialized_ || !integration_) {
        return;
    }

    integration_->handle_training_correction(recognized_text, correct_text);

    // Update statistics
    std::lock_guard<std::mutex> lock(stats_mutex_);
    current_stats_.total_corrections_submitted++;

    std::cout << "Submitted correction: '" << recognized_text << "' -> '" << correct_text << "'" << std::endl;
}

void LearningSystemManager::request_model_updates() {
    if (!initialized_ || !integration_) {
        return;
    }

    integration_->request_model_updates();
    std::cout << "Requested model updates from learning API" << std::endl;
}

void LearningSystemManager::update_speaker_profile(const std::string& speaker_id,
                                                  const std::vector<std::string>& sample_texts) {
    if (!initialized_ || !adaptive_learning_) {
        return;
    }

    // Create recognition contexts for speaker adaptation
    for (const auto& text : sample_texts) {
        AdaptiveLearningSystem::RecognitionContext context;
        context.text = text;
        context.speaker_id = speaker_id;
        context.confidence = 1.0;  // High confidence for manual samples
        context.domain = "general";
        context.timestamp = std::chrono::steady_clock::now();

        adaptive_learning_->record_recognition(context);
    }

    std::cout << "Updated speaker profile for " << speaker_id
              << " with " << sample_texts.size() << " samples" << std::endl;
}

LearningSystemManager::LearningStats LearningSystemManager::get_learning_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    LearningStats stats = current_stats_;

    // Update with current component stats
    if (api_client_) {
        const auto& api_stats = api_client_->get_stats();
        stats.total_recognitions_submitted = api_stats.total_requests.load();
        stats.api_online = api_client_->is_online();
        stats.pending_api_requests = api_client_->get_pending_requests();
        stats.offline_requests_queued = api_client_->get_offline_queue_size();
        stats.last_api_contact = api_stats.last_successful_request;
    }

    if (ensemble_processor_) {
        stats.current_ensemble_accuracy = ensemble_processor_->get_current_accuracy();
    }

    return stats;
}

void LearningSystemManager::enable_learning_system(bool enable) {
    config_.enabled = enable;

    if (initialized_) {
        if (enable) {
            // Re-initialize if currently disabled
            if (!api_client_) {
                initialize();
            }
        } else {
            // Cleanup components but keep structure
            if (adaptive_learning_) {
                adaptive_learning_->stop_real_time_learning();
            }
            if (ensemble_processor_) {
                ensemble_processor_->stop_continuous_learning();
            }
        }
    }

    std::cout << "Learning system " << (enable ? "enabled" : "disabled") << std::endl;
}

void LearningSystemManager::enable_real_time_learning(bool enable) {
    config_.enable_real_time_learning = enable;

    if (initialized_) {
        if (enable) {
            if (adaptive_learning_) {
                adaptive_learning_->start_real_time_learning();
            }
            if (ensemble_processor_) {
                ensemble_processor_->start_continuous_learning();
            }
        } else {
            if (adaptive_learning_) {
                adaptive_learning_->stop_real_time_learning();
            }
            if (ensemble_processor_) {
                ensemble_processor_->stop_continuous_learning();
            }
        }
    }

    std::cout << "Real-time learning " << (enable ? "enabled" : "disabled") << std::endl;
}

void LearningSystemManager::enable_ensemble_processing(bool enable) {
    if (enable && !ensemble_processor_ && initialized_) {
        // Initialize ensemble processor
        ensemble_processor_ = std::make_shared<EnsembleWhisperProcessor>(config_.ensemble_config);
        if (ensemble_processor_->initialize()) {
            ensemble_processor_->set_learning_integration(integration_);
            if (config_.enable_real_time_learning) {
                ensemble_processor_->start_continuous_learning();
            }
            std::cout << "Ensemble processing enabled" << std::endl;
        } else {
            ensemble_processor_.reset();
            std::cerr << "Failed to enable ensemble processing" << std::endl;
        }
    } else if (!enable && ensemble_processor_) {
        ensemble_processor_->cleanup();
        ensemble_processor_.reset();
        std::cout << "Ensemble processing disabled" << std::endl;
    }
}

void LearningSystemManager::setup_integration_callbacks() {
    if (!integration_) {
        return;
    }

    // Set up session and user context
    integration_->set_session_id("session_" + std::to_string(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()));
    integration_->set_domain("general");
}

void LearningSystemManager::update_stats() {
    // This method would update internal statistics
    // Called periodically or on events
}

// Factory function
std::shared_ptr<LearningSystemManager> create_learning_system_from_settings(const SettingsManager& settings) {
    auto config = LearningSystemManager::load_config_from_settings(settings);
    auto manager = std::make_shared<LearningSystemManager>(config);

    if (!manager->initialize()) {
        std::cerr << "Failed to initialize learning system from settings" << std::endl;
        return nullptr;
    }

    return manager;
}

// Configuration conversion utilities
LearningApiConfig convert_settings_to_api_config(const Json::Value& settings) {
    LearningApiConfig config;

    if (settings.isMember("base_url")) {
        config.base_url = settings["base_url"].asString();
    }

    if (settings.isMember("api_version")) {
        config.api_version = settings["api_version"].asString();
    }

    if (settings.isMember("api_key")) {
        config.api_key = settings["api_key"].asString();
    }

    if (settings.isMember("connection_timeout_seconds")) {
        config.connection_timeout = std::chrono::seconds(settings["connection_timeout_seconds"].asInt());
    }

    if (settings.isMember("request_timeout_seconds")) {
        config.request_timeout = std::chrono::seconds(settings["request_timeout_seconds"].asInt());
    }

    if (settings.isMember("max_retries")) {
        config.max_retries = settings["max_retries"].asInt();
    }

    if (settings.isMember("enable_async_processing")) {
        config.enable_async_processing = settings["enable_async_processing"].asBool();
    }

    if (settings.isMember("max_concurrent_requests")) {
        config.max_concurrent_requests = settings["max_concurrent_requests"].asUInt();
    }

    if (settings.isMember("enable_batching")) {
        config.enable_batching = settings["enable_batching"].asBool();
    }

    if (settings.isMember("batch_size")) {
        config.batch_size = settings["batch_size"].asUInt();
    }

    if (settings.isMember("enable_offline_queue")) {
        config.enable_offline_queue = settings["enable_offline_queue"].asBool();
    }

    if (settings.isMember("offline_queue_max_size")) {
        config.offline_queue_max_size = settings["offline_queue_max_size"].asUInt();
    }

    return config;
}

EnsembleWhisperProcessor::EnsembleConfig convert_settings_to_ensemble_config(const Json::Value& settings) {
    EnsembleWhisperProcessor::EnsembleConfig config;

    if (settings.isMember("accuracy_target")) {
        config.accuracy_target = settings["accuracy_target"].asDouble();
    }

    if (settings.isMember("confidence_threshold")) {
        config.confidence_threshold = settings["confidence_threshold"].asDouble();
    }

    if (settings.isMember("min_ensemble_size")) {
        config.min_ensemble_size = settings["min_ensemble_size"].asUInt();
    }

    if (settings.isMember("max_ensemble_size")) {
        config.max_ensemble_size = settings["max_ensemble_size"].asUInt();
    }

    if (settings.isMember("enable_uk_dialect_optimization")) {
        config.enable_uk_dialect_optimization = settings["enable_uk_dialect_optimization"].asBool();
    }

    if (settings.isMember("enable_model_caching")) {
        config.enable_model_caching = settings["enable_model_caching"].asBool();
    }

    if (settings.isMember("enable_parallel_inference")) {
        config.enable_parallel_inference = settings["enable_parallel_inference"].asBool();
    }

    if (settings.isMember("max_parallel_models")) {
        config.max_parallel_models = settings["max_parallel_models"].asUInt();
    }

    if (settings.isMember("learning_update_interval_seconds")) {
        config.learning_update_interval = std::chrono::seconds(settings["learning_update_interval_seconds"].asInt());
    }

    // Convert learning mode string to enum
    if (settings.isMember("learning_mode")) {
        std::string mode = settings["learning_mode"].asString();
        if (mode == "CONSERVATIVE") {
            config.learning_mode = EnsembleWhisperProcessor::LearningMode::CONSERVATIVE;
        } else if (mode == "BALANCED") {
            config.learning_mode = EnsembleWhisperProcessor::LearningMode::BALANCED;
        } else if (mode == "EXPERIMENTAL") {
            config.learning_mode = EnsembleWhisperProcessor::LearningMode::EXPERIMENTAL;
        } else if (mode == "ADAPTIVE") {
            config.learning_mode = EnsembleWhisperProcessor::LearningMode::ADAPTIVE;
        }
    }

    return config;
}

} // namespace vtt