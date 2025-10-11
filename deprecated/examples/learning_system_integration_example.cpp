/*
 * VoiceStand Learning System Integration Example
 *
 * This example demonstrates how to integrate the complete learning system
 * with the VoiceStand C++ application.
 */

#include "core/learning_system_manager.h"
#include "core/whisper_processor.h"
#include "core/settings_manager.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace vtt;

void example_basic_integration() {
    std::cout << "=== Basic Learning System Integration Example ===" << std::endl;

    // Load settings
    SettingsManager& settings = SettingsManager::instance();
    settings.load_settings();

    // Enable learning system in settings
    settings.set("learning_api.enabled", true);
    settings.set("learning_api.base_url", "http://localhost:5000");
    settings.set("learning_api.api_key", "your-api-key-here");
    settings.save_settings();

    // Create learning system from settings
    auto learning_manager = create_learning_system_from_settings(settings);
    if (!learning_manager) {
        std::cerr << "Failed to create learning system" << std::endl;
        return;
    }

    // Create whisper processor
    WhisperProcessor whisper_processor;
    WhisperConfig whisper_config;
    whisper_config.model_path = "models/ggml-base.bin";
    whisper_config.language = "en";

    if (!whisper_processor.initialize(whisper_config)) {
        std::cerr << "Failed to initialize whisper processor" << std::endl;
        return;
    }

    // Integrate learning system with whisper processor
    learning_manager->integrate_with_whisper_processor(&whisper_processor);

    std::cout << "Learning system integrated successfully!" << std::endl;

    // Simulate some audio processing
    std::vector<float> dummy_audio(16000, 0.1f); // 1 second of dummy audio
    whisper_processor.start_streaming();
    whisper_processor.process_audio(dummy_audio.data(), dummy_audio.size(), 16000);

    // Wait a moment for processing
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Submit a manual correction
    learning_manager->submit_correction("hello word", "hello world");

    // Request model updates
    learning_manager->request_model_updates();

    // Show statistics
    auto stats = learning_manager->get_learning_stats();
    std::cout << "Learning Statistics:" << std::endl;
    std::cout << "  API Online: " << (stats.api_online ? "Yes" : "No") << std::endl;
    std::cout << "  Total Recognitions Submitted: " << stats.total_recognitions_submitted << std::endl;
    std::cout << "  Total Corrections Submitted: " << stats.total_corrections_submitted << std::endl;
    std::cout << "  Pending API Requests: " << stats.pending_api_requests << std::endl;

    whisper_processor.stop_streaming();
    whisper_processor.cleanup();
}

void example_ensemble_processing() {
    std::cout << "\n=== Ensemble Processing Example ===" << std::endl;

    SettingsManager& settings = SettingsManager::instance();

    // Configure ensemble processing
    settings.set("ensemble.enabled", true);
    settings.set("ensemble.learning_mode", "ADAPTIVE");
    settings.set("ensemble.min_ensemble_size", 3);
    settings.set("ensemble.max_ensemble_size", 5);
    settings.set("ensemble.enable_uk_dialect_optimization", true);

    // Configure model paths for ensemble
    Json::Value model_paths(Json::arrayValue);
    model_paths.append("models/ggml-small.bin");
    model_paths.append("models/ggml-medium.bin");
    model_paths.append("models/ggml-large.bin");
    settings.set("ensemble.model_paths", model_paths);

    settings.save_settings();

    auto learning_manager = create_learning_system_from_settings(settings);
    if (!learning_manager) {
        std::cerr << "Failed to create learning system with ensemble" << std::endl;
        return;
    }

    auto ensemble_processor = learning_manager->get_ensemble_processor();
    if (ensemble_processor) {
        std::cout << "Ensemble processor created with "
                  << ensemble_processor->get_config().model_paths.size() << " models" << std::endl;

        // Simulate ensemble recognition
        std::vector<float> audio_data(16000 * 2, 0.1f); // 2 seconds of audio
        auto result = ensemble_processor->recognize(audio_data.data(), audio_data.size(), 16000);

        std::cout << "Ensemble Recognition Result:" << std::endl;
        std::cout << "  Text: " << result.text << std::endl;
        std::cout << "  Confidence: " << result.confidence << std::endl;
        std::cout << "  Model Used: " << result.model_used << std::endl;
        std::cout << "  Is Ensemble Result: " << (result.is_ensemble_result ? "Yes" : "No") << std::endl;
        std::cout << "  Ensemble Agreement: " << result.ensemble_agreement << std::endl;
        std::cout << "  Processing Time: " << result.processing_time.count() << "ms" << std::endl;

        // Report ground truth for learning
        ensemble_processor->report_ground_truth(result.text, "correct transcription");
    } else {
        std::cout << "Ensemble processor not available" << std::endl;
    }
}

void example_adaptive_learning() {
    std::cout << "\n=== Adaptive Learning Example ===" << std::endl;

    SettingsManager& settings = SettingsManager::instance();

    // Configure adaptive learning
    settings.set("adaptive_learning.enabled", true);
    settings.set("adaptive_learning.database_url", "postgresql://localhost/voicestand_learning");
    settings.set("adaptive_learning.pattern_confidence_threshold", 0.7);
    settings.set("adaptive_learning.enable_uk_english_learning", true);

    auto learning_manager = create_learning_system_from_settings(settings);
    if (!learning_manager) {
        std::cerr << "Failed to create adaptive learning system" << std::endl;
        return;
    }

    auto adaptive_learning = learning_manager->get_adaptive_learning();
    if (adaptive_learning) {
        std::cout << "Adaptive learning system initialized" << std::endl;

        // Create some sample recognition contexts
        AdaptiveLearningSystem::RecognitionContext context1;
        context1.text = "Hello world";
        context1.confidence = 0.95;
        context1.domain = "general";
        context1.speaker_id = "user1";
        context1.timestamp = std::chrono::steady_clock::now();
        context1.acoustic_features = {0.5, 0.3, 0.8, 0.2}; // Dummy features

        AdaptiveLearningSystem::RecognitionContext context2;
        context2.text = "Goodbye world";
        context2.confidence = 0.88;
        context2.domain = "general";
        context2.speaker_id = "user1";
        context2.timestamp = std::chrono::steady_clock::now();
        context2.acoustic_features = {0.4, 0.6, 0.7, 0.3}; // Dummy features

        // Record contexts
        adaptive_learning->record_recognition(context1);
        adaptive_learning->record_recognition(context2);

        // Analyze patterns
        auto insights = adaptive_learning->analyze_patterns();
        std::cout << "Generated " << insights.size() << " learning insights" << std::endl;

        for (const auto& insight : insights) {
            std::cout << "  Insight: " << insight.description
                      << " (confidence: " << insight.confidence << ")" << std::endl;
        }

        // Get optimal model weights
        auto optimal_weights = adaptive_learning->get_optimal_model_weights();
        std::cout << "Optimal model weights:" << std::endl;
        for (const auto& weight : optimal_weights) {
            std::cout << "  " << weight.first << ": " << weight.second << std::endl;
        }

        // Get model recommendations
        auto recommendations = adaptive_learning->recommend_model_combinations();
        std::cout << "Recommended models:" << std::endl;
        for (const auto& model : recommendations) {
            std::cout << "  " << model << std::endl;
        }
    } else {
        std::cout << "Adaptive learning system not available" << std::endl;
    }
}

void example_audio_feature_extraction() {
    std::cout << "\n=== Audio Feature Extraction Example ===" << std::endl;

    AudioFeatureSerializer serializer;

    // Create some dummy audio data
    std::vector<float> audio_samples(16000); // 1 second at 16kHz
    for (size_t i = 0; i < audio_samples.size(); ++i) {
        // Generate a simple sine wave
        audio_samples[i] = 0.5 * std::sin(2.0 * M_PI * 440.0 * i / 16000.0);
    }

    // Extract audio features
    auto audio_features = serializer.extract_audio_features(
        audio_samples.data(), audio_samples.size(), 16000);

    std::cout << "Extracted Audio Features:" << std::endl;
    std::cout << "  Energy: " << audio_features.energy << std::endl;
    std::cout << "  RMS: " << audio_features.rms << std::endl;
    std::cout << "  Zero Crossing Rate: " << audio_features.zero_crossing_rate << std::endl;
    std::cout << "  Spectral Centroid: " << audio_features.spectral_centroid << std::endl;
    std::cout << "  SNR: " << audio_features.snr << " dB" << std::endl;
    std::cout << "  MFCC Coefficients: " << audio_features.mfcc_coefficients.size() << std::endl;

    // Create recognition features
    AudioFeatureSerializer::RecognitionFeatures recognition_features;
    recognition_features.text = "Test transcription";
    recognition_features.confidence = 0.92;
    recognition_features.model_name = "whisper-base";
    recognition_features.processing_time = std::chrono::milliseconds(250);

    // Create context features
    AudioFeatureSerializer::ContextFeatures context_features;
    context_features.audio_source_type = "microphone";
    context_features.user_id = "test_user";
    context_features.session_id = "session_123";
    context_features.device_type = "desktop";

    // Serialize complete submission
    auto json_submission = serializer.serialize_complete_submission(
        audio_features, recognition_features, context_features);

    std::cout << "JSON submission created with "
              << json_submission.getMemberNames().size() << " top-level fields" << std::endl;

    // Convert to feature vector for ML
    auto feature_vector = serializer.create_combined_feature_vector(
        audio_features, recognition_features, context_features);

    std::cout << "Combined feature vector has " << feature_vector.size() << " dimensions" << std::endl;
}

void example_api_configuration() {
    std::cout << "\n=== API Configuration Example ===" << std::endl;

    // Manual API configuration
    LearningApiConfig api_config;
    api_config.base_url = "https://your-learning-api.com";
    api_config.api_version = "v1";
    api_config.api_key = "sk-your-secret-key";
    api_config.connection_timeout = std::chrono::seconds(15);
    api_config.request_timeout = std::chrono::seconds(45);
    api_config.max_retries = 5;
    api_config.enable_async_processing = true;
    api_config.max_concurrent_requests = 10;
    api_config.enable_batching = true;
    api_config.batch_size = 20;
    api_config.enable_offline_queue = true;
    api_config.offline_queue_max_size = 50000;

    // Create API client with custom configuration
    LearningApiClient api_client(api_config);
    if (!api_client.initialize()) {
        std::cerr << "Failed to initialize API client" << std::endl;
        return;
    }

    std::cout << "API client configured:" << std::endl;
    std::cout << "  Base URL: " << api_config.base_url << std::endl;
    std::cout << "  Max concurrent requests: " << api_config.max_concurrent_requests << std::endl;
    std::cout << "  Batching enabled: " << (api_config.enable_batching ? "Yes" : "No") << std::endl;
    std::cout << "  Offline queue enabled: " << (api_config.enable_offline_queue ? "Yes" : "No") << std::endl;
    std::cout << "  Online status: " << (api_client.is_online() ? "Online" : "Offline") << std::endl;

    // Test API submission
    RecognitionSubmission submission;
    submission.text = "Test recognition submission";
    submission.confidence = 0.89;
    submission.model_used = "whisper-base";
    submission.session_id = "test_session";
    submission.timestamp = std::chrono::steady_clock::now();
    submission.audio_features = {0.1, 0.2, 0.3, 0.4, 0.5};

    // Submit asynchronously with callback
    api_client.submit_recognition_callback(submission, [](const HttpResponse& response) {
        if (response.success) {
            std::cout << "Recognition submitted successfully (HTTP " << response.status_code << ")" << std::endl;
        } else {
            std::cout << "Recognition submission failed: " << response.error_message << std::endl;
        }
    });

    // Wait for async operations to complete
    std::this_thread::sleep_for(std::chrono::seconds(2));

    api_client.cleanup();
}

int main() {
    std::cout << "VoiceStand Learning System Integration Examples" << std::endl;
    std::cout << "===============================================" << std::endl;

    try {
        example_basic_integration();
        example_ensemble_processing();
        example_adaptive_learning();
        example_audio_feature_extraction();
        example_api_configuration();

        std::cout << "\n=== All Examples Completed Successfully ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in examples: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}