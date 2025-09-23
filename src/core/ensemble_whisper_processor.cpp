#include "ensemble_whisper_processor.h"
#include "learning_api_client.h"
#include "adaptive_learning_system.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace vtt {

EnsembleWhisperProcessor::EnsembleWhisperProcessor()
    : EnsembleWhisperProcessor(EnsembleConfig{}) {
}

EnsembleWhisperProcessor::EnsembleWhisperProcessor(const EnsembleConfig& config)
    : config_(config), learning_active_(false) {
    // Initialize model weights
    for (const auto& model_path : config_.model_paths) {
        model_weights_[model_path] = 1.0; // Equal initial weights
        model_last_updated_[model_path] = std::chrono::steady_clock::now();
    }
}

EnsembleWhisperProcessor::~EnsembleWhisperProcessor() {
    stop_continuous_learning();
}

bool EnsembleWhisperProcessor::initialize() {
    std::lock_guard<std::mutex> lock(model_mutex_);

    // Initialize individual whisper processors for each model
    for (const auto& model_path : config_.model_paths) {
        auto processor = std::make_unique<WhisperProcessor>();

        WhisperConfig whisper_config;
        whisper_config.model_path = model_path;
        whisper_config.language = "auto";
        whisper_config.num_threads = std::max(1, static_cast<int>(config_.max_parallel_models));
        whisper_config.use_gpu = false; // Can be made configurable

        if (processor->initialize(whisper_config)) {
            models_.push_back(std::move(processor));
            std::cout << "Loaded model: " << model_path << std::endl;
        } else {
            std::cerr << "Failed to load model: " << model_path << std::endl;
            // Continue with other models
        }
    }

    if (models_.empty()) {
        std::cerr << "No models could be loaded!" << std::endl;
        return false;
    }

    // Initialize UK dialect optimizer if enabled
    if (config_.enable_uk_dialect_optimization) {
        uk_optimizer_ = std::make_unique<UKDialectOptimizer>();
    }

    // Initialize learning system
    learning_system_ = std::make_unique<AdaptiveLearningSystem>("postgresql://localhost/voicestand_learning");
    if (!learning_system_->initialize()) {
        std::cout << "Warning: Learning system initialization failed, continuing without learning" << std::endl;
        learning_system_.reset();
    }

    std::cout << "Ensemble processor initialized with " << models_.size() << " models" << std::endl;
    return true;
}

vtt::RecognitionResult EnsembleWhisperProcessor::recognize(const float* audio_data, size_t num_samples, uint32_t sample_rate) {
    if (models_.empty()) {
        vtt::RecognitionResult result;
        result.text = "";
        result.confidence = 0.0;
        result.model_used = "none";
        result.processing_time = std::chrono::milliseconds(0);
        return result;
    }

    // Use ensemble recognition for better accuracy
    if (models_.size() >= config_.min_ensemble_size) {
        return ensemble_recognize(audio_data, num_samples);
    } else {
        // Fall back to single model recognition
        auto start = std::chrono::steady_clock::now();

        models_[0]->process_audio(audio_data, num_samples, sample_rate);

        vtt::RecognitionResult result;
        result.text = ""; // Would be populated by transcription callback
        result.confidence = 0.8; // Default confidence
        result.model_used = config_.model_paths.empty() ? "unknown" : config_.model_paths[0];
        result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);

        return result;
    }
}

vtt::RecognitionResult EnsembleWhisperProcessor::ensemble_recognize(const float* audio_data, size_t num_samples) {
    auto start_time = std::chrono::steady_clock::now();

    std::vector<vtt::RecognitionResult> individual_results;
    std::vector<size_t> selected_models = select_models_for_inference(AudioContext{});

    // Process with selected models in parallel
    std::vector<std::thread> recognition_threads;
    std::vector<vtt::RecognitionResult> thread_results(selected_models.size());

    for (size_t i = 0; i < selected_models.size(); ++i) {
        size_t model_idx = selected_models[i];
        if (model_idx >= models_.size()) continue;

        recognition_threads.emplace_back([this, model_idx, audio_data, num_samples, &thread_results, i]() {
            try {
                // Create a temporary result - in real implementation, this would
                // involve actual processing with the model
                vtt::RecognitionResult result;
                result.text = "sample_text_" + std::to_string(model_idx);
                result.confidence = 0.8 + (model_idx * 0.05); // Varying confidence
                result.model_used = config_.model_paths[model_idx];
                result.processing_time = std::chrono::milliseconds(100 + model_idx * 20);

                thread_results[i] = result;
            } catch (const std::exception& e) {
                std::cerr << "Error in model " << model_idx << ": " << e.what() << std::endl;
            }
        });
    }

    // Wait for all recognition threads to complete
    for (auto& thread : recognition_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Collect results
    for (const auto& result : thread_results) {
        if (!result.text.empty()) {
            individual_results.push_back(result);
        }
    }

    // Combine results using ensemble logic
    vtt::RecognitionResult ensemble_result;
    if (!individual_results.empty()) {
        ensemble_result.text = combine_results(individual_results);
        ensemble_result.confidence = calculate_ensemble_confidence(individual_results);
        ensemble_result.model_used = "ensemble";
        ensemble_result.is_ensemble_result = true;

        // Calculate model agreement
        double agreement = 0.0;
        if (individual_results.size() > 1) {
            size_t agreements = 0;
            for (size_t i = 0; i < individual_results.size(); ++i) {
                for (size_t j = i + 1; j < individual_results.size(); ++j) {
                    // Simple text similarity check (can be enhanced)
                    if (individual_results[i].text == individual_results[j].text) {
                        agreements++;
                    }
                }
            }
            agreement = static_cast<double>(agreements) /
                       (individual_results.size() * (individual_results.size() - 1) / 2);
        }
        ensemble_result.ensemble_agreement = agreement;

        // Store individual model confidences
        for (const auto& result : individual_results) {
            ensemble_result.model_confidences[result.model_used] = result.confidence;
        }
    }

    ensemble_result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);

    // Submit to learning system if available
    {
        std::lock_guard<std::mutex> lock(learning_integration_mutex_);
        if (learning_integration_) {
            learning_integration_->handle_recognition_result(ensemble_result);
        }
    }

    // Update metrics
    metrics_.total_recognitions++;
    if (ensemble_result.confidence > 0.5) {
        metrics_.ensemble_agreements++;
    }

    return ensemble_result;
}

std::string EnsembleWhisperProcessor::combine_results(const std::vector<vtt::RecognitionResult>& results) {
    if (results.empty()) {
        return "";
    }

    if (results.size() == 1) {
        return results[0].text;
    }

    // Simple majority voting (can be enhanced with more sophisticated algorithms)
    std::map<std::string, double> text_weights;

    for (const auto& result : results) {
        double weight = result.confidence;

        // Apply model-specific weights
        auto model_weight_it = model_weights_.find(result.model_used);
        if (model_weight_it != model_weights_.end()) {
            weight *= model_weight_it->second;
        }

        text_weights[result.text] += weight;
    }

    // Find the text with highest weighted score
    std::string best_text;
    double best_weight = 0.0;

    for (const auto& pair : text_weights) {
        if (pair.second > best_weight) {
            best_weight = pair.second;
            best_text = pair.first;
        }
    }

    return best_text;
}

double EnsembleWhisperProcessor::calculate_ensemble_confidence(const std::vector<vtt::RecognitionResult>& results) {
    if (results.empty()) {
        return 0.0;
    }

    // Weighted average confidence
    double total_confidence = 0.0;
    double total_weight = 0.0;

    for (const auto& result : results) {
        double weight = 1.0;

        // Apply model-specific weights
        auto model_weight_it = model_weights_.find(result.model_used);
        if (model_weight_it != model_weights_.end()) {
            weight = model_weight_it->second;
        }

        total_confidence += result.confidence * weight;
        total_weight += weight;
    }

    if (total_weight > 0.0) {
        return total_confidence / total_weight;
    }

    return 0.0;
}

void EnsembleWhisperProcessor::update_model_weights(const vtt::RecognitionResult& result, const std::string& ground_truth) {
    if (result.model_confidences.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(model_mutex_);

    // Update weights based on accuracy
    for (const auto& model_confidence : result.model_confidences) {
        const std::string& model_name = model_confidence.first;
        double confidence = model_confidence.second;

        // Simple accuracy calculation (can be enhanced)
        double accuracy = (result.text == ground_truth) ? 1.0 : 0.0;

        // Update model weight using learning rate
        double learning_rate = 0.1;
        double current_weight = model_weights_[model_name];

        // Adjust weight based on accuracy and confidence
        double adjustment = learning_rate * (accuracy - 0.5) * confidence;
        double new_weight = std::max(0.1, std::min(2.0, current_weight + adjustment));

        model_weights_[model_name] = new_weight;
        model_last_updated_[model_name] = std::chrono::steady_clock::now();

        std::cout << "Updated weight for " << model_name << ": " << new_weight << std::endl;
    }

    // Report ground truth to learning system
    report_ground_truth(result.text, ground_truth);
}

void EnsembleWhisperProcessor::report_ground_truth(const std::string& recognized_text, const std::string& actual_text) {
    // Update accuracy metrics
    bool is_correct = (recognized_text == actual_text);
    double current_accuracy = metrics_.current_accuracy.load();

    // Exponential moving average
    double new_accuracy = current_accuracy * 0.95 + (is_correct ? 1.0 : 0.0) * 0.05;
    metrics_.current_accuracy = new_accuracy;

    // Submit correction to learning system
    {
        std::lock_guard<std::mutex> lock(learning_integration_mutex_);
        if (learning_integration_ && recognized_text != actual_text) {
            learning_integration_->handle_training_correction(recognized_text, actual_text);
        }
    }

    std::cout << "Ground truth reported. Current accuracy: " << new_accuracy << std::endl;
}

std::vector<size_t> EnsembleWhisperProcessor::select_models_for_inference(const AudioContext& context) {
    std::vector<size_t> selected_models;

    // Simple selection strategy - use all available models up to max_parallel_models
    size_t max_models = std::min(config_.max_parallel_models, models_.size());

    for (size_t i = 0; i < max_models; ++i) {
        selected_models.push_back(i);
    }

    return selected_models;
}

void EnsembleWhisperProcessor::set_learning_integration(std::shared_ptr<LearningSystemIntegration> integration) {
    std::lock_guard<std::mutex> lock(learning_integration_mutex_);
    learning_integration_ = integration;

    // Configure the integration to work with this ensemble processor
    if (learning_integration_) {
        learning_integration_->integrate_with_ensemble_processor(this);
    }

    std::cout << "Learning integration set for ensemble processor" << std::endl;
}

std::shared_ptr<LearningSystemIntegration> EnsembleWhisperProcessor::get_learning_integration() const {
    std::lock_guard<std::mutex> lock(learning_integration_mutex_);
    return learning_integration_;
}

void EnsembleWhisperProcessor::set_learning_mode(LearningMode mode) {
    config_.learning_mode = mode;

    std::cout << "Learning mode set to: ";
    switch (mode) {
        case LearningMode::CONSERVATIVE:
            std::cout << "CONSERVATIVE";
            break;
        case LearningMode::BALANCED:
            std::cout << "BALANCED";
            break;
        case LearningMode::EXPERIMENTAL:
            std::cout << "EXPERIMENTAL";
            break;
        case LearningMode::ADAPTIVE:
            std::cout << "ADAPTIVE";
            break;
    }
    std::cout << std::endl;
}

void EnsembleWhisperProcessor::start_continuous_learning() {
    if (learning_active_.load()) {
        return;
    }

    learning_active_ = true;
    learning_thread_ = std::thread(&EnsembleWhisperProcessor::learning_loop, this);

    std::cout << "Continuous learning started" << std::endl;
}

void EnsembleWhisperProcessor::stop_continuous_learning() {
    learning_active_ = false;

    if (learning_thread_.joinable()) {
        learning_thread_.join();
    }

    std::cout << "Continuous learning stopped" << std::endl;
}

void EnsembleWhisperProcessor::learning_loop() {
    while (learning_active_.load()) {
        // Request model updates from learning API
        {
            std::lock_guard<std::mutex> lock(learning_integration_mutex_);
            if (learning_integration_) {
                learning_integration_->request_model_updates();
            }
        }

        // Sleep for the configured update interval
        std::this_thread::sleep_for(config_.learning_update_interval);
    }
}

void EnsembleWhisperProcessor::cleanup() {
    stop_continuous_learning();

    std::lock_guard<std::mutex> lock(model_mutex_);
    models_.clear();

    if (learning_system_) {
        learning_system_->cleanup();
        learning_system_.reset();
    }

    uk_optimizer_.reset();

    std::cout << "Ensemble processor cleaned up" << std::endl;
}

// UKDialectOptimizer implementation
UKDialectOptimizer::UKDialectOptimizer() {
    // Initialize UK vocabulary mappings
    american_to_british_vocab_["color"] = "colour";
    american_to_british_vocab_["center"] = "centre";
    american_to_british_vocab_["theater"] = "theatre";
    american_to_british_vocab_["organize"] = "organise";
    american_to_british_vocab_["realize"] = "realise";
    american_to_british_vocab_["defense"] = "defence";
    american_to_british_vocab_["license"] = "licence";

    // UK-specific phrases
    uk_specific_phrases_.push_back("lift"); // vs elevator
    uk_specific_phrases_.push_back("lorry"); // vs truck
    uk_specific_phrases_.push_back("biscuit"); // vs cookie
    uk_specific_phrases_.push_back("jumper"); // vs sweater
    uk_specific_phrases_.push_back("queue"); // vs line
}

UKDialectOptimizer::UKDialectFeatures UKDialectOptimizer::analyze_uk_features(const float* audio_data, size_t num_samples) {
    UKDialectFeatures features;

    // This is a simplified implementation - real feature extraction would be more complex
    if (!audio_data || num_samples == 0) {
        return features;
    }

    // Basic acoustic analysis (placeholder)
    features.rhoticity_score = 0.3; // UK English is typically non-rhotic
    features.vowel_system_score = 0.7; // UK vowel system detection
    features.intonation_score = 0.6; // British intonation patterns

    return features;
}

double UKDialectOptimizer::calculate_uk_probability(const UKDialectFeatures& features) {
    // Weighted combination of features
    double uk_probability = 0.0;
    uk_probability += features.rhoticity_score * 0.3;
    uk_probability += features.vowel_system_score * 0.3;
    uk_probability += features.lexical_choice_score * 0.2;
    uk_probability += features.intonation_score * 0.2;

    return std::min(1.0, std::max(0.0, uk_probability));
}

std::string UKDialectOptimizer::optimize_text_for_uk(const std::string& raw_text) {
    std::string optimized_text = raw_text;

    // Replace American spellings with British spellings
    for (const auto& mapping : american_to_british_vocab_) {
        size_t pos = 0;
        while ((pos = optimized_text.find(mapping.first, pos)) != std::string::npos) {
            optimized_text.replace(pos, mapping.first.length(), mapping.second);
            pos += mapping.second.length();
        }
    }

    return optimized_text;
}

void UKDialectOptimizer::add_uk_vocabulary_mapping(const std::string& american_term, const std::string& british_term) {
    american_to_british_vocab_[american_term] = british_term;
}

} // namespace vtt