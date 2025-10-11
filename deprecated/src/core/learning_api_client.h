#pragma once

#include <string>
#include <memory>
#include <functional>
#include <future>
#include <chrono>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <queue>
#include <thread>
#include <condition_variable>
#include <json/json.h>
#include "whisper_processor.h"

namespace vtt {

// Forward declarations
struct TranscriptionResult;

// RecognitionResult struct to avoid circular dependencies
struct RecognitionResult {
    std::string text;
    double confidence;
    std::string model_used;
    std::chrono::milliseconds processing_time;
    std::map<std::string, double> model_confidences; // Per-model confidence
    bool is_ensemble_result = false;
    double ensemble_agreement = 0.0; // Agreement between models
};

// HTTP Response structure
struct HttpResponse {
    int status_code = 0;
    std::string body;
    std::map<std::string, std::string> headers;
    bool success = false;
    std::string error_message;
    std::chrono::milliseconds duration{0};
};

// API Request types
struct RecognitionSubmission {
    std::string text;
    double confidence;
    std::vector<double> audio_features;
    std::string model_used;
    std::chrono::steady_clock::time_point timestamp;
    std::string session_id;
    bool is_final = true;
    std::string speaker_id = "";
    std::string domain = "general";
    std::map<std::string, double> metadata;
};

struct ModelWeightUpdate {
    std::string model_name;
    std::map<std::string, double> weights;
    double accuracy_score;
    std::chrono::steady_clock::time_point updated_at;
    std::string version;
    bool is_incremental = false;
};

struct TrainingDataSubmission {
    std::string recognized_text;
    std::string corrected_text;
    std::vector<double> audio_features;
    double confidence;
    std::string correction_type; // "spelling", "grammar", "vocabulary", etc.
    std::string context;
    std::chrono::steady_clock::time_point timestamp;
};

struct AccuracyFeedback {
    std::string recognition_id;
    double accuracy_score;
    std::string feedback_type; // "correct", "incorrect", "partial"
    std::vector<std::string> improvement_suggestions;
    std::chrono::steady_clock::time_point feedback_time;
};

// Configuration for API client
struct LearningApiConfig {
    std::string base_url = "http://localhost:5000";
    std::string api_version = "v1";
    std::string api_key = "";

    // Timeout settings
    std::chrono::seconds connection_timeout{10};
    std::chrono::seconds request_timeout{30};

    // Retry settings
    int max_retries = 3;
    std::chrono::milliseconds retry_delay{1000};

    // Async processing
    bool enable_async_processing = true;
    size_t max_concurrent_requests = 5;
    size_t queue_max_size = 1000;

    // Batch processing
    bool enable_batching = true;
    size_t batch_size = 10;
    std::chrono::milliseconds batch_timeout{5000};

    // Offline handling
    bool enable_offline_queue = true;
    size_t offline_queue_max_size = 10000;
    std::string offline_queue_file = "";
};

// Callback types for async operations
using RecognitionCallback = std::function<void(const HttpResponse&)>;
using ModelUpdateCallback = std::function<void(const std::vector<ModelWeightUpdate>&, const HttpResponse&)>;
using TrainingCallback = std::function<void(const HttpResponse&)>;
using AccuracyCallback = std::function<void(const AccuracyFeedback&, const HttpResponse&)>;

// Main API client class
class LearningApiClient {
public:
    explicit LearningApiClient(const LearningApiConfig& config = LearningApiConfig{});
    ~LearningApiClient();

    // Initialization and cleanup
    bool initialize();
    void cleanup();
    bool is_initialized() const { return initialized_; }

    // Synchronous API calls
    HttpResponse submit_recognition(const RecognitionSubmission& submission);
    HttpResponse get_model_weights(const std::string& model_name = "");
    HttpResponse submit_training_data(const TrainingDataSubmission& training_data);
    HttpResponse get_accuracy_feedback(const std::string& session_id = "");

    // Asynchronous API calls
    std::future<HttpResponse> submit_recognition_async(const RecognitionSubmission& submission);
    std::future<HttpResponse> get_model_weights_async(const std::string& model_name = "");
    std::future<HttpResponse> submit_training_data_async(const TrainingDataSubmission& training_data);

    // Callback-based async calls (non-blocking, fire-and-forget)
    void submit_recognition_callback(const RecognitionSubmission& submission,
                                   RecognitionCallback callback = nullptr);
    void get_model_weights_callback(const std::string& model_name,
                                  ModelUpdateCallback callback);
    void submit_training_data_callback(const TrainingDataSubmission& training_data,
                                     TrainingCallback callback = nullptr);

    // Batch operations
    void add_to_recognition_batch(const RecognitionSubmission& submission);
    void flush_recognition_batch();
    void add_to_training_batch(const TrainingDataSubmission& training_data);
    void flush_training_batch();

    // Configuration and status
    void update_config(const LearningApiConfig& new_config);
    const LearningApiConfig& get_config() const { return config_; }
    bool is_online() const { return online_status_.load(); }
    size_t get_queue_size() const;
    size_t get_pending_requests() const { return pending_requests_.load(); }

    // Offline handling
    void enable_offline_mode(bool enable);
    void process_offline_queue();
    size_t get_offline_queue_size() const;

    // Statistics
    struct ApiStats {
        std::atomic<size_t> total_requests{0};
        std::atomic<size_t> successful_requests{0};
        std::atomic<size_t> failed_requests{0};
        std::atomic<size_t> timeout_requests{0};
        std::atomic<double> average_response_time{0.0};
        std::atomic<size_t> offline_requests_queued{0};
        std::atomic<size_t> batch_requests_sent{0};
        std::chrono::steady_clock::time_point last_successful_request;
    };

    const ApiStats& get_stats() const { return stats_; }
    void reset_stats();

private:
    // Configuration
    LearningApiConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> running_{false};

    // HTTP client implementation (using libcurl)
    void* curl_handle_;  // CURL*
    void* curl_multi_;   // CURLM*

    // Threading
    std::vector<std::thread> worker_threads_;
    std::thread batch_processor_thread_;
    std::thread offline_processor_thread_;

    // Request queue management
    struct QueuedRequest {
        enum Type { RECOGNITION, MODEL_UPDATE, TRAINING, FEEDBACK };
        Type type;
        Json::Value payload;
        std::string endpoint;
        std::function<void(const HttpResponse&)> callback;
        std::chrono::steady_clock::time_point queued_at;
        int retry_count = 0;
    };

    std::queue<QueuedRequest> request_queue_;
    std::queue<QueuedRequest> offline_queue_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex offline_queue_mutex_;
    std::condition_variable queue_cv_;

    // Batch processing
    std::vector<RecognitionSubmission> recognition_batch_;
    std::vector<TrainingDataSubmission> training_batch_;
    std::mutex batch_mutex_;
    std::chrono::steady_clock::time_point last_batch_flush_;

    // Status tracking
    std::atomic<bool> online_status_{false};
    std::atomic<size_t> pending_requests_{0};
    ApiStats stats_;

    // Internal methods
    void worker_thread_loop();
    void batch_processor_loop();
    void offline_processor_loop();

    HttpResponse execute_http_request(const std::string& method,
                                    const std::string& endpoint,
                                    const Json::Value& payload = Json::Value(),
                                    const std::map<std::string, std::string>& headers = {});

    std::string build_url(const std::string& endpoint) const;
    std::map<std::string, std::string> get_default_headers() const;

    bool should_retry(const HttpResponse& response, int retry_count) const;
    void update_online_status(bool status);
    void update_response_time_stats(std::chrono::milliseconds duration);

    // JSON serialization helpers
    Json::Value serialize_recognition(const RecognitionSubmission& submission);
    Json::Value serialize_training_data(const TrainingDataSubmission& training_data);
    std::vector<ModelWeightUpdate> deserialize_model_updates(const Json::Value& json);
    AccuracyFeedback deserialize_accuracy_feedback(const Json::Value& json);

    // Offline queue management
    void save_offline_queue();
    void load_offline_queue();
    void add_to_offline_queue(const QueuedRequest& request);

    // Error handling
    void handle_request_error(const QueuedRequest& request, const HttpResponse& response);
    void log_api_error(const std::string& operation, const HttpResponse& response);
};

// High-level integration helper class
class LearningSystemIntegration {
public:
    explicit LearningSystemIntegration(std::shared_ptr<LearningApiClient> client);
    ~LearningSystemIntegration();

    // Integration with existing components
    void integrate_with_whisper_processor(WhisperProcessor* processor);
    void integrate_with_ensemble_processor(class EnsembleWhisperProcessor* processor);
    void integrate_with_adaptive_learning(class AdaptiveLearningSystem* learning_system);

    // Callback handlers for transcription results
    void handle_transcription_result(const TranscriptionResult& result);
    void handle_recognition_result(const vtt::RecognitionResult& result);
    void handle_training_correction(const std::string& recognized, const std::string& corrected);

    // Model weight management
    void request_model_updates();
    void apply_model_updates(const std::vector<ModelWeightUpdate>& updates);

    // Configuration
    void set_session_id(const std::string& session_id) { session_id_ = session_id; }
    void set_speaker_id(const std::string& speaker_id) { speaker_id_ = speaker_id; }
    void set_domain(const std::string& domain) { domain_ = domain; }

private:
    std::shared_ptr<LearningApiClient> api_client_;
    std::string session_id_;
    std::string speaker_id_;
    std::string domain_;

    // Feature extraction for API submissions
    std::vector<double> extract_audio_features(const float* samples, size_t num_samples);
    std::map<std::string, double> extract_metadata(const TranscriptionResult& result);

    // Integration state
    std::atomic<bool> auto_submit_enabled_{true};
    std::atomic<bool> model_updates_enabled_{true};
};

} // namespace vtt