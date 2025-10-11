#include "learning_api_client.h"
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace vtt {

// Forward declarations
class EnsembleWhisperProcessor;

// libcurl callback for writing response data
size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

LearningApiClient::LearningApiClient(const LearningApiConfig& config)
    : config_(config), curl_handle_(nullptr), curl_multi_(nullptr) {
}

LearningApiClient::~LearningApiClient() {
    cleanup();
}

bool LearningApiClient::initialize() {
    if (initialized_.load()) {
        return true;
    }

    // Initialize libcurl
    CURLcode global_init = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (global_init != CURLE_OK) {
        std::cerr << "Failed to initialize libcurl: " << curl_easy_strerror(global_init) << std::endl;
        return false;
    }

    curl_handle_ = curl_easy_init();
    if (!curl_handle_) {
        std::cerr << "Failed to create CURL handle" << std::endl;
        curl_global_cleanup();
        return false;
    }

    curl_multi_ = curl_multi_init();
    if (!curl_multi_) {
        std::cerr << "Failed to create CURL multi handle" << std::endl;
        curl_easy_cleanup(static_cast<CURL*>(curl_handle_));
        curl_global_cleanup();
        return false;
    }

    // Load offline queue if it exists
    if (config_.enable_offline_queue && !config_.offline_queue_file.empty()) {
        load_offline_queue();
    }

    // Start worker threads
    running_ = true;

    if (config_.enable_async_processing) {
        for (size_t i = 0; i < config_.max_concurrent_requests; ++i) {
            worker_threads_.emplace_back(&LearningApiClient::worker_thread_loop, this);
        }
    }

    if (config_.enable_batching) {
        batch_processor_thread_ = std::thread(&LearningApiClient::batch_processor_loop, this);
    }

    if (config_.enable_offline_queue) {
        offline_processor_thread_ = std::thread(&LearningApiClient::offline_processor_loop, this);
    }

    // Test connectivity
    HttpResponse test_response = execute_http_request("GET", "/api/" + config_.api_version + "/health");
    update_online_status(test_response.success);

    initialized_ = true;
    std::cout << "Learning API client initialized. Online: " << is_online() << std::endl;

    return true;
}

void LearningApiClient::cleanup() {
    if (!initialized_.load()) {
        return;
    }

    running_ = false;
    queue_cv_.notify_all();

    // Wait for worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    if (batch_processor_thread_.joinable()) {
        batch_processor_thread_.join();
    }

    if (offline_processor_thread_.joinable()) {
        offline_processor_thread_.join();
    }

    // Save offline queue
    if (config_.enable_offline_queue && !config_.offline_queue_file.empty()) {
        save_offline_queue();
    }

    // Cleanup libcurl
    if (curl_multi_) {
        curl_multi_cleanup(static_cast<CURLM*>(curl_multi_));
        curl_multi_ = nullptr;
    }

    if (curl_handle_) {
        curl_easy_cleanup(static_cast<CURL*>(curl_handle_));
        curl_handle_ = nullptr;
    }

    curl_global_cleanup();
    initialized_ = false;

    std::cout << "Learning API client cleaned up" << std::endl;
}

HttpResponse LearningApiClient::submit_recognition(const RecognitionSubmission& submission) {
    Json::Value payload = serialize_recognition(submission);
    return execute_http_request("POST", "/api/" + config_.api_version + "/recognition", payload);
}

HttpResponse LearningApiClient::get_model_weights(const std::string& model_name) {
    std::string endpoint = "/api/" + config_.api_version + "/models";
    if (!model_name.empty()) {
        endpoint += "?model=" + model_name;
    }
    return execute_http_request("GET", endpoint);
}

HttpResponse LearningApiClient::submit_training_data(const TrainingDataSubmission& training_data) {
    Json::Value payload = serialize_training_data(training_data);
    return execute_http_request("POST", "/api/" + config_.api_version + "/training", payload);
}

HttpResponse LearningApiClient::get_accuracy_feedback(const std::string& session_id) {
    std::string endpoint = "/api/" + config_.api_version + "/feedback";
    if (!session_id.empty()) {
        endpoint += "?session=" + session_id;
    }
    return execute_http_request("GET", endpoint);
}

std::future<HttpResponse> LearningApiClient::submit_recognition_async(const RecognitionSubmission& submission) {
    return std::async(std::launch::async, [this, submission]() {
        return submit_recognition(submission);
    });
}

std::future<HttpResponse> LearningApiClient::get_model_weights_async(const std::string& model_name) {
    return std::async(std::launch::async, [this, model_name]() {
        return get_model_weights(model_name);
    });
}

std::future<HttpResponse> LearningApiClient::submit_training_data_async(const TrainingDataSubmission& training_data) {
    return std::async(std::launch::async, [this, training_data]() {
        return submit_training_data(training_data);
    });
}

void LearningApiClient::submit_recognition_callback(const RecognitionSubmission& submission, RecognitionCallback callback) {
    if (!config_.enable_async_processing) {
        // Execute synchronously if async is disabled
        HttpResponse response = submit_recognition(submission);
        if (callback) {
            callback(response);
        }
        return;
    }

    QueuedRequest request;
    request.type = QueuedRequest::RECOGNITION;
    request.payload = serialize_recognition(submission);
    request.endpoint = "/api/" + config_.api_version + "/recognition";
    request.callback = callback;
    request.queued_at = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (request_queue_.size() >= config_.queue_max_size) {
        std::cerr << "Request queue full, dropping recognition submission" << std::endl;
        stats_.failed_requests++;
        return;
    }

    request_queue_.push(request);
    queue_cv_.notify_one();
}

void LearningApiClient::get_model_weights_callback(const std::string& model_name, ModelUpdateCallback callback) {
    if (!config_.enable_async_processing) {
        HttpResponse response = get_model_weights(model_name);
        if (callback) {
            std::vector<ModelWeightUpdate> updates = deserialize_model_updates(Json::Value());
            callback(updates, response);
        }
        return;
    }

    QueuedRequest request;
    request.type = QueuedRequest::MODEL_UPDATE;
    request.payload = Json::Value(); // No payload for GET request
    request.endpoint = "/api/" + config_.api_version + "/models" +
                      (model_name.empty() ? "" : "?model=" + model_name);
    request.callback = [this, callback](const HttpResponse& response) {
        if (callback) {
            std::vector<ModelWeightUpdate> updates;
            if (response.success) {
                try {
                    Json::Value json;
                    Json::Reader reader;
                    if (reader.parse(response.body, json)) {
                        updates = deserialize_model_updates(json);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing model updates: " << e.what() << std::endl;
                }
            }
            callback(updates, response);
        }
    };
    request.queued_at = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (request_queue_.size() >= config_.queue_max_size) {
        std::cerr << "Request queue full, dropping model update request" << std::endl;
        stats_.failed_requests++;
        return;
    }

    request_queue_.push(request);
    queue_cv_.notify_one();
}

void LearningApiClient::submit_training_data_callback(const TrainingDataSubmission& training_data, TrainingCallback callback) {
    if (!config_.enable_async_processing) {
        HttpResponse response = submit_training_data(training_data);
        if (callback) {
            callback(response);
        }
        return;
    }

    QueuedRequest request;
    request.type = QueuedRequest::TRAINING;
    request.payload = serialize_training_data(training_data);
    request.endpoint = "/api/" + config_.api_version + "/training";
    request.callback = callback;
    request.queued_at = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (request_queue_.size() >= config_.queue_max_size) {
        std::cerr << "Request queue full, dropping training submission" << std::endl;
        stats_.failed_requests++;
        return;
    }

    request_queue_.push(request);
    queue_cv_.notify_one();
}

void LearningApiClient::add_to_recognition_batch(const RecognitionSubmission& submission) {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    recognition_batch_.push_back(submission);

    if (recognition_batch_.size() >= config_.batch_size) {
        flush_recognition_batch();
    }
}

void LearningApiClient::flush_recognition_batch() {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    if (recognition_batch_.empty()) {
        return;
    }

    Json::Value batch_payload(Json::arrayValue);
    for (const auto& submission : recognition_batch_) {
        batch_payload.append(serialize_recognition(submission));
    }

    QueuedRequest request;
    request.type = QueuedRequest::RECOGNITION;
    request.payload = batch_payload;
    request.endpoint = "/api/" + config_.api_version + "/recognition/batch";
    request.queued_at = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        request_queue_.push(request);
        queue_cv_.notify_one();
    }

    stats_.batch_requests_sent++;
    recognition_batch_.clear();
    last_batch_flush_ = std::chrono::steady_clock::now();
}

void LearningApiClient::add_to_training_batch(const TrainingDataSubmission& training_data) {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    training_batch_.push_back(training_data);

    if (training_batch_.size() >= config_.batch_size) {
        flush_training_batch();
    }
}

void LearningApiClient::flush_training_batch() {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    if (training_batch_.empty()) {
        return;
    }

    Json::Value batch_payload(Json::arrayValue);
    for (const auto& training_data : training_batch_) {
        batch_payload.append(serialize_training_data(training_data));
    }

    QueuedRequest request;
    request.type = QueuedRequest::TRAINING;
    request.payload = batch_payload;
    request.endpoint = "/api/" + config_.api_version + "/training/batch";
    request.queued_at = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        request_queue_.push(request);
        queue_cv_.notify_one();
    }

    stats_.batch_requests_sent++;
    training_batch_.clear();
}

void LearningApiClient::worker_thread_loop() {
    while (running_) {
        QueuedRequest request;

        // Wait for work
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !request_queue_.empty() || !running_; });

            if (!running_) {
                break;
            }

            if (request_queue_.empty()) {
                continue;
            }

            request = request_queue_.front();
            request_queue_.pop();
        }

        pending_requests_++;

        // Execute the request
        std::string method = (request.payload.isNull() || request.payload.empty()) ? "GET" : "POST";
        HttpResponse response = execute_http_request(method, request.endpoint, request.payload);

        // Handle retry logic
        if (!response.success && should_retry(response, request.retry_count)) {
            request.retry_count++;

            // Add back to queue with delay
            std::this_thread::sleep_for(config_.retry_delay * request.retry_count);

            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (request_queue_.size() < config_.queue_max_size) {
                request_queue_.push(request);
                queue_cv_.notify_one();
            }

            pending_requests_--;
            continue;
        }

        // Update online status
        update_online_status(response.success);

        // Handle offline queueing
        if (!response.success && config_.enable_offline_queue) {
            add_to_offline_queue(request);
        }

        // Execute callback
        if (request.callback) {
            try {
                request.callback(response);
            } catch (const std::exception& e) {
                std::cerr << "Error in API callback: " << e.what() << std::endl;
            }
        }

        // Update statistics
        stats_.total_requests++;
        if (response.success) {
            stats_.successful_requests++;
            stats_.last_successful_request = std::chrono::steady_clock::now();
        } else {
            stats_.failed_requests++;
        }

        update_response_time_stats(response.duration);
        pending_requests_--;
    }
}

void LearningApiClient::batch_processor_loop() {
    while (running_) {
        std::this_thread::sleep_for(config_.batch_timeout);

        auto now = std::chrono::steady_clock::now();
        auto time_since_last_flush = now - last_batch_flush_;

        if (time_since_last_flush >= config_.batch_timeout) {
            flush_recognition_batch();
            flush_training_batch();
        }
    }
}

void LearningApiClient::offline_processor_loop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(30)); // Check every 30 seconds

        if (is_online()) {
            process_offline_queue();
        }
    }
}

HttpResponse LearningApiClient::execute_http_request(const std::string& method,
                                                   const std::string& endpoint,
                                                   const Json::Value& payload,
                                                   const std::map<std::string, std::string>& headers) {
    HttpResponse response;

    if (!curl_handle_) {
        response.error_message = "CURL not initialized";
        return response;
    }

    auto start_time = std::chrono::steady_clock::now();

    CURL* curl = static_cast<CURL*>(curl_handle_);
    std::string response_body;
    std::string url = build_url(endpoint);

    // Reset curl handle
    curl_easy_reset(curl);

    // Set basic options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(config_.request_timeout.count()));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, static_cast<long>(config_.connection_timeout.count()));

    // Set method and payload
    struct curl_slist* curl_headers = nullptr;
    std::string json_data;

    if (method == "POST" && !payload.isNull()) {
        Json::StreamWriterBuilder builder;
        json_data = Json::writeString(builder, payload);

        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_data.length());

        curl_headers = curl_slist_append(curl_headers, "Content-Type: application/json");
    }

    // Add default headers
    auto default_headers = get_default_headers();
    for (const auto& header : default_headers) {
        std::string header_str = header.first + ": " + header.second;
        curl_headers = curl_slist_append(curl_headers, header_str.c_str());
    }

    // Add custom headers
    for (const auto& header : headers) {
        std::string header_str = header.first + ": " + header.second;
        curl_headers = curl_slist_append(curl_headers, header_str.c_str());
    }

    if (curl_headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_headers);
    }

    // Execute the request
    CURLcode res = curl_easy_perform(curl);

    auto end_time = std::chrono::steady_clock::now();
    response.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (res == CURLE_OK) {
        long response_code;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        response.status_code = static_cast<int>(response_code);
        response.body = response_body;
        response.success = (response_code >= 200 && response_code < 300);

        if (!response.success) {
            response.error_message = "HTTP " + std::to_string(response_code);
        }
    } else {
        response.error_message = curl_easy_strerror(res);

        // Set appropriate status codes for different error types
        switch (res) {
            case CURLE_OPERATION_TIMEDOUT:
                response.status_code = 408; // Request Timeout
                stats_.timeout_requests++;
                break;
            case CURLE_COULDNT_CONNECT:
                response.status_code = 503; // Service Unavailable
                break;
            default:
                response.status_code = 0; // Unknown error
        }
    }

    // Cleanup
    if (curl_headers) {
        curl_slist_free_all(curl_headers);
    }

    return response;
}

std::string LearningApiClient::build_url(const std::string& endpoint) const {
    return config_.base_url + endpoint;
}

std::map<std::string, std::string> LearningApiClient::get_default_headers() const {
    std::map<std::string, std::string> headers;
    headers["User-Agent"] = "VoiceStand-LearningClient/1.0";
    headers["Accept"] = "application/json";

    if (!config_.api_key.empty()) {
        headers["Authorization"] = "Bearer " + config_.api_key;
    }

    return headers;
}

Json::Value LearningApiClient::serialize_recognition(const RecognitionSubmission& submission) {
    Json::Value json;
    json["text"] = submission.text;
    json["confidence"] = submission.confidence;
    json["model_used"] = submission.model_used;
    json["session_id"] = submission.session_id;
    json["is_final"] = submission.is_final;
    json["speaker_id"] = submission.speaker_id;
    json["domain"] = submission.domain;

    // Convert timestamp to ISO 8601 string
    auto time_t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now() +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            submission.timestamp - std::chrono::steady_clock::now()
        )
    );
    json["timestamp"] = std::to_string(time_t);

    // Add audio features
    Json::Value features(Json::arrayValue);
    for (double feature : submission.audio_features) {
        features.append(feature);
    }
    json["audio_features"] = features;

    // Add metadata
    Json::Value metadata;
    for (const auto& pair : submission.metadata) {
        metadata[pair.first] = pair.second;
    }
    json["metadata"] = metadata;

    return json;
}

Json::Value LearningApiClient::serialize_training_data(const TrainingDataSubmission& training_data) {
    Json::Value json;
    json["recognized_text"] = training_data.recognized_text;
    json["corrected_text"] = training_data.corrected_text;
    json["confidence"] = training_data.confidence;
    json["correction_type"] = training_data.correction_type;
    json["context"] = training_data.context;

    // Convert timestamp
    auto time_t = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now() +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            training_data.timestamp - std::chrono::steady_clock::now()
        )
    );
    json["timestamp"] = std::to_string(time_t);

    // Add audio features
    Json::Value features(Json::arrayValue);
    for (double feature : training_data.audio_features) {
        features.append(feature);
    }
    json["audio_features"] = features;

    return json;
}

std::vector<ModelWeightUpdate> LearningApiClient::deserialize_model_updates(const Json::Value& json) {
    std::vector<ModelWeightUpdate> updates;

    if (!json.isArray()) {
        return updates;
    }

    for (const auto& item : json) {
        if (!item.isObject()) {
            continue;
        }

        ModelWeightUpdate update;
        update.model_name = item.get("model_name", "").asString();
        update.accuracy_score = item.get("accuracy_score", 0.0).asDouble();
        update.version = item.get("version", "").asString();
        update.is_incremental = item.get("is_incremental", false).asBool();

        // Parse weights
        const Json::Value& weights = item["weights"];
        if (weights.isObject()) {
            for (const auto& member : weights.getMemberNames()) {
                update.weights[member] = weights[member].asDouble();
            }
        }

        update.updated_at = std::chrono::steady_clock::now();
        updates.push_back(update);
    }

    return updates;
}

AccuracyFeedback LearningApiClient::deserialize_accuracy_feedback(const Json::Value& json) {
    AccuracyFeedback feedback;

    if (!json.isObject()) {
        return feedback;
    }

    feedback.recognition_id = json.get("recognition_id", "").asString();
    feedback.accuracy_score = json.get("accuracy_score", 0.0).asDouble();
    feedback.feedback_type = json.get("feedback_type", "").asString();
    feedback.feedback_time = std::chrono::steady_clock::now();

    const Json::Value& suggestions = json["improvement_suggestions"];
    if (suggestions.isArray()) {
        for (const auto& suggestion : suggestions) {
            feedback.improvement_suggestions.push_back(suggestion.asString());
        }
    }

    return feedback;
}

bool LearningApiClient::should_retry(const HttpResponse& response, int retry_count) const {
    if (retry_count >= config_.max_retries) {
        return false;
    }

    // Retry on timeout or server errors
    return response.status_code == 408 ||  // Request Timeout
           response.status_code == 503 ||  // Service Unavailable
           response.status_code == 502 ||  // Bad Gateway
           response.status_code == 504 ||  // Gateway Timeout
           response.status_code == 0;      // Network error
}

void LearningApiClient::update_online_status(bool status) {
    bool was_online = online_status_.exchange(status);

    if (!was_online && status) {
        std::cout << "Learning API is now online" << std::endl;
    } else if (was_online && !status) {
        std::cout << "Learning API is now offline" << std::endl;
    }
}

void LearningApiClient::update_response_time_stats(std::chrono::milliseconds duration) {
    double current_avg = stats_.average_response_time.load();
    size_t total_requests = stats_.total_requests.load();

    if (total_requests == 0) {
        stats_.average_response_time = duration.count();
    } else {
        // Exponential moving average
        double new_avg = current_avg * 0.9 + duration.count() * 0.1;
        stats_.average_response_time = new_avg;
    }
}

void LearningApiClient::add_to_offline_queue(const QueuedRequest& request) {
    std::lock_guard<std::mutex> lock(offline_queue_mutex_);

    if (offline_queue_.size() >= config_.offline_queue_max_size) {
        // Remove oldest request
        offline_queue_.pop();
    }

    offline_queue_.push(request);
    stats_.offline_requests_queued++;
}

void LearningApiClient::process_offline_queue() {
    std::lock_guard<std::mutex> lock(offline_queue_mutex_);

    if (offline_queue_.empty()) {
        return;
    }

    std::cout << "Processing " << offline_queue_.size() << " offline requests" << std::endl;

    while (!offline_queue_.empty() && is_online()) {
        QueuedRequest request = offline_queue_.front();
        offline_queue_.pop();

        // Add back to main queue for processing
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        if (request_queue_.size() < config_.queue_max_size) {
            request_queue_.push(request);
            queue_cv_.notify_one();
        }
    }
}

size_t LearningApiClient::get_queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return request_queue_.size();
}

size_t LearningApiClient::get_offline_queue_size() const {
    std::lock_guard<std::mutex> lock(offline_queue_mutex_);
    return offline_queue_.size();
}

void LearningApiClient::reset_stats() {
    stats_.total_requests = 0;
    stats_.successful_requests = 0;
    stats_.failed_requests = 0;
    stats_.timeout_requests = 0;
    stats_.average_response_time = 0.0;
    stats_.offline_requests_queued = 0;
    stats_.batch_requests_sent = 0;
}

void LearningApiClient::save_offline_queue() {
    // Implementation for saving offline queue to disk
    if (config_.offline_queue_file.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(offline_queue_mutex_);

    try {
        std::ofstream file(config_.offline_queue_file);
        Json::Value queue_json(Json::arrayValue);

        std::queue<QueuedRequest> temp_queue = offline_queue_;
        while (!temp_queue.empty()) {
            const auto& request = temp_queue.front();
            Json::Value request_json;
            request_json["type"] = static_cast<int>(request.type);
            request_json["endpoint"] = request.endpoint;
            request_json["payload"] = request.payload;
            request_json["retry_count"] = request.retry_count;

            queue_json.append(request_json);
            temp_queue.pop();
        }

        Json::StreamWriterBuilder builder;
        file << Json::writeString(builder, queue_json);
        file.close();
    } catch (const std::exception& e) {
        std::cerr << "Error saving offline queue: " << e.what() << std::endl;
    }
}

void LearningApiClient::load_offline_queue() {
    if (config_.offline_queue_file.empty()) {
        return;
    }

    try {
        std::ifstream file(config_.offline_queue_file);
        if (!file.is_open()) {
            return; // File doesn't exist yet
        }

        Json::Value queue_json;
        Json::Reader reader;

        if (!reader.parse(file, queue_json) || !queue_json.isArray()) {
            return;
        }

        std::lock_guard<std::mutex> lock(offline_queue_mutex_);

        for (const auto& request_json : queue_json) {
            QueuedRequest request;
            request.type = static_cast<QueuedRequest::Type>(request_json["type"].asInt());
            request.endpoint = request_json["endpoint"].asString();
            request.payload = request_json["payload"];
            request.retry_count = request_json["retry_count"].asInt();
            request.queued_at = std::chrono::steady_clock::now();

            offline_queue_.push(request);
        }

        file.close();
        std::cout << "Loaded " << offline_queue_.size() << " requests from offline queue" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading offline queue: " << e.what() << std::endl;
    }
}

// LearningSystemIntegration implementation
LearningSystemIntegration::LearningSystemIntegration(std::shared_ptr<LearningApiClient> client)
    : api_client_(client), session_id_(""), speaker_id_(""), domain_("general") {
}

LearningSystemIntegration::~LearningSystemIntegration() {
}

void LearningSystemIntegration::integrate_with_whisper_processor(WhisperProcessor* processor) {
    if (!processor) {
        return;
    }

    // Set up callback to handle transcription results
    processor->set_transcription_callback([this](const TranscriptionResult& result) {
        this->handle_transcription_result(result);
    });
}

void LearningSystemIntegration::integrate_with_ensemble_processor(EnsembleWhisperProcessor* processor) {
    if (!processor) {
        return;
    }

    // The ensemble processor will call handle_recognition_result directly
    // when processing ensemble results
    std::cout << "Integrated with ensemble processor" << std::endl;
}

void LearningSystemIntegration::integrate_with_adaptive_learning(AdaptiveLearningSystem* learning_system) {
    if (!learning_system) {
        return;
    }

    // Integration with adaptive learning system would happen here
    // This could involve setting up callbacks for learning insights
    std::cout << "Integrated with adaptive learning system" << std::endl;
}

void LearningSystemIntegration::handle_transcription_result(const TranscriptionResult& result) {
    if (!auto_submit_enabled_.load()) {
        return;
    }

    // Extract features and create submission
    RecognitionSubmission submission;
    submission.text = result.text;
    submission.confidence = result.confidence;
    submission.session_id = session_id_;
    submission.speaker_id = speaker_id_;
    submission.domain = domain_;
    submission.is_final = result.is_final;
    submission.timestamp = result.timestamp;
    submission.model_used = "whisper"; // Default model name
    submission.metadata = extract_metadata(result);

    // Submit asynchronously
    api_client_->submit_recognition_callback(submission, [](const HttpResponse& response) {
        if (!response.success) {
            std::cerr << "Failed to submit recognition: " << response.error_message << std::endl;
        }
    });
}

void LearningSystemIntegration::handle_recognition_result(const vtt::RecognitionResult& result) {
    if (!auto_submit_enabled_.load()) {
        return;
    }

    // Convert ensemble recognition result to API submission format
    RecognitionSubmission submission;
    submission.text = result.text;
    submission.confidence = result.confidence;
    submission.session_id = session_id_;
    submission.speaker_id = speaker_id_;
    submission.domain = domain_;
    submission.is_final = true; // Ensemble results are typically final
    submission.timestamp = std::chrono::steady_clock::now();
    submission.model_used = result.model_used;

    // Add ensemble-specific metadata
    submission.metadata["is_ensemble"] = result.is_ensemble_result ? 1.0 : 0.0;
    submission.metadata["ensemble_agreement"] = result.ensemble_agreement;
    submission.metadata["processing_time_ms"] = static_cast<double>(result.processing_time.count());

    // Add individual model confidences
    for (const auto& model_conf : result.model_confidences) {
        submission.metadata["model_confidence_" + model_conf.first] = model_conf.second;
    }

    // Submit asynchronously
    api_client_->submit_recognition_callback(submission, [](const HttpResponse& response) {
        if (!response.success) {
            std::cerr << "Failed to submit ensemble recognition: " << response.error_message << std::endl;
        } else {
            std::cout << "Ensemble recognition submitted successfully" << std::endl;
        }
    });
}

std::vector<double> LearningSystemIntegration::extract_audio_features(const float* samples, size_t num_samples) {
    std::vector<double> features;

    if (!samples || num_samples == 0) {
        return features;
    }

    // Basic feature extraction - you can enhance this
    // Energy
    double energy = 0.0;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += samples[i] * samples[i];
    }
    energy /= num_samples;
    features.push_back(energy);

    // Zero crossing rate
    int zero_crossings = 0;
    for (size_t i = 1; i < num_samples; ++i) {
        if ((samples[i-1] >= 0.0f) != (samples[i] >= 0.0f)) {
            zero_crossings++;
        }
    }
    double zcr = static_cast<double>(zero_crossings) / num_samples;
    features.push_back(zcr);

    // Spectral centroid (simplified)
    double spectral_centroid = 0.0;
    double magnitude_sum = 0.0;
    for (size_t i = 0; i < num_samples; ++i) {
        double magnitude = std::abs(samples[i]);
        spectral_centroid += i * magnitude;
        magnitude_sum += magnitude;
    }
    if (magnitude_sum > 0) {
        spectral_centroid /= magnitude_sum;
    }
    features.push_back(spectral_centroid);

    return features;
}

std::map<std::string, double> LearningSystemIntegration::extract_metadata(const TranscriptionResult& result) {
    std::map<std::string, double> metadata;

    metadata["text_length"] = static_cast<double>(result.text.length());
    metadata["confidence"] = result.confidence;
    metadata["is_final"] = result.is_final ? 1.0 : 0.0;

    // Add timestamp as seconds since epoch
    auto duration = result.timestamp.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    metadata["timestamp"] = static_cast<double>(seconds.count());

    return metadata;
}

void LearningSystemIntegration::handle_training_correction(const std::string& recognized, const std::string& corrected) {
    TrainingDataSubmission training_data;
    training_data.recognized_text = recognized;
    training_data.corrected_text = corrected;
    training_data.confidence = 1.0; // High confidence for manual corrections
    training_data.correction_type = "manual_correction";
    training_data.context = domain_;
    training_data.timestamp = std::chrono::steady_clock::now();

    api_client_->submit_training_data_callback(training_data, [](const HttpResponse& response) {
        if (!response.success) {
            std::cerr << "Failed to submit training data: " << response.error_message << std::endl;
        }
    });
}

void LearningSystemIntegration::request_model_updates() {
    if (!model_updates_enabled_.load()) {
        return;
    }

    api_client_->get_model_weights_callback("", [this](const std::vector<ModelWeightUpdate>& updates, const HttpResponse& response) {
        if (response.success) {
            this->apply_model_updates(updates);
        } else {
            std::cerr << "Failed to get model updates: " << response.error_message << std::endl;
        }
    });
}

void LearningSystemIntegration::apply_model_updates(const std::vector<ModelWeightUpdate>& updates) {
    std::cout << "Applying " << updates.size() << " model updates" << std::endl;

    // This would integrate with your actual model management system
    for (const auto& update : updates) {
        std::cout << "Model: " << update.model_name
                  << ", Accuracy: " << update.accuracy_score
                  << ", Weights: " << update.weights.size() << std::endl;
    }
}

} // namespace vtt