#pragma once

#include "whisper.h"
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <chrono>
#include <atomic>

namespace vtt {

// WHISPER_SAMPLE_RATE is already defined in whisper.h

struct WhisperConfig {
    std::string model_path = "models/ggml-base.bin";
    std::string language = "auto";
    int num_threads = 4;
    bool use_gpu = false;
};

struct TranscriptionResult {
    std::string text;
    std::chrono::steady_clock::time_point timestamp;
    bool is_final;
    float confidence = 0.0f;
};

using TranscriptionCallback = std::function<void(const TranscriptionResult&)>;

struct AudioChunk {
    std::vector<float> samples;
    std::chrono::steady_clock::time_point timestamp;
};

class WhisperProcessor {
public:
    WhisperProcessor();
    ~WhisperProcessor();
    
    bool initialize(const WhisperConfig& config = {});
    void process_audio(const float* samples, size_t num_samples, uint32_t sample_rate);
    void start_streaming();
    void stop_streaming();
    void set_transcription_callback(TranscriptionCallback callback);
    
    bool is_initialized() const { return is_initialized_; }
    
    static bool download_model(const std::string& model_size, const std::string& dest_path);
    
private:
    void cleanup();
    void processing_loop();
    std::string transcribe_audio(const float* samples, size_t num_samples);
    std::vector<float> resample_audio(const float* input, size_t input_size,
                                     uint32_t input_rate, uint32_t output_rate);
    
    whisper_context* ctx_;
    std::atomic<bool> is_initialized_;
    std::atomic<bool> is_processing_;
    
    std::string model_path_;
    std::string language_;
    int num_threads_;
    
    std::queue<AudioChunk> audio_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    TranscriptionCallback transcription_callback_;
    std::mutex callback_mutex_;
    
    std::thread processing_thread_;
    std::mutex init_mutex_;
};

}