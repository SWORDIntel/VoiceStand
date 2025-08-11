#include "whisper_processor.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <map>

namespace vtt {

WhisperProcessor::WhisperProcessor()
    : ctx_(nullptr)
    , is_initialized_(false)
    , is_processing_(false)
    , language_("auto")
    , num_threads_(4)
    , transcription_callback_(nullptr) {
}

WhisperProcessor::~WhisperProcessor() {
    cleanup();
}

bool WhisperProcessor::initialize(const WhisperConfig& config) {
    std::lock_guard<std::mutex> lock(init_mutex_);
    
    if (is_initialized_) {
        return true;
    }
    
    model_path_ = config.model_path;
    language_ = config.language;
    num_threads_ = config.num_threads;
    
    // Check if model file exists
    if (!std::ifstream(model_path_).good()) {
        std::cerr << "[ERROR] Model file not found: " << model_path_ << "\n";
        std::cerr << "Please download a model using: ./build.sh download-model <size>\n";
        std::cerr << "Available sizes: tiny, base, small, medium, large\n";
        return false;
    }
    
    // Check if file is readable and has content
    std::ifstream file(model_path_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open model file: " << model_path_ << "\n";
        std::cerr << "Check file permissions\n";
        return false;
    }
    
    size_t file_size = file.tellg();
    file.close();
    
    if (file_size < 1024 * 1024) { // Less than 1MB is suspicious
        std::cerr << "[ERROR] Model file appears to be corrupted or incomplete: " 
                  << model_path_ << " (" << file_size << " bytes)\n";
        std::cerr << "Please re-download the model\n";
        return false;
    }
    
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = config.use_gpu;
    
    std::cout << "[INFO] Loading Whisper model from: " << model_path_ << "...\n";
    ctx_ = whisper_init_from_file_with_params(model_path_.c_str(), cparams);
    if (!ctx_) {
        std::cerr << "[ERROR] Failed to initialize Whisper context\n";
        std::cerr << "Possible causes:\n";
        std::cerr << "  - Corrupted model file\n";
        std::cerr << "  - Incompatible model version\n";
        std::cerr << "  - Insufficient memory\n";
        std::cerr << "Try re-downloading the model or using a smaller model size\n";
        return false;
    }
    
    std::cout << "[INFO] Whisper model loaded successfully\n";
    is_initialized_ = true;
    
    processing_thread_ = std::thread(&WhisperProcessor::processing_loop, this);
    
    return true;
}

void WhisperProcessor::process_audio(const float* samples, size_t num_samples, 
                                    uint32_t sample_rate) {
    if (!is_initialized_) {
        std::cerr << "[WARNING] WhisperProcessor not initialized, dropping audio\n";
        return;
    }
    
    if (!samples || num_samples == 0) {
        return;
    }
    
    std::vector<float> resampled;
    if (sample_rate != WHISPER_SAMPLE_RATE) {
        resampled = resample_audio(samples, num_samples, sample_rate, WHISPER_SAMPLE_RATE);
        samples = resampled.data();
        num_samples = resampled.size();
    }
    
    AudioChunk chunk;
    chunk.samples.assign(samples, samples + num_samples);
    chunk.timestamp = std::chrono::steady_clock::now();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        audio_queue_.push(std::move(chunk));
    }
    queue_cv_.notify_one();
}

void WhisperProcessor::start_streaming() {
    is_processing_ = true;
}

void WhisperProcessor::stop_streaming() {
    is_processing_ = false;
    queue_cv_.notify_all();
}

void WhisperProcessor::set_transcription_callback(TranscriptionCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    transcription_callback_ = callback;
}

void WhisperProcessor::cleanup() {
    stop_streaming();
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
    
    is_initialized_ = false;
}

void WhisperProcessor::processing_loop() {
    std::vector<float> accumulated_samples;
    const size_t min_samples = WHISPER_SAMPLE_RATE * 1;
    const size_t max_samples = WHISPER_SAMPLE_RATE * 30;
    
    while (is_processing_ || !audio_queue_.empty()) {
        AudioChunk chunk;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { 
                return !audio_queue_.empty() || !is_processing_; 
            });
            
            if (audio_queue_.empty()) {
                continue;
            }
            
            chunk = std::move(audio_queue_.front());
            audio_queue_.pop();
        }
        
        accumulated_samples.insert(accumulated_samples.end(), 
                                 chunk.samples.begin(), 
                                 chunk.samples.end());
        
        if (accumulated_samples.size() >= min_samples) {
            size_t samples_to_process = std::min(accumulated_samples.size(), max_samples);
            
            std::string transcription = transcribe_audio(
                accumulated_samples.data(), 
                samples_to_process
            );
            
            if (!transcription.empty() && transcription_callback_) {
                TranscriptionResult result;
                result.text = transcription;
                result.timestamp = chunk.timestamp;
                result.is_final = false;
                
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (transcription_callback_) {
                    transcription_callback_(result);
                }
            }
            
            if (accumulated_samples.size() > max_samples) {
                size_t overlap = WHISPER_SAMPLE_RATE * 2;
                accumulated_samples.erase(
                    accumulated_samples.begin(),
                    accumulated_samples.begin() + (accumulated_samples.size() - overlap)
                );
            }
        }
    }
}

std::string WhisperProcessor::transcribe_audio(const float* samples, size_t num_samples) {
    if (!ctx_) {
        std::cerr << "[ERROR] Whisper context is null - cannot transcribe\n";
        return "";
    }
    
    if (!samples || num_samples == 0) {
        return "";
    }
    
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.translate = false;
    wparams.single_segment = false;
    wparams.max_tokens = 0;
    wparams.language = language_ == "auto" ? nullptr : language_.c_str();
    wparams.n_threads = num_threads_;
    wparams.audio_ctx = 0;
    // speed_up option removed in newer whisper.cpp
    
    wparams.tdrz_enable = false;
    
    wparams.temperature = 0.0f;
    wparams.temperature_inc = 0.2f;
    wparams.greedy.best_of = 5;
    
    wparams.beam_search.beam_size = -1;
    
    wparams.suppress_blank = true;
    // suppress_non_speech_tokens option removed in newer whisper.cpp
    
    if (whisper_full(ctx_, wparams, samples, num_samples) != 0) {
        std::cerr << "Failed to process audio with Whisper\n";
        return "";
    }
    
    std::string result;
    const int n_segments = whisper_full_n_segments(ctx_);
    
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx_, i);
        if (text) {
            result += text;
            if (i < n_segments - 1) {
                result += " ";
            }
        }
    }
    
    return result;
}

std::vector<float> WhisperProcessor::resample_audio(const float* input, size_t input_size,
                                                   uint32_t input_rate, uint32_t output_rate) {
    if (input_rate == output_rate) {
        return std::vector<float>(input, input + input_size);
    }
    
    double ratio = static_cast<double>(output_rate) / input_rate;
    size_t output_size = static_cast<size_t>(input_size * ratio);
    std::vector<float> output(output_size);
    
    for (size_t i = 0; i < output_size; ++i) {
        double src_idx = i / ratio;
        size_t idx = static_cast<size_t>(src_idx);
        double frac = src_idx - idx;
        
        if (idx < input_size - 1) {
            output[i] = input[idx] * (1.0 - frac) + input[idx + 1] * frac;
        } else if (idx < input_size) {
            output[i] = input[idx];
        } else {
            output[i] = 0.0f;
        }
    }
    
    return output;
}

bool WhisperProcessor::download_model(const std::string& model_size, 
                                     const std::string& dest_path) {
    const std::map<std::string, std::string> model_urls = {
        {"tiny", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"},
        {"base", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"},
        {"small", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"},
        {"medium", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin"},
        {"large", "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"}
    };
    
    auto it = model_urls.find(model_size);
    if (it == model_urls.end()) {
        std::cerr << "Unknown model size: " << model_size << "\n";
        return false;
    }
    
    std::cout << "Downloading " << model_size << " model...\n";
    std::string cmd = "wget -O \"" + dest_path + "\" \"" + it->second + "\"";
    
    int result = std::system(cmd.c_str());
    if (result != 0) {
        std::cerr << "Failed to download model\n";
        return false;
    }
    
    std::cout << "Model downloaded successfully to " << dest_path << "\n";
    return true;
}

}