#pragma once

#include <pulse/pulseaudio.h>
#include <atomic>
#include <thread>
#include <functional>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <string>

namespace vtt {

struct AudioConfig {
    uint32_t sample_rate = 16000;
    float vad_threshold = 0.3f;
    std::string device_name = "default";
};

struct AudioData {
    const float* samples;
    size_t num_samples;
    uint32_t sample_rate;
    bool is_speech_end;
};

using AudioCallback = std::function<void(const AudioData&)>;
using ErrorCallback = std::function<void(const std::string&)>;

template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity = 16384) 
        : buffer_(capacity), write_pos_(0), read_pos_(0), size_(0) {}
    
    bool write(const T* data, size_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size_ + count > buffer_.size()) {
            return false;
        }
        
        for (size_t i = 0; i < count; ++i) {
            buffer_[write_pos_] = data[i];
            write_pos_ = (write_pos_ + 1) % buffer_.size();
        }
        
        size_ += count;
        cv_.notify_one();
        return true;
    }
    
    size_t read(T* data, size_t count, int timeout_ms = 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (timeout_ms > 0) {
            cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [this, count] { return size_ >= count; });
        }
        
        size_t to_read = std::min(count, size_.load());
        for (size_t i = 0; i < to_read; ++i) {
            data[i] = buffer_[read_pos_];
            read_pos_ = (read_pos_ + 1) % buffer_.size();
        }
        
        size_ -= to_read;
        return to_read;
    }
    
    size_t available() const { return size_; }
    void clear() { 
        std::lock_guard<std::mutex> lock(mutex_);
        write_pos_ = read_pos_ = size_ = 0; 
    }
    
private:
    std::vector<T> buffer_;
    size_t write_pos_;
    size_t read_pos_;
    std::atomic<size_t> size_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

struct VADState {
    bool is_speaking = false;
    int consecutive_speech_frames = 0;
    int consecutive_silence_frames = 0;
    uint64_t speech_start_frame = 0;
    uint64_t speech_end_frame = 0;
    uint64_t frame_count = 0;
};

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();
    
    bool initialize(const AudioConfig& config = {});
    bool start();
    void stop();
    void set_audio_callback(AudioCallback callback);
    void set_error_callback(ErrorCallback callback);
    
    bool is_recording() const { return is_recording_; }
    uint32_t sample_rate() const { return sample_rate_; }
    
private:
    void cleanup();
    static void context_state_callback(pa_context* c, void* userdata);
    static void stream_state_callback(pa_stream* s, void* userdata);
    static void stream_read_callback(pa_stream* s, size_t length, void* userdata);
    
    void processing_loop();
    float calculate_energy(const float* samples, size_t num_samples);
    float apply_noise_gate(float sample, float threshold);
    void apply_noise_reduction(float* samples, size_t num_samples);
    
    pa_mainloop* pa_ml_;
    pa_context* pa_ctx_;
    pa_stream* pa_stream_;
    
    std::atomic<bool> is_recording_;
    uint32_t sample_rate_;
    float vad_threshold_;
    
    RingBuffer<float> ring_buffer_;
    VADState vad_state_;
    
    AudioCallback audio_callback_;
    ErrorCallback error_callback_;
    std::mutex callback_mutex_;
    std::thread processing_thread_;
};

}