#include "audio_capture.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>

namespace vtt {

AudioCapture::AudioCapture()
    : pa_ml_(nullptr)
    , pa_ctx_(nullptr)
    , pa_stream_(nullptr)
    , is_recording_(false)
    , sample_rate_(16000)
    , vad_threshold_(0.3f)
    , audio_callback_(nullptr) {
}

AudioCapture::~AudioCapture() {
    cleanup();
}

bool AudioCapture::initialize(const AudioConfig& config) {
    sample_rate_ = config.sample_rate;
    vad_threshold_ = config.vad_threshold;
    
    pa_ml_ = pa_mainloop_new();
    if (!pa_ml_) {
        std::cerr << "Failed to create PulseAudio mainloop\n";
        return false;
    }
    
    pa_mainloop_api* mlapi = pa_mainloop_get_api(pa_ml_);
    pa_ctx_ = pa_context_new(mlapi, "VoiceToText");
    if (!pa_ctx_) {
        std::cerr << "Failed to create PulseAudio context\n";
        cleanup();
        return false;
    }
    
    pa_context_set_state_callback(pa_ctx_, context_state_callback, this);
    
    if (pa_context_connect(pa_ctx_, nullptr, PA_CONTEXT_NOFLAGS, nullptr) < 0) {
        std::cerr << "[ERROR] Failed to connect to PulseAudio server\n";
        std::cerr << "Please ensure PulseAudio is running:\n";
        std::cerr << "  systemctl --user status pulseaudio\n";
        std::cerr << "  systemctl --user start pulseaudio\n";
        cleanup();
        return false;
    }
    
    int pa_ready = 0;
    while (pa_ready == 0) {
        pa_mainloop_iterate(pa_ml_, 1, nullptr);
        
        pa_context_state_t state = pa_context_get_state(pa_ctx_);
        if (state == PA_CONTEXT_READY) {
            pa_ready = 1;
        } else if (!PA_CONTEXT_IS_GOOD(state)) {
            std::cerr << "[ERROR] PulseAudio context failed with state: " << state << "\n";
            if (state == PA_CONTEXT_FAILED) {
                std::cerr << "Connection to PulseAudio server failed\n";
            } else if (state == PA_CONTEXT_TERMINATED) {
                std::cerr << "PulseAudio connection terminated\n";
            }
            cleanup();
            return false;
        }
    }
    
    return true;
}

bool AudioCapture::start() {
    if (is_recording_) {
        return true;
    }
    
    pa_sample_spec ss;
    ss.format = PA_SAMPLE_FLOAT32LE;
    ss.channels = 1;
    ss.rate = sample_rate_;
    
    pa_stream_ = pa_stream_new(pa_ctx_, "VoiceToText Stream", &ss, nullptr);
    if (!pa_stream_) {
        std::cerr << "[ERROR] Failed to create PulseAudio stream\n";
        std::cerr << "Error: " << pa_strerror(pa_context_errno(pa_ctx_)) << "\n";
        return false;
    }
    
    pa_stream_set_read_callback(pa_stream_, stream_read_callback, this);
    pa_stream_set_state_callback(pa_stream_, stream_state_callback, this);
    
    pa_buffer_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.fragsize = pa_usec_to_bytes(20 * PA_USEC_PER_MSEC, &ss);
    attr.maxlength = (uint32_t) -1;
    
    if (pa_stream_connect_record(pa_stream_, nullptr, &attr, 
                                  PA_STREAM_ADJUST_LATENCY) < 0) {
        std::cerr << "[ERROR] Failed to connect recording stream\n";
        std::cerr << "Error: " << pa_strerror(pa_context_errno(pa_ctx_)) << "\n";
        std::cerr << "Please check:\n";
        std::cerr << "  - Microphone permissions\n";
        std::cerr << "  - Default audio input device is configured\n";
        std::cerr << "  - Run: pactl info\n";
        pa_stream_unref(pa_stream_);
        pa_stream_ = nullptr;
        return false;
    }
    
    is_recording_ = true;
    
    processing_thread_ = std::thread(&AudioCapture::processing_loop, this);
    
    return true;
}

void AudioCapture::stop() {
    if (!is_recording_) {
        return;
    }
    
    is_recording_ = false;
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    if (pa_stream_) {
        pa_stream_disconnect(pa_stream_);
        pa_stream_unref(pa_stream_);
        pa_stream_ = nullptr;
    }
}

void AudioCapture::set_audio_callback(AudioCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    audio_callback_ = callback;
}

void AudioCapture::set_error_callback(ErrorCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    error_callback_ = callback;
}

void AudioCapture::cleanup() {
    stop();
    
    if (pa_ctx_) {
        pa_context_disconnect(pa_ctx_);
        pa_context_unref(pa_ctx_);
        pa_ctx_ = nullptr;
    }
    
    if (pa_ml_) {
        pa_mainloop_free(pa_ml_);
        pa_ml_ = nullptr;
    }
}

void AudioCapture::context_state_callback(pa_context* c, void* userdata) {
}

void AudioCapture::stream_state_callback(pa_stream* s, void* userdata) {
    auto* capture = static_cast<AudioCapture*>(userdata);
    
    pa_stream_state_t state = pa_stream_get_state(s);
    
    switch (state) {
        case PA_STREAM_READY:
            std::cout << "[INFO] Audio stream ready\n";
            break;
            
        case PA_STREAM_FAILED:
            std::cerr << "[ERROR] Audio stream failed\n";
            std::cerr << "Error: " << pa_strerror(pa_context_errno(capture->pa_ctx_)) << "\n";
            
            // Attempt to recover
            if (capture->is_recording_) {
                std::cerr << "[INFO] Attempting to recover audio stream...\n";
                capture->stop();
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (!capture->start()) {
                    std::cerr << "[ERROR] Failed to recover audio stream\n";
                    // Notify the application of the failure
                    if (capture->error_callback_) {
                        capture->error_callback_("Audio stream failed and could not be recovered");
                    }
                }
            }
            break;
            
        case PA_STREAM_TERMINATED:
            std::cerr << "[WARNING] Audio stream terminated\n";
            break;
            
        default:
            break;
    }
}

void AudioCapture::stream_read_callback(pa_stream* s, size_t length, void* userdata) {
    auto* capture = static_cast<AudioCapture*>(userdata);
    
    const void* data;
    if (pa_stream_peek(s, &data, &length) < 0) {
        std::cerr << "Failed to read from stream\n";
        return;
    }
    
    if (data && length > 0) {
        const float* samples = static_cast<const float*>(data);
        size_t num_samples = length / sizeof(float);
        
        capture->ring_buffer_.write(samples, num_samples);
        
        float energy = capture->calculate_energy(samples, num_samples);
        if (energy > capture->vad_threshold_) {
            capture->vad_state_.consecutive_speech_frames++;
            capture->vad_state_.consecutive_silence_frames = 0;
            
            if (capture->vad_state_.consecutive_speech_frames >= 10 && 
                !capture->vad_state_.is_speaking) {
                capture->vad_state_.is_speaking = true;
                capture->vad_state_.speech_start_frame = capture->vad_state_.frame_count;
            }
        } else {
            capture->vad_state_.consecutive_silence_frames++;
            capture->vad_state_.consecutive_speech_frames = 0;
            
            if (capture->vad_state_.consecutive_silence_frames >= 30 && 
                capture->vad_state_.is_speaking) {
                capture->vad_state_.is_speaking = false;
                capture->vad_state_.speech_end_frame = capture->vad_state_.frame_count;
                
                if (capture->audio_callback_) {
                    std::lock_guard<std::mutex> lock(capture->callback_mutex_);
                    if (capture->audio_callback_) {
                        AudioData audio_data;
                        audio_data.samples = samples;
                        audio_data.num_samples = num_samples;
                        audio_data.sample_rate = capture->sample_rate_;
                        audio_data.is_speech_end = true;
                        capture->audio_callback_(audio_data);
                    }
                }
            }
        }
        
        capture->vad_state_.frame_count++;
    }
    
    if (data) {
        pa_stream_drop(s);
    }
}

void AudioCapture::processing_loop() {
    while (is_recording_) {
        if (pa_ml_) {
            pa_mainloop_iterate(pa_ml_, 0, nullptr);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

float AudioCapture::calculate_energy(const float* samples, size_t num_samples) {
    if (!samples || num_samples == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        sum += samples[i] * samples[i];
    }
    
    return std::sqrt(sum / num_samples);
}

float AudioCapture::apply_noise_gate(float sample, float threshold) {
    return (std::abs(sample) < threshold) ? 0.0f : sample;
}

void AudioCapture::apply_noise_reduction(float* samples, size_t num_samples) {
    const float noise_gate_threshold = 0.01f;
    
    for (size_t i = 0; i < num_samples; ++i) {
        samples[i] = apply_noise_gate(samples[i], noise_gate_threshold);
    }
    
    const int filter_size = 5;
    if (num_samples > filter_size) {
        std::vector<float> filtered(num_samples);
        
        for (size_t i = filter_size/2; i < num_samples - filter_size/2; ++i) {
            float sum = 0.0f;
            for (int j = -filter_size/2; j <= filter_size/2; ++j) {
                sum += samples[i + j];
            }
            filtered[i] = sum / filter_size;
        }
        
        for (size_t i = filter_size/2; i < num_samples - filter_size/2; ++i) {
            samples[i] = filtered[i];
        }
    }
}

}