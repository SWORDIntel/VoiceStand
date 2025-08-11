#pragma once

#include <vector>
#include <deque>
#include <cmath>
#include <complex>
#include <algorithm>
#include <memory>

namespace vtt {

// Advanced noise cancellation using spectral subtraction and adaptive filtering
class NoiseCancellation {
public:
    struct Config {
        size_t frame_size = 480;        // 30ms at 16kHz
        size_t fft_size = 512;
        float noise_gate_threshold = 0.01f;
        float spectral_floor = 0.1f;    // Minimum spectral magnitude
        float over_subtraction = 1.5f;   // Over-subtraction factor
        size_t noise_frames = 20;        // Frames for noise estimation
        bool use_wiener_filter = true;
        bool use_echo_cancellation = true;
    };
    
    NoiseCancellation(const Config& config = Config(), uint32_t sample_rate = 16000);
    ~NoiseCancellation();
    
    // Process audio frame
    std::vector<float> process(const float* input, size_t num_samples);
    
    // Update noise profile
    void update_noise_profile(const float* samples, size_t num_samples);
    
    // Reset to initial state
    void reset();
    
    // Enable/disable processing
    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }
    
    // Get processing statistics
    struct Stats {
        float average_noise_reduction_db = 0.0f;
        float peak_noise_reduction_db = 0.0f;
        size_t frames_processed = 0;
        float computation_time_ms = 0.0f;
    };
    
    const Stats& get_stats() const { return stats_; }
    
private:
    // FFT operations
    void fft(const std::vector<float>& input, std::vector<std::complex<float>>& output);
    void ifft(const std::vector<std::complex<float>>& input, std::vector<float>& output);
    
    // Window functions
    std::vector<float> apply_window(const std::vector<float>& frame);
    std::vector<float> hann_window(size_t size);
    
    // Spectral subtraction
    std::vector<std::complex<float>> spectral_subtraction(
        const std::vector<std::complex<float>>& spectrum);
    
    // Wiener filtering
    std::vector<std::complex<float>> wiener_filter(
        const std::vector<std::complex<float>>& spectrum);
    
    // Echo cancellation (simplified)
    std::vector<float> echo_cancellation(const std::vector<float>& input);
    
    // Noise estimation
    void estimate_noise(const std::vector<std::complex<float>>& spectrum);
    
    // Compute magnitude spectrum
    std::vector<float> magnitude_spectrum(const std::vector<std::complex<float>>& spectrum);
    
    // Smoothing filters
    float exponential_average(float current, float previous, float alpha);
    
    Config config_;
    uint32_t sample_rate_;
    bool enabled_ = true;
    
    // Buffers
    std::deque<float> input_buffer_;
    std::vector<float> window_;
    std::vector<float> noise_profile_;
    std::vector<float> prev_magnitude_;
    
    // Echo cancellation
    std::deque<float> reference_buffer_;
    std::vector<float> adaptive_filter_;
    
    // State
    size_t frames_for_noise_ = 0;
    bool noise_estimated_ = false;
    
    Stats stats_;
};

// Simplified implementation
inline NoiseCancellation::NoiseCancellation(const Config& config, uint32_t sample_rate)
    : config_(config)
    , sample_rate_(sample_rate) {
    
    // Initialize window function
    window_ = hann_window(config.frame_size);
    
    // Initialize noise profile
    noise_profile_.resize(config.fft_size / 2 + 1, 0.0f);
    prev_magnitude_.resize(config.fft_size / 2 + 1, 0.0f);
    
    // Initialize adaptive filter for echo cancellation
    adaptive_filter_.resize(config.frame_size, 0.0f);
}

inline NoiseCancellation::~NoiseCancellation() = default;

inline std::vector<float> NoiseCancellation::process(const float* input, size_t num_samples) {
    if (!enabled_ || !input || num_samples == 0) {
        return std::vector<float>(input, input + num_samples);
    }
    
    std::vector<float> output;
    output.reserve(num_samples);
    
    // Add input to buffer
    for (size_t i = 0; i < num_samples; ++i) {
        input_buffer_.push_back(input[i]);
    }
    
    // Process in frames
    while (input_buffer_.size() >= config_.frame_size) {
        // Extract frame
        std::vector<float> frame(input_buffer_.begin(), 
                                input_buffer_.begin() + config_.frame_size);
        
        // Remove processed samples from buffer
        for (size_t i = 0; i < config_.frame_size / 2; ++i) {
            input_buffer_.pop_front();
        }
        
        // Apply window
        auto windowed = apply_window(frame);
        
        // FFT
        std::vector<std::complex<float>> spectrum(config_.fft_size);
        fft(windowed, spectrum);
        
        // Noise estimation (first few frames)
        if (!noise_estimated_ && frames_for_noise_ < config_.noise_frames) {
            estimate_noise(spectrum);
            frames_for_noise_++;
            if (frames_for_noise_ >= config_.noise_frames) {
                noise_estimated_ = true;
            }
        }
        
        // Apply noise reduction
        std::vector<std::complex<float>> clean_spectrum = spectrum;
        
        if (noise_estimated_) {
            // Spectral subtraction
            clean_spectrum = spectral_subtraction(spectrum);
            
            // Wiener filter
            if (config_.use_wiener_filter) {
                clean_spectrum = wiener_filter(clean_spectrum);
            }
        }
        
        // IFFT
        std::vector<float> clean_frame(config_.fft_size);
        ifft(clean_spectrum, clean_frame);
        
        // Echo cancellation
        if (config_.use_echo_cancellation) {
            clean_frame = echo_cancellation(clean_frame);
        }
        
        // Overlap-add
        output.insert(output.end(), 
                     clean_frame.begin(), 
                     clean_frame.begin() + config_.frame_size / 2);
        
        stats_.frames_processed++;
    }
    
    return output;
}

inline void NoiseCancellation::update_noise_profile(const float* samples, size_t num_samples) {
    if (!samples || num_samples == 0) return;
    
    // Process samples to estimate noise
    std::vector<float> frame(samples, samples + std::min(num_samples, config_.frame_size));
    
    // Pad if necessary
    if (frame.size() < config_.fft_size) {
        frame.resize(config_.fft_size, 0.0f);
    }
    
    // FFT
    std::vector<std::complex<float>> spectrum(config_.fft_size);
    fft(frame, spectrum);
    
    // Update noise estimate
    estimate_noise(spectrum);
    noise_estimated_ = true;
}

inline void NoiseCancellation::fft(const std::vector<float>& input, 
                                  std::vector<std::complex<float>>& output) {
    // Simplified DFT (in production, use FFTW or similar)
    size_t N = std::min(input.size(), output.size());
    
    for (size_t k = 0; k < N; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (size_t n = 0; n < N; ++n) {
            float angle = -2.0f * M_PI * k * n / N;
            sum += input[n] * std::complex<float>(cos(angle), sin(angle));
        }
        output[k] = sum;
    }
}

inline void NoiseCancellation::ifft(const std::vector<std::complex<float>>& input,
                                   std::vector<float>& output) {
    // Simplified IDFT
    size_t N = std::min(input.size(), output.size());
    
    for (size_t n = 0; n < N; ++n) {
        std::complex<float> sum(0.0f, 0.0f);
        for (size_t k = 0; k < N; ++k) {
            float angle = 2.0f * M_PI * k * n / N;
            sum += input[k] * std::complex<float>(cos(angle), sin(angle));
        }
        output[n] = sum.real() / N;
    }
}

inline std::vector<float> NoiseCancellation::hann_window(size_t size) {
    std::vector<float> window(size);
    for (size_t i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - cos(2.0f * M_PI * i / (size - 1)));
    }
    return window;
}

inline std::vector<float> NoiseCancellation::apply_window(const std::vector<float>& frame) {
    std::vector<float> windowed(frame.size());
    for (size_t i = 0; i < frame.size() && i < window_.size(); ++i) {
        windowed[i] = frame[i] * window_[i];
    }
    return windowed;
}

inline std::vector<std::complex<float>> NoiseCancellation::spectral_subtraction(
    const std::vector<std::complex<float>>& spectrum) {
    
    std::vector<std::complex<float>> result(spectrum.size());
    
    for (size_t i = 0; i < spectrum.size() && i < noise_profile_.size(); ++i) {
        float magnitude = std::abs(spectrum[i]);
        float phase = std::arg(spectrum[i]);
        
        // Subtract noise with over-subtraction factor
        float clean_magnitude = magnitude - config_.over_subtraction * noise_profile_[i];
        
        // Apply spectral floor
        clean_magnitude = std::max(clean_magnitude, config_.spectral_floor * magnitude);
        
        // Reconstruct complex number
        result[i] = std::polar(clean_magnitude, phase);
    }
    
    return result;
}

inline std::vector<std::complex<float>> NoiseCancellation::wiener_filter(
    const std::vector<std::complex<float>>& spectrum) {
    
    std::vector<std::complex<float>> result(spectrum.size());
    
    for (size_t i = 0; i < spectrum.size() && i < noise_profile_.size(); ++i) {
        float magnitude = std::abs(spectrum[i]);
        float noise_power = noise_profile_[i] * noise_profile_[i];
        float signal_power = magnitude * magnitude;
        
        // Wiener gain
        float gain = signal_power / (signal_power + noise_power + 1e-10f);
        
        // Apply gain
        result[i] = spectrum[i] * gain;
    }
    
    return result;
}

inline std::vector<float> NoiseCancellation::echo_cancellation(
    const std::vector<float>& input) {
    
    if (!config_.use_echo_cancellation) {
        return input;
    }
    
    std::vector<float> output(input.size());
    
    // Simple adaptive filter (NLMS algorithm)
    const float step_size = 0.01f;
    const float regularization = 0.001f;
    
    for (size_t i = 0; i < input.size(); ++i) {
        // Predict echo
        float echo_estimate = 0.0f;
        for (size_t j = 0; j < adaptive_filter_.size() && j < reference_buffer_.size(); ++j) {
            echo_estimate += adaptive_filter_[j] * reference_buffer_[j];
        }
        
        // Subtract estimated echo
        float error = input[i] - echo_estimate;
        output[i] = error;
        
        // Update adaptive filter (NLMS)
        float norm = 0.0f;
        for (float ref : reference_buffer_) {
            norm += ref * ref;
        }
        norm += regularization;
        
        float update_factor = step_size * error / norm;
        for (size_t j = 0; j < adaptive_filter_.size() && j < reference_buffer_.size(); ++j) {
            adaptive_filter_[j] += update_factor * reference_buffer_[j];
        }
        
        // Update reference buffer
        reference_buffer_.push_front(input[i]);
        if (reference_buffer_.size() > config_.frame_size) {
            reference_buffer_.pop_back();
        }
    }
    
    return output;
}

inline void NoiseCancellation::estimate_noise(
    const std::vector<std::complex<float>>& spectrum) {
    
    const float alpha = 0.95f;  // Smoothing factor
    
    for (size_t i = 0; i < spectrum.size() && i < noise_profile_.size(); ++i) {
        float magnitude = std::abs(spectrum[i]);
        
        // Exponential averaging
        if (frames_for_noise_ == 0) {
            noise_profile_[i] = magnitude;
        } else {
            noise_profile_[i] = alpha * noise_profile_[i] + (1.0f - alpha) * magnitude;
        }
    }
}

inline void NoiseCancellation::reset() {
    input_buffer_.clear();
    reference_buffer_.clear();
    std::fill(noise_profile_.begin(), noise_profile_.end(), 0.0f);
    std::fill(prev_magnitude_.begin(), prev_magnitude_.end(), 0.0f);
    std::fill(adaptive_filter_.begin(), adaptive_filter_.end(), 0.0f);
    frames_for_noise_ = 0;
    noise_estimated_ = false;
    stats_ = Stats();
}

}  // namespace vtt