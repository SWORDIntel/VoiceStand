#include "enhanced_whisper_processor.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace vtt {

void EnhancedWhisperProcessor::vad_stage(MemoryPool<float>::BlockPtr buffer) {
    if (!buffer || buffer->data.empty()) return;
    
    const float* samples = buffer->data.data();
    size_t num_samples = buffer->data.size();
    
    // Enhanced VAD detection
    bool has_speech = detect_speech_enhanced(samples, num_samples);
    
    if (has_speech) {
        // Pass to next stage if speech detected
        if (pipeline_) {
            // Note: In real implementation, would pass to next stage
            // For now, we'll process inline
            resample_stage(buffer);
        }
    } else {
        // Return buffer to pool if no speech
        pipeline_->get_buffer_pool()->return_buffer(buffer);
    }
}

void EnhancedWhisperProcessor::resample_stage(MemoryPool<float>::BlockPtr buffer) {
    if (!buffer || buffer->data.empty()) return;
    
    // Check if resampling is needed
    // Assuming input is at correct sample rate for now
    // In production, would check actual sample rate
    
    // Pass to whisper stage
    whisper_stage(buffer);
}

void EnhancedWhisperProcessor::whisper_stage(MemoryPool<float>::BlockPtr buffer) {
    if (!buffer || buffer->data.empty() || !ctx_) {
        if (buffer) {
            pipeline_->get_buffer_pool()->return_buffer(buffer);
        }
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Transcribe using base class method
    std::string transcription = transcribe_audio(
        buffer->data.data(), 
        buffer->data.size()
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    metrics_.chunks_processed++;
    metrics_.update_latency(latency);
    
    // Send transcription result if available
    if (!transcription.empty()) {
        TranscriptionResult result;
        result.text = transcription;
        result.is_final = true;
        result.confidence = 0.95f;  // Placeholder
        
        // Call transcription callback
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (transcription_callback_) {
            transcription_callback_(result);
        }
    }
    
    // Return buffer to pool
    pipeline_->get_buffer_pool()->return_buffer(buffer);
}

void EnhancedWhisperProcessor::postprocess_stage(MemoryPool<float>::BlockPtr buffer) {
    // Post-processing could include:
    // - Punctuation restoration
    // - Capitalization
    // - Number formatting
    // - Custom vocabulary replacement
    
    // For now, just return buffer to pool
    if (buffer) {
        pipeline_->get_buffer_pool()->return_buffer(buffer);
    }
}

bool EnhancedWhisperProcessor::detect_speech_enhanced(const float* samples, size_t num_samples) {
    if (!samples || num_samples == 0) return false;
    
    // Calculate energy
    float energy = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += samples[i] * samples[i];
    }
    energy = std::sqrt(energy / num_samples);
    
    // Calculate zero-crossing rate
    size_t zero_crossings = 0;
    for (size_t i = 1; i < num_samples; ++i) {
        if ((samples[i-1] >= 0 && samples[i] < 0) ||
            (samples[i-1] < 0 && samples[i] >= 0)) {
            zero_crossings++;
        }
    }
    float zcr = static_cast<float>(zero_crossings) / num_samples;
    
    // Enhanced VAD logic
    bool is_speech = false;
    
    if (energy > vad_state_.energy_threshold && zcr < vad_state_.zcr_threshold) {
        vad_state_.speech_frames++;
        vad_state_.silence_frames = 0;
        
        if (vad_state_.speech_frames >= VADState::MIN_SPEECH_FRAMES) {
            vad_state_.is_speaking = true;
        }
    } else {
        vad_state_.silence_frames++;
        vad_state_.speech_frames = 0;
        
        if (vad_state_.silence_frames >= VADState::MIN_SILENCE_FRAMES) {
            vad_state_.is_speaking = false;
        }
    }
    
    return vad_state_.is_speaking;
}

std::vector<float> EnhancedWhisperProcessor::resample_optimized(
    const float* input, size_t input_size,
    uint32_t input_rate, uint32_t output_rate) {
    
    if (input_rate == output_rate) {
        return std::vector<float>(input, input + input_size);
    }
    
    // Use linear interpolation for resampling
    // In production, would use a proper resampling library
    double ratio = static_cast<double>(output_rate) / input_rate;
    size_t output_size = static_cast<size_t>(input_size * ratio);
    std::vector<float> output(output_size);
    
    // SIMD-optimized resampling could go here
    if (use_simd_) {
        // Placeholder for SIMD implementation
        // Would use AVX/SSE instructions for x86
    }
    
    // Fallback to scalar implementation
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

}  // namespace vtt