#pragma once

#include "whisper_processor.h"
#include "pipeline.h"
#include "memory_pool.h"
#include <atomic>
#include <chrono>

namespace vtt {

// Enhanced Whisper processor with multi-threaded pipeline and memory pooling
class EnhancedWhisperProcessor : public WhisperProcessor {
public:
    EnhancedWhisperProcessor();
    ~EnhancedWhisperProcessor();
    
    // Override base class methods
    bool initialize(const WhisperConfig& config) override;
    void process_audio(const float* samples, size_t num_samples, 
                      uint32_t sample_rate) override;
    void cleanup() override;
    
    // Performance metrics
    struct PerformanceMetrics {
        std::atomic<size_t> total_samples_processed{0};
        std::atomic<size_t> chunks_processed{0};
        std::atomic<size_t> total_latency_ms{0};
        std::atomic<size_t> min_latency_ms{SIZE_MAX};
        std::atomic<size_t> max_latency_ms{0};
        std::atomic<size_t> dropped_chunks{0};
        
        void update_latency(size_t latency_ms) {
            total_latency_ms += latency_ms;
            
            size_t min_lat = min_latency_ms.load();
            while (latency_ms < min_lat && 
                   !min_latency_ms.compare_exchange_weak(min_lat, latency_ms));
                   
            size_t max_lat = max_latency_ms.load();
            while (latency_ms > max_lat && 
                   !max_latency_ms.compare_exchange_weak(max_lat, latency_ms));
        }
        
        size_t average_latency_ms() const {
            size_t chunks = chunks_processed.load();
            return chunks > 0 ? total_latency_ms.load() / chunks : 0;
        }
    };
    
    const PerformanceMetrics& get_metrics() const { return metrics_; }
    void print_performance_report() const;
    
private:
    // Pipeline stages
    void vad_stage(MemoryPool<float>::BlockPtr buffer);
    void resample_stage(MemoryPool<float>::BlockPtr buffer);
    void whisper_stage(MemoryPool<float>::BlockPtr buffer);
    void postprocess_stage(MemoryPool<float>::BlockPtr buffer);
    
    // Enhanced VAD with energy and zero-crossing rate
    bool detect_speech_enhanced(const float* samples, size_t num_samples);
    
    // Optimized resampling
    std::vector<float> resample_optimized(const float* input, size_t input_size,
                                          uint32_t input_rate, uint32_t output_rate);
    
    std::unique_ptr<AudioPipeline> pipeline_;
    PerformanceMetrics metrics_;
    
    // VAD state
    struct VADState {
        float energy_threshold = 0.01f;
        float zcr_threshold = 0.1f;
        int speech_frames = 0;
        int silence_frames = 0;
        bool is_speaking = false;
        
        static constexpr int MIN_SPEECH_FRAMES = 5;
        static constexpr int MIN_SILENCE_FRAMES = 10;
    };
    VADState vad_state_;
    
    // Optimization flags
    bool use_gpu_ = false;
    bool use_simd_ = true;
    int num_threads_ = 4;
};

// Implementation
inline EnhancedWhisperProcessor::EnhancedWhisperProcessor() 
    : pipeline_(std::make_unique<AudioPipeline>()) {
}

inline EnhancedWhisperProcessor::~EnhancedWhisperProcessor() {
    cleanup();
}

inline bool EnhancedWhisperProcessor::initialize(const WhisperConfig& config) {
    // Initialize base class
    if (!WhisperProcessor::initialize(config)) {
        return false;
    }
    
    // Set optimization flags
    num_threads_ = config.num_threads;
    use_gpu_ = config.use_gpu;
    
    // Setup pipeline stages
    pipeline_->add_stage("VAD", 
        [this](auto buffer) { vad_stage(buffer); });
        
    pipeline_->add_stage("Resample", 
        [this](auto buffer) { resample_stage(buffer); });
        
    pipeline_->add_stage("Whisper", 
        [this](auto buffer) { whisper_stage(buffer); });
        
    pipeline_->add_stage("PostProcess", 
        [this](auto buffer) { postprocess_stage(buffer); });
    
    // Start pipeline
    pipeline_->start();
    
    std::cout << "[INFO] Enhanced Whisper processor initialized with " 
              << num_threads_ << " threads\n";
    
    return true;
}

inline void EnhancedWhisperProcessor::process_audio(const float* samples, 
                                                   size_t num_samples, 
                                                   uint32_t sample_rate) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use pipeline for processing
    pipeline_->process_audio(samples, num_samples);
    
    metrics_.total_samples_processed += num_samples;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    if (latency > 0) {
        metrics_.update_latency(latency);
    }
}

inline void EnhancedWhisperProcessor::cleanup() {
    if (pipeline_) {
        pipeline_->stop();
        pipeline_->print_stats();
    }
    
    print_performance_report();
    
    WhisperProcessor::cleanup();
}

inline void EnhancedWhisperProcessor::print_performance_report() const {
    printf("\n=== Enhanced Whisper Performance Report ===\n");
    printf("Total samples processed: %zu\n", metrics_.total_samples_processed.load());
    printf("Chunks processed: %zu\n", metrics_.chunks_processed.load());
    printf("Dropped chunks: %zu\n", metrics_.dropped_chunks.load());
    printf("Average latency: %zu ms\n", metrics_.average_latency_ms());
    printf("Min latency: %zu ms\n", metrics_.min_latency_ms.load());
    printf("Max latency: %zu ms\n", metrics_.max_latency_ms.load());
    
    float throughput = metrics_.total_samples_processed.load() / 16000.0f;  // seconds
    printf("Throughput: %.2f seconds of audio\n", throughput);
    printf("==========================================\n");
}

}  // namespace vtt