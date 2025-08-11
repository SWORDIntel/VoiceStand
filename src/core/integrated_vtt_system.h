#pragma once

#include "enhanced_whisper_processor.h"
#include "speaker_diarization.h"
#include "punctuation_restoration.h"
#include "wake_word_detector.h"
#include "noise_cancellation.h"
#include "pipeline.h"
#include <json/json.h>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <memory>
#include <atomic>
#include <functional>

namespace vtt {

// Integrated Voice-to-Text system with all advanced features
class IntegratedVTTSystem {
public:
    struct Config {
        // Core settings
        WhisperConfig whisper_config;
        bool enable_wake_words = true;
        bool enable_noise_cancellation = true;
        bool enable_speaker_diarization = true;
        bool enable_punctuation = true;
        
        // Wake word settings
        std::vector<std::string> wake_words = {"hey computer", "okay system"};
        float wake_word_threshold = 0.75f;
        
        // Processing settings
        bool auto_start_on_wake = true;
        size_t silence_timeout_ms = 2000;
        bool continuous_listening = false;
    };
    
    // Transcription result with all enhancements
    struct EnhancedResult {
        std::string text;               // Final transcribed text
        std::string raw_text;           // Raw text without punctuation
        std::string speaker_id;         // Speaker identification
        std::string speaker_name;       // Speaker name
        float confidence;               // Overall confidence
        bool is_final;                  // Is this a final result
        std::chrono::steady_clock::time_point timestamp;
        
        // Additional metadata
        float noise_reduction_db = 0.0f;
        bool wake_word_triggered = false;
        std::string wake_word_used;
    };
    
    using ResultCallback = std::function<void(const EnhancedResult&)>;
    
    IntegratedVTTSystem(const Config& config = Config());
    ~IntegratedVTTSystem();
    
    // Initialize all subsystems
    bool initialize();
    
    // Process audio through full pipeline
    void process_audio(const float* samples, size_t num_samples, uint32_t sample_rate);
    
    // Control methods
    void start_listening();
    void stop_listening();
    void toggle_listening();
    bool is_listening() const { return is_listening_; }
    
    // Wake word management
    void register_wake_word(const std::string& word, std::function<void()> callback = nullptr);
    void train_wake_word(const std::string& word, const std::vector<std::vector<float>>& samples);
    
    // Speaker management
    void add_speaker_profile(const std::string& name, const std::vector<float>& voice_sample);
    std::vector<std::string> get_speaker_list() const;
    
    // Callbacks
    void set_result_callback(ResultCallback callback) { result_callback_ = callback; }
    void set_wake_word_callback(std::function<void(const std::string&)> callback) {
        wake_word_callback_ = callback;
    }
    
    // Get system statistics
    struct SystemStats {
        size_t total_audio_processed_ms = 0;
        size_t total_transcriptions = 0;
        size_t wake_word_detections = 0;
        size_t speaker_changes = 0;
        float average_confidence = 0.0f;
        float average_latency_ms = 0.0f;
        float noise_reduction_avg_db = 0.0f;
    };
    
    SystemStats get_stats() const;
    void print_performance_report() const;
    
    // Export session data
    std::string export_session() const;
    
private:
    // Pipeline processing stages
    void noise_cancellation_stage(const float* samples, size_t num_samples);
    void wake_word_stage(const float* samples, size_t num_samples);
    void whisper_stage(const float* samples, size_t num_samples);
    void diarization_stage(const std::string& text);
    void punctuation_stage(std::string& text);
    
    // Handle complete transcription
    void handle_transcription(const std::string& text, float confidence);
    
    // Silence detection
    bool detect_silence(const float* samples, size_t num_samples);
    
    Config config_;
    
    // Core components
    std::unique_ptr<EnhancedWhisperProcessor> whisper_;
    std::unique_ptr<SpeakerDiarization> diarization_;
    std::unique_ptr<PunctuationRestoration> punctuation_;
    std::unique_ptr<WakeWordDetector> wake_detector_;
    std::unique_ptr<NoiseCancellation> noise_cancellation_;
    
    // State
    std::atomic<bool> is_listening_{false};
    std::atomic<bool> wake_word_active_{false};
    std::string current_speaker_;
    std::string accumulated_text_;
    
    // Timing
    std::chrono::steady_clock::time_point last_speech_time_;
    std::chrono::steady_clock::time_point session_start_time_;
    
    // Callbacks
    ResultCallback result_callback_;
    std::function<void(const std::string&)> wake_word_callback_;
    
    // Statistics
    mutable SystemStats stats_;
    
    // Buffers
    std::vector<float> processed_buffer_;
    size_t silence_frames_ = 0;
    static constexpr size_t SILENCE_THRESHOLD_FRAMES = 50;
};

// Implementation
inline IntegratedVTTSystem::IntegratedVTTSystem(const Config& config) 
    : config_(config) {
    
    // Initialize components
    whisper_ = std::make_unique<EnhancedWhisperProcessor>();
    diarization_ = std::make_unique<SpeakerDiarization>();
    punctuation_ = std::make_unique<PunctuationRestoration>();
    wake_detector_ = std::make_unique<WakeWordDetector>();
    noise_cancellation_ = std::make_unique<NoiseCancellation>();
}

inline IntegratedVTTSystem::~IntegratedVTTSystem() {
    stop_listening();
    print_performance_report();
}

inline bool IntegratedVTTSystem::initialize() {
    // Initialize Whisper
    if (!whisper_->initialize(config_.whisper_config)) {
        std::cerr << "[ERROR] Failed to initialize Whisper processor\n";
        return false;
    }
    
    // Register wake words
    if (config_.enable_wake_words) {
        for (const auto& word : config_.wake_words) {
            wake_detector_->register_wake_word(word, 
                [this, word]() {
                    if (config_.auto_start_on_wake) {
                        start_listening();
                    }
                    if (wake_word_callback_) {
                        wake_word_callback_(word);
                    }
                    stats_.wake_word_detections++;
                },
                config_.wake_word_threshold
            );
        }
    }
    
    // Set up Whisper callback
    whisper_->set_transcription_callback(
        [this](const TranscriptionResult& result) {
            handle_transcription(result.text, result.confidence);
        }
    );
    
    session_start_time_ = std::chrono::steady_clock::now();
    
    std::cout << "[INFO] Integrated VTT System initialized successfully\n";
    std::cout << "Features enabled:\n";
    std::cout << "  - Wake Words: " << (config_.enable_wake_words ? "Yes" : "No") << "\n";
    std::cout << "  - Noise Cancellation: " << (config_.enable_noise_cancellation ? "Yes" : "No") << "\n";
    std::cout << "  - Speaker Diarization: " << (config_.enable_speaker_diarization ? "Yes" : "No") << "\n";
    std::cout << "  - Punctuation: " << (config_.enable_punctuation ? "Yes" : "No") << "\n";
    
    return true;
}

inline void IntegratedVTTSystem::process_audio(const float* samples, size_t num_samples, 
                                              uint32_t sample_rate) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Stage 1: Noise cancellation
    std::vector<float> clean_audio;
    if (config_.enable_noise_cancellation) {
        clean_audio = noise_cancellation_->process(samples, num_samples);
        samples = clean_audio.data();
        
        // Update noise stats
        auto nc_stats = noise_cancellation_->get_stats();
        stats_.noise_reduction_avg_db = nc_stats.average_noise_reduction_db;
    }
    
    // Stage 2: Wake word detection
    if (config_.enable_wake_words && !is_listening_) {
        std::string detected = wake_detector_->process_audio(samples, num_samples);
        if (!detected.empty()) {
            wake_word_active_ = true;
            EnhancedResult result;
            result.wake_word_triggered = true;
            result.wake_word_used = detected;
            result.timestamp = std::chrono::steady_clock::now();
            
            if (result_callback_) {
                result_callback_(result);
            }
        }
    }
    
    // Stage 3: Speech processing (if listening)
    if (is_listening_ || (wake_word_active_ && config_.auto_start_on_wake)) {
        // Check for silence
        if (detect_silence(samples, num_samples)) {
            silence_frames_++;
            
            // Check timeout
            if (silence_frames_ > SILENCE_THRESHOLD_FRAMES && !config_.continuous_listening) {
                // End of speech segment
                if (!accumulated_text_.empty()) {
                    EnhancedResult result;
                    result.text = accumulated_text_;
                    result.is_final = true;
                    result.timestamp = std::chrono::steady_clock::now();
                    
                    if (result_callback_) {
                        result_callback_(result);
                    }
                    
                    accumulated_text_.clear();
                }
                
                if (!config_.continuous_listening) {
                    stop_listening();
                }
            }
        } else {
            silence_frames_ = 0;
            last_speech_time_ = std::chrono::steady_clock::now();
            
            // Process through Whisper
            whisper_->process_audio(samples, num_samples, sample_rate);
        }
    }
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    stats_.total_audio_processed_ms += (num_samples * 1000) / sample_rate;
    stats_.average_latency_ms = (stats_.average_latency_ms * stats_.total_transcriptions + latency) 
                                / (stats_.total_transcriptions + 1);
}

inline void IntegratedVTTSystem::handle_transcription(const std::string& text, float confidence) {
    if (text.empty()) return;
    
    EnhancedResult result;
    result.raw_text = text;
    result.text = text;
    result.confidence = confidence;
    result.timestamp = std::chrono::steady_clock::now();
    
    // Stage 4: Speaker diarization
    if (config_.enable_speaker_diarization) {
        result.speaker_id = diarization_->process_audio(nullptr, 0, 16000);  // Use cached features
        result.speaker_name = "Speaker " + result.speaker_id;
        diarization_->add_transcription(text);
        
        if (result.speaker_id != current_speaker_) {
            stats_.speaker_changes++;
            current_speaker_ = result.speaker_id;
        }
    }
    
    // Stage 5: Punctuation restoration
    if (config_.enable_punctuation) {
        result.text = punctuation_->restore(result.text);
    }
    
    // Update accumulated text
    accumulated_text_ += result.text + " ";
    
    // Update statistics
    stats_.total_transcriptions++;
    stats_.average_confidence = (stats_.average_confidence * (stats_.total_transcriptions - 1) 
                                 + confidence) / stats_.total_transcriptions;
    
    // Send result
    if (result_callback_) {
        result_callback_(result);
    }
}

inline void IntegratedVTTSystem::start_listening() {
    if (!is_listening_) {
        is_listening_ = true;
        silence_frames_ = 0;
        last_speech_time_ = std::chrono::steady_clock::now();
        std::cout << "[INFO] Started listening\n";
    }
}

inline void IntegratedVTTSystem::stop_listening() {
    if (is_listening_) {
        is_listening_ = false;
        wake_word_active_ = false;
        std::cout << "[INFO] Stopped listening\n";
    }
}

inline void IntegratedVTTSystem::toggle_listening() {
    if (is_listening_) {
        stop_listening();
    } else {
        start_listening();
    }
}

inline bool IntegratedVTTSystem::detect_silence(const float* samples, size_t num_samples) {
    if (!samples || num_samples == 0) return true;
    
    // Simple energy-based silence detection
    float energy = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
        energy += samples[i] * samples[i];
    }
    energy = std::sqrt(energy / num_samples);
    
    return energy < 0.01f;  // Threshold
}

inline IntegratedVTTSystem::SystemStats IntegratedVTTSystem::get_stats() const {
    return stats_;
}

inline void IntegratedVTTSystem::print_performance_report() const {
    auto session_duration = std::chrono::steady_clock::now() - session_start_time_;
    auto duration_min = std::chrono::duration_cast<std::chrono::minutes>(session_duration).count();
    
    std::cout << "\n=== Integrated VTT System Performance Report ===\n";
    std::cout << "Session duration: " << duration_min << " minutes\n";
    std::cout << "Total audio processed: " << stats_.total_audio_processed_ms / 1000 << " seconds\n";
    std::cout << "Total transcriptions: " << stats_.total_transcriptions << "\n";
    std::cout << "Wake word detections: " << stats_.wake_word_detections << "\n";
    std::cout << "Speaker changes: " << stats_.speaker_changes << "\n";
    std::cout << "Average confidence: " << std::fixed << std::setprecision(2) 
              << stats_.average_confidence * 100 << "%\n";
    std::cout << "Average latency: " << stats_.average_latency_ms << " ms\n";
    std::cout << "Noise reduction: " << stats_.noise_reduction_avg_db << " dB\n";
    
    // Whisper stats
    auto whisper_metrics = whisper_->get_metrics();
    std::cout << "\nWhisper Performance:\n";
    std::cout << "  Chunks processed: " << whisper_metrics.chunks_processed << "\n";
    std::cout << "  Average latency: " << whisper_metrics.average_latency_ms() << " ms\n";
    std::cout << "  Min/Max latency: " << whisper_metrics.min_latency_ms 
              << "/" << whisper_metrics.max_latency_ms << " ms\n";
    
    std::cout << "===============================================\n";
}

inline std::string IntegratedVTTSystem::export_session() const {
    Json::Value root;
    Json::Value stats_json;
    
    // Add statistics
    stats_json["total_audio_processed_ms"] = static_cast<Json::Int64>(stats_.total_audio_processed_ms);
    stats_json["total_transcriptions"] = static_cast<Json::Int64>(stats_.total_transcriptions);
    stats_json["wake_word_detections"] = static_cast<Json::Int64>(stats_.wake_word_detections);
    stats_json["speaker_changes"] = static_cast<Json::Int64>(stats_.speaker_changes);
    stats_json["average_confidence"] = stats_.average_confidence;
    stats_json["average_latency_ms"] = stats_.average_latency_ms;
    stats_json["noise_reduction_avg_db"] = stats_.noise_reduction_avg_db;
    
    root["statistics"] = stats_json;
    
    // Add diarization results if enabled
    if (config_.enable_speaker_diarization) {
        root["diarization"] = diarization_->export_results();
    }
    
    // Convert to string
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    
    std::stringstream ss;
    writer->write(root, &ss);
    
    return ss.str();
}

}  // namespace vtt