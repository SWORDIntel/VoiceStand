#include "src/core/wake_word_detector.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "🔍 Debug Wake Word Detection Pipeline\n";
    std::cout << "========================================\n\n";

    // Minimal configuration
    vtt::WakeWordDetector::Config config;
    config.detection_threshold = 0.01f;  // Extremely low threshold
    config.use_vad = false;               // Disable VAD
    config.window_size_ms = 400;          // Window size
    config.buffer_size_ms = 1000;         // Buffer size

    vtt::WakeWordDetector detector(config);

    // Register single word with very low threshold
    detector.register_wake_word("test", nullptr, 0.01f);
    std::cout << "✓ Registered wake word: 'test'\n";

    // Generate test audio - longer duration to fill buffer
    size_t samples = 20000;  // 1.25 seconds at 16kHz
    std::vector<float> audio(samples);

    for (size_t i = 0; i < samples; ++i) {
        float t = static_cast<float>(i) / 16000.0f;
        // Multi-frequency signal closer to speech
        audio[i] = 0.1f * (
            std::sin(2.0f * 3.14159f * 440.0f * t) +
            0.5f * std::sin(2.0f * 3.14159f * 880.0f * t) +
            0.3f * std::sin(2.0f * 3.14159f * 1320.0f * t)
        );
    }

    std::cout << "✓ Generated " << samples << " audio samples\n";

    // Test processing step by step
    std::cout << "🔄 Processing audio in chunks...\n";

    // Process in smaller chunks to fill buffer gradually
    size_t chunk_size = 1600;  // 100ms chunks
    std::string result;

    for (size_t i = 0; i < audio.size(); i += chunk_size) {
        size_t current_chunk = std::min(chunk_size, audio.size() - i);
        result = detector.process_audio(audio.data() + i, current_chunk);

        if (!result.empty()) {
            std::cout << "   🎯 DETECTED at chunk " << (i/chunk_size) << ": " << result << "\n";
            break;
        }
    }

    std::cout << "\n📊 Detection Results:\n";
    std::cout << "   Final detected word: " << (result.empty() ? "NONE" : result) << "\n";

    if (detector.is_npu_available()) {
        std::cout << "   NPU status: ACTIVE\n";
    }

    const auto& stats = detector.get_stats();
    std::cout << "   Total detections: " << stats.total_detections << "\n";
    std::cout << "   Avg confidence: " << stats.average_confidence << "\n";

    // Test with zero threshold
    detector.register_wake_word("zero", nullptr, 0.0f);
    result = detector.process_audio(audio.data(), audio.size());
    std::cout << "\n🔬 Zero threshold test: " << (result.empty() ? "NONE" : result) << "\n";

    return 0;
}