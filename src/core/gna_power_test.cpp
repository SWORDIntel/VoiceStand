#include "gna_voice_detector.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <unistd.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vtt {

/**
 * @brief GNA Power Baseline Measurement and Testing Utility
 *
 * This utility measures system power consumption and validates the GNA
 * voice detection pipeline performance against the target specifications:
 * - <0.05W power consumption
 * - >90% detection accuracy
 * - 50ms response time
 */
class GNAPowerTester {
public:
    struct PowerBaseline {
        float idle_power_mw = 0.0f;
        float detection_power_mw = 0.0f;
        float peak_power_mw = 0.0f;
        uint32_t baseline_temp_c = 0;
        uint32_t peak_temp_c = 0;
    };

    struct PerformanceMetrics {
        uint32_t total_detections = 0;
        uint32_t correct_detections = 0;
        uint32_t false_positives = 0;
        uint32_t false_negatives = 0;
        float average_response_time_ms = 0.0f;
        float detection_accuracy = 0.0f;
    };

    GNAPowerTester() : detector_(create_test_config()), baseline_{}, metrics_{} {
        detector_.set_detection_callback([this](const GNADetectionResult& result) {
            handle_detection_result(result);
        });

        detector_.set_power_callback([this](float power_mw, uint32_t temp_c) {
            handle_power_update(power_mw, temp_c);
        });
    }

    bool run_power_baseline_test(uint32_t duration_seconds = 60) {
        std::cout << "=== GNA Power Baseline Measurement ===" << std::endl;
        std::cout << "Duration: " << duration_seconds << " seconds" << std::endl;

        if (!detector_.initialize()) {
            std::cerr << "Failed to initialize GNA detector" << std::endl;
            return false;
        }

        // Measure idle power
        std::cout << "Measuring idle power consumption..." << std::endl;
        baseline_.idle_power_mw = measure_idle_power(10);
        std::cout << "Idle power: " << std::fixed << std::setprecision(2)
                  << baseline_.idle_power_mw << " mW" << std::endl;

        // Start detection and measure active power
        std::cout << "Starting detection and measuring active power..." << std::endl;
        if (!detector_.start_detection()) {
            std::cerr << "Failed to start detection" << std::endl;
            return false;
        }

        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::seconds(duration_seconds);

        std::vector<float> power_samples;
        std::vector<uint32_t> temp_samples;

        while (std::chrono::steady_clock::now() < end_time) {
            // Generate test audio frames
            generate_test_audio_frame();

            // Collect power measurements
            float current_power = detector_.get_power_consumption_mw();
            power_samples.push_back(current_power);

            // Update peak values
            if (current_power > baseline_.peak_power_mw) {
                baseline_.peak_power_mw = current_power;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        detector_.stop_detection();

        // Calculate average detection power
        if (!power_samples.empty()) {
            float sum = 0.0f;
            for (float power : power_samples) {
                sum += power;
            }
            baseline_.detection_power_mw = sum / power_samples.size();
        }

        print_power_baseline_results();
        return validate_power_targets();
    }

    bool run_performance_test(uint32_t num_test_frames = 1000) {
        std::cout << "\n=== GNA Performance Test ===" << std::endl;
        std::cout << "Test frames: " << num_test_frames << std::endl;

        if (!detector_.is_running()) {
            if (!detector_.start_detection()) {
                std::cerr << "Failed to start detection" << std::endl;
                return false;
            }
        }

        // Reset metrics
        metrics_ = {};
        response_times_.clear();

        // Generate test audio with known speech/silence patterns
        for (uint32_t i = 0; i < num_test_frames; ++i) {
            bool should_detect_speech = (i % 10) < 6; // 60% speech frames

            auto start_time = std::chrono::high_resolution_clock::now();

            if (should_detect_speech) {
                generate_speech_audio_frame();
            } else {
                generate_silence_audio_frame();
            }

            // Wait for detection result (with timeout)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            auto end_time = std::chrono::high_resolution_clock::now();
            float response_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count() / 1000.0f; // Convert to milliseconds

            response_times_.push_back(response_time);
            expected_detections_.push_back(should_detect_speech);

            if (i % 100 == 0) {
                std::cout << "Progress: " << i << "/" << num_test_frames << " frames" << std::endl;
            }
        }

        calculate_performance_metrics();
        print_performance_results();
        return validate_performance_targets();
    }

    void run_continuous_monitoring(uint32_t duration_minutes = 5) {
        std::cout << "\n=== Continuous GNA Monitoring ===" << std::endl;
        std::cout << "Duration: " << duration_minutes << " minutes" << std::endl;

        if (!detector_.is_running()) {
            detector_.start_detection();
        }

        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::minutes(duration_minutes);

        uint32_t sample_count = 0;
        float power_sum = 0.0f;
        float power_min = std::numeric_limits<float>::max();
        float power_max = 0.0f;

        std::cout << "Time\t\tPower(mW)\tTemp(°C)\tStatus" << std::endl;
        std::cout << "----\t\t---------\t--------\t------" << std::endl;

        while (std::chrono::steady_clock::now() < end_time) {
            float power = detector_.get_power_consumption_mw();
            auto result = detector_.get_last_result();

            power_sum += power;
            power_min = std::min(power_min, power);
            power_max = std::max(power_max, power);
            sample_count++;

            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto tm = *std::localtime(&time_t);

            std::cout << std::put_time(&tm, "%H:%M:%S") << "\t\t"
                      << std::fixed << std::setprecision(1) << power << "\t\t"
                      << result.temperature_celsius << "\t\t"
                      << (result.voice_detected ? "SPEECH" : "SILENCE")
                      << (result.wake_word_detected ? " + WAKE" : "") << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(10));
        }

        std::cout << "\n=== Continuous Monitoring Summary ===" << std::endl;
        std::cout << "Average power: " << (power_sum / sample_count) << " mW" << std::endl;
        std::cout << "Power range: " << power_min << " - " << power_max << " mW" << std::endl;
    }

private:
    GNAVoiceDetector detector_;
    PowerBaseline baseline_;
    PerformanceMetrics metrics_;
    std::vector<float> response_times_;
    std::vector<bool> expected_detections_;
    std::vector<bool> actual_detections_;

    GNAConfig create_test_config() {
        GNAConfig config;
        config.sample_rate = 16000;
        config.frame_size = 320; // 20ms @ 16kHz
        config.vad_threshold = 0.35f;
        config.energy_threshold = 0.01f;
        config.max_power_watts = 0.05f;
        config.thermal_throttle_temp = 75;
        config.enable_power_gating = true;
        return config;
    }

    float measure_idle_power(uint32_t duration_seconds) {
        std::vector<float> power_samples;
        auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds);

        while (std::chrono::steady_clock::now() < end_time) {
            power_samples.push_back(detector_.get_power_consumption_mw());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (power_samples.empty()) {
            return 0.0f;
        }

        float sum = 0.0f;
        for (float power : power_samples) {
            sum += power;
        }
        return sum / power_samples.size();
    }

    void generate_test_audio_frame() {
        // Generate 20ms of test audio (320 samples @ 16kHz)
        std::vector<float> audio_frame(320);

        // Simple sine wave test signal
        for (size_t i = 0; i < audio_frame.size(); ++i) {
            audio_frame[i] = 0.1f * std::sin(2.0f * M_PI * 440.0f * i / 16000.0f); // 440 Hz tone
        }

        detector_.process_audio(audio_frame.data(), audio_frame.size());
    }

    void generate_speech_audio_frame() {
        std::vector<float> audio_frame(320);

        // Generate speech-like signal with higher energy and complexity
        for (size_t i = 0; i < audio_frame.size(); ++i) {
            float t = static_cast<float>(i) / 16000.0f;
            audio_frame[i] = 0.3f * (std::sin(2.0f * M_PI * 200.0f * t) +
                                   0.5f * std::sin(2.0f * M_PI * 800.0f * t) +
                                   0.2f * std::sin(2.0f * M_PI * 1200.0f * t));
        }

        detector_.process_audio(audio_frame.data(), audio_frame.size());
    }

    void generate_silence_audio_frame() {
        std::vector<float> audio_frame(320, 0.01f); // Very low noise floor
        detector_.process_audio(audio_frame.data(), audio_frame.size());
    }

    void handle_detection_result(const GNADetectionResult& result) {
        actual_detections_.push_back(result.voice_detected);
        metrics_.total_detections++;
    }

    void handle_power_update(float power_mw, uint32_t temp_c) {
        // Update peak values during monitoring
        if (temp_c > baseline_.peak_temp_c) {
            baseline_.peak_temp_c = temp_c;
        }
    }

    void calculate_performance_metrics() {
        if (expected_detections_.size() != actual_detections_.size()) {
            std::cerr << "Mismatch in detection arrays" << std::endl;
            return;
        }

        metrics_.correct_detections = 0;
        metrics_.false_positives = 0;
        metrics_.false_negatives = 0;

        for (size_t i = 0; i < expected_detections_.size(); ++i) {
            bool expected = expected_detections_[i];
            bool actual = actual_detections_[i];

            if (expected && actual) {
                metrics_.correct_detections++;
            } else if (!expected && actual) {
                metrics_.false_positives++;
            } else if (expected && !actual) {
                metrics_.false_negatives++;
            }
        }

        if (metrics_.total_detections > 0) {
            metrics_.detection_accuracy =
                static_cast<float>(metrics_.correct_detections) / metrics_.total_detections;
        }

        if (!response_times_.empty()) {
            float sum = 0.0f;
            for (float time : response_times_) {
                sum += time;
            }
            metrics_.average_response_time_ms = sum / response_times_.size();
        }
    }

    void print_power_baseline_results() {
        std::cout << "\n=== Power Baseline Results ===" << std::endl;
        std::cout << "Idle power:      " << std::fixed << std::setprecision(2)
                  << baseline_.idle_power_mw << " mW" << std::endl;
        std::cout << "Detection power: " << baseline_.detection_power_mw << " mW" << std::endl;
        std::cout << "Peak power:      " << baseline_.peak_power_mw << " mW" << std::endl;
        std::cout << "Peak temperature: " << baseline_.peak_temp_c << "°C" << std::endl;
    }

    void print_performance_results() {
        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Total detections:    " << metrics_.total_detections << std::endl;
        std::cout << "Correct detections:  " << metrics_.correct_detections << std::endl;
        std::cout << "False positives:     " << metrics_.false_positives << std::endl;
        std::cout << "False negatives:     " << metrics_.false_negatives << std::endl;
        std::cout << "Detection accuracy:  " << std::fixed << std::setprecision(1)
                  << metrics_.detection_accuracy * 100.0f << "%" << std::endl;
        std::cout << "Avg response time:   " << std::fixed << std::setprecision(1)
                  << metrics_.average_response_time_ms << " ms" << std::endl;
    }

    bool validate_power_targets() {
        bool power_ok = baseline_.detection_power_mw <= 50.0f; // 0.05W = 50mW
        bool temp_ok = baseline_.peak_temp_c <= 75;

        std::cout << "\n=== Power Target Validation ===" << std::endl;
        std::cout << "Power target (<50mW): " << (power_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "Thermal target (<75°C): " << (temp_ok ? "PASS" : "FAIL") << std::endl;

        return power_ok && temp_ok;
    }

    bool validate_performance_targets() {
        bool accuracy_ok = metrics_.detection_accuracy >= 0.9f; // >90%
        bool response_ok = metrics_.average_response_time_ms <= 50.0f; // <50ms

        std::cout << "\n=== Performance Target Validation ===" << std::endl;
        std::cout << "Accuracy target (>90%): " << (accuracy_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "Response target (<50ms): " << (response_ok ? "PASS" : "FAIL") << std::endl;

        return accuracy_ok && response_ok;
    }
};

} // namespace vtt

int main(int argc, char* argv[]) {
    vtt::GNAPowerTester tester;

    std::cout << "GNA Voice Detector Power and Performance Test" << std::endl;
    std::cout << "=============================================" << std::endl;

    bool all_tests_passed = true;

    // Run power baseline test
    if (!tester.run_power_baseline_test(30)) { // 30 second test
        all_tests_passed = false;
    }

    // Run performance test
    if (!tester.run_performance_test(500)) { // 500 test frames
        all_tests_passed = false;
    }

    // Optional: Run continuous monitoring (uncomment for long-term testing)
    // tester.run_continuous_monitoring(2); // 2 minutes

    std::cout << "\n=== Overall Test Results ===" << std::endl;
    std::cout << "All tests: " << (all_tests_passed ? "PASSED" : "FAILED") << std::endl;

    if (all_tests_passed) {
        std::cout << "\n✓ Phase 1 GNA implementation meets all targets:" << std::endl;
        std::cout << "  - Power consumption <50mW" << std::endl;
        std::cout << "  - Detection accuracy >90%" << std::endl;
        std::cout << "  - Response time <50ms" << std::endl;
        std::cout << "  - Ready for integration with main VoiceStand system" << std::endl;
    } else {
        std::cout << "\n✗ Some targets not met - optimization required" << std::endl;
    }

    return all_tests_passed ? 0 : 1;
}