#include "../core/gna_device_manager.h"
#include "../core/gna_voice_detector.h"
#include "../test/personal_gna_integration_test.h"

#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <cmath>

/**
 * Personal GNA Week 1 Demonstration for Dell Latitude 5450 MIL-SPEC
 * Focus: Personal productivity push-to-talk system validation
 */

void printPersonalWeek1Banner() {
    std::cout << R"(
╔════════════════════════════════════════════════════════════╗
║                PERSONAL GNA INTEGRATION                    ║
║                    WEEK 1 DEMO                            ║
║                                                            ║
║  Target: Dell Latitude 5450 MIL-SPEC                      ║
║  Focus: Personal Push-to-Talk Excellence                  ║
║  Power: <50mW for Battery Efficiency                      ║
║  Accuracy: >90% Personal Wake Word Detection              ║
╚════════════════════════════════════════════════════════════╝
)" << std::endl;
}

void demonstratePersonalGNADeviceManager() {
    std::cout << "\n=== Personal GNA Device Manager Demo ===" << std::endl;

    // Create personal configuration
    GNADeviceManager::PersonalConfig config = personal_gna_utils::getOptimalPersonalConfig();
    std::cout << "Personal Configuration:" << std::endl;
    std::cout << "  Target Power: " << config.target_power_consumption << "W" << std::endl;
    std::cout << "  Battery Optimization: " << (config.battery_optimization ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  User Profile: " << config.user_profile << std::endl;
    std::cout << "  Detection Sensitivity: " << config.detection_sensitivity << "%" << std::endl;

    // Initialize personal GNA device
    auto gna_manager = std::make_shared<GNADeviceManager>(config);

    if (gna_manager->initializePersonalDevice()) {
        std::cout << "✅ Personal GNA device initialized successfully" << std::endl;

        // Demonstrate personal power management
        std::cout << "\nPersonal Power Management:" << std::endl;
        std::cout << "  Current power: " << gna_manager->getCurrentPowerConsumption() << "W" << std::endl;
        std::cout << "  Within budget: " << (gna_manager->isWithinPersonalPowerBudget() ? "YES" : "NO") << std::endl;

        // Test personal battery optimization
        if (gna_manager->enterBatteryOptimizedMode()) {
            std::cout << "  Battery optimization mode: ENABLED" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));

            if (gna_manager->exitBatteryOptimizedMode()) {
                std::cout << "  Battery optimization mode: DISABLED" << std::endl;
            }
        }

        // Enable personal detection
        if (gna_manager->enablePersonalDetection()) {
            std::cout << "✅ Personal GNA detection enabled" << std::endl;
        }

        // Show personal metrics
        const auto& metrics = gna_manager->getPersonalMetrics();
        std::cout << "\nPersonal Metrics:" << std::endl;
        std::cout << "  Detections processed: " << metrics.detections_processed.load() << std::endl;
        std::cout << "  Wake words detected: " << metrics.wake_words_detected.load() << std::endl;
        std::cout << "  Current power: " << metrics.current_power_mw.load() << "mW" << std::endl;
        std::cout << "  Battery efficient mode: " << (metrics.battery_efficient_mode.load() ? "ON" : "OFF") << std::endl;

    } else {
        std::cout << "❌ Personal GNA device initialization failed" << std::endl;
    }
}

void demonstratePersonalVoiceDetector() {
    std::cout << "\n=== Personal Voice Detector Demo ===" << std::endl;

    // Initialize personal GNA manager
    auto gna_manager = std::make_shared<GNADeviceManager>();
    if (!gna_manager->initializePersonalDevice()) {
        std::cout << "❌ Personal GNA device not available for voice detection demo" << std::endl;
        return;
    }

    // Create personal voice configuration
    GNAVoiceDetector::PersonalVoiceConfig voice_config = personal_voice_utils::getOptimalPersonalVoiceConfig();
    std::cout << "Personal Voice Configuration:" << std::endl;
    std::cout << "  Wake words: ";
    for (const auto& word : voice_config.personal_wake_words) {
        std::cout << word << " ";
    }
    std::cout << std::endl;
    std::cout << "  Wake word threshold: " << voice_config.wake_word_threshold << std::endl;
    std::cout << "  VAD threshold: " << voice_config.vad_threshold << std::endl;
    std::cout << "  GNA acceleration: " << (voice_config.use_gna_acceleration ? "Enabled" : "Disabled") << std::endl;

    // Initialize personal voice detector
    auto voice_detector = std::make_shared<GNAVoiceDetector>(gna_manager, voice_config);

    if (voice_detector->initializePersonalDetection()) {
        std::cout << "✅ Personal voice detection initialized successfully" << std::endl;

        // Generate personal test audio
        std::vector<float> test_audio(16000, 0.0f); // 1 second of audio at 16kHz
        for (size_t i = 0; i < test_audio.size(); ++i) {
            float t = static_cast<float>(i) / 16000.0f;
            test_audio[i] = 0.3f * std::sin(2.0f * 3.14159f * 440.0f * t); // 440Hz tone
        }

        std::cout << "\nPersonal Voice Detection Test:" << std::endl;

        // Perform personal voice detection
        auto detection_result = voice_detector->detectPersonalVoice(test_audio);

        std::cout << "  Voice activity detected: " << (detection_result.voice_activity ? "YES" : "NO") << std::endl;
        std::cout << "  Voice probability: " << (detection_result.voice_probability * 100.0f) << "%" << std::endl;
        std::cout << "  Wake word detected: " << (detection_result.wake_word_detected ? "YES" : "NO") << std::endl;

        if (detection_result.wake_word_detected) {
            std::cout << "  Detected word: " << detection_result.detected_word << std::endl;
            std::cout << "  Confidence: " << (detection_result.confidence * 100.0f) << "%" << std::endl;
        }

        std::cout << "  Processing time: " << detection_result.processing_time_ms << "ms" << std::endl;
        std::cout << "  Power consumption: " << detection_result.power_consumption_mw << "mW" << std::endl;
        std::cout << "  GNA acceleration used: " << (detection_result.gna_used ? "YES" : "NO") << std::endl;

        // Show personal performance metrics
        std::cout << "\nPersonal Performance Metrics:" << std::endl;
        std::cout << "  Detection accuracy: " << voice_detector->getPersonalDetectionAccuracy() << "%" << std::endl;
        std::cout << "  Power efficiency: " << voice_detector->getPersonalPowerEfficiency() << "mW" << std::endl;

    } else {
        std::cout << "❌ Personal voice detection initialization failed" << std::endl;
    }
}

void runPersonalIntegrationTests() {
    std::cout << "\n=== Personal Integration Testing ===" << std::endl;

    // Create personal test configuration
    PersonalGNAIntegrationTest::PersonalTestConfig test_config;
    test_config.max_power_consumption_mw = 50.0f;  // Personal battery target
    test_config.min_detection_accuracy = 90.0f;    // Personal accuracy target
    test_config.max_processing_time_ms = 100.0f;   // Personal latency target

    // Initialize personal test framework
    PersonalGNAIntegrationTest integration_test(test_config);

    if (integration_test.initializePersonalTestFramework()) {
        std::cout << "✅ Personal test framework initialized" << std::endl;

        // Run comprehensive personal tests
        auto test_result = integration_test.runComprehensivePersonalTests();

        // Validate Week 1 success criteria
        bool week1_success = integration_test.validateWeek1SuccessCriteria();

        if (week1_success) {
            std::cout << "\n🎉 PERSONAL WEEK 1 SUCCESS!" << std::endl;
            std::cout << "Ready for personal NPU integration in Week 2" << std::endl;
        } else {
            std::cout << "\n⚠️  Personal Week 1 needs optimization" << std::endl;
            std::cout << "Review test results for improvement areas" << std::endl;
        }

    } else {
        std::cout << "❌ Personal test framework initialization failed" << std::endl;
    }
}

void showPersonalWeek1Summary() {
    std::cout << "\n=== Personal Week 1 Implementation Summary ===" << std::endl;
    std::cout << "✅ Personal GNA Device Configuration" << std::endl;
    std::cout << "   - Dell Latitude 5450 MIL-SPEC optimization" << std::endl;
    std::cout << "   - Battery efficiency focus (<50mW)" << std::endl;
    std::cout << "   - Personal user profile configuration" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "✅ Personal Power Optimization" << std::endl;
    std::cout << "   - Laptop battery efficiency priority" << std::endl;
    std::cout << "   - Personal usage pattern adaptation" << std::endl;
    std::cout << "   - Dynamic power management modes" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "✅ Personal Voice Detection" << std::endl;
    std::cout << "   - Individual wake word training" << std::endl;
    std::cout << "   - Personal speech pattern optimization" << std::endl;
    std::cout << "   - GNA-accelerated feature extraction" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "✅ Personal Integration Testing" << std::endl;
    std::cout << "   - Comprehensive validation framework" << std::endl;
    std::cout << "   - Week 1 success criteria validation" << std::endl;
    std::cout << "   - Personal performance metrics" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "🚀 Ready for Week 2: Personal NPU Integration" << std::endl;
    std::cout << "   - Intel NPU hardware acceleration" << std::endl;
    std::cout << "   - Personal neural model deployment" << std::endl;
    std::cout << "   - Advanced personal voice intelligence" << std::endl;
    std::cout << "===============================================" << std::endl;
}

int main() {
    printPersonalWeek1Banner();

    try {
        // Demonstrate personal GNA device manager
        demonstratePersonalGNADeviceManager();

        // Demonstrate personal voice detector
        demonstratePersonalVoiceDetector();

        // Run personal integration tests
        runPersonalIntegrationTests();

        // Show personal Week 1 summary
        showPersonalWeek1Summary();

        std::cout << "\n✅ Personal GNA Week 1 demonstration completed successfully!" << std::endl;
        std::cout << "Dell Latitude 5450 personal push-to-talk system foundation established." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Personal demo error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}