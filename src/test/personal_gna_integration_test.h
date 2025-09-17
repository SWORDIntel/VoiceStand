#pragma once

#include "../core/gna_device_manager.h"
#include "../core/gna_voice_detector.h"
#include <memory>
#include <vector>
#include <string>
#include <chrono>

/**
 * Personal GNA Integration Testing Framework for Dell Latitude 5450
 * Week 1 Implementation: Validate personal push-to-talk system
 *
 * Focus: Personal productivity, battery efficiency, individual validation
 */

class PersonalGNAIntegrationTest {
public:
    struct PersonalTestConfig {
        // Personal test parameters
        bool enable_device_tests = true;      // Test GNA device access
        bool enable_power_tests = true;       // Test battery efficiency
        bool enable_voice_tests = true;       // Test voice detection
        bool enable_performance_tests = true; // Test personal performance

        // Personal validation thresholds
        float max_power_consumption_mw = 50.0f;  // Personal battery target
        float min_detection_accuracy = 90.0f;    // Personal accuracy target
        float max_processing_time_ms = 100.0f;   // Personal latency target
        int min_wake_word_count = 2;             // Personal wake word minimum

        // Personal test audio parameters
        int test_sample_rate = 16000;            // Personal audio format
        int test_frame_duration_ms = 30;         // Personal frame size
        int test_audio_length_seconds = 5;       // Personal test duration
    };

    struct PersonalTestResult {
        bool overall_success = false;
        std::string test_name;
        std::chrono::steady_clock::time_point timestamp;

        // Personal device test results
        bool gna_device_accessible = false;
        bool gna_device_initialized = false;
        std::string gna_device_info;

        // Personal power test results
        bool power_within_budget = false;
        float measured_power_consumption_mw = 0.0f;
        bool battery_optimization_working = false;

        // Personal voice detection results
        bool voice_detection_working = false;
        float detection_accuracy_percent = 0.0f;
        int wake_words_trained = 0;
        bool vad_functional = false;

        // Personal performance results
        float average_processing_time_ms = 0.0f;
        bool meets_latency_target = false;
        bool gna_acceleration_working = false;

        // Personal error information
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
    };

    // Constructor for personal testing
    PersonalGNAIntegrationTest();
    explicit PersonalGNAIntegrationTest(const PersonalTestConfig& config);
    ~PersonalGNAIntegrationTest();

    // Core personal test operations
    bool initializePersonalTestFramework();
    PersonalTestResult runComprehensivePersonalTests();
    bool validateWeek1SuccessCriteria();
    void shutdown();

    // Individual personal test categories
    PersonalTestResult testPersonalGNADeviceAccess();
    PersonalTestResult testPersonalPowerOptimization();
    PersonalTestResult testPersonalVoiceDetection();
    PersonalTestResult testPersonalPerformanceMetrics();

    // Personal test utilities
    std::vector<float> generatePersonalTestAudio(int duration_seconds, float frequency = 440.0f);
    bool simulatePersonalWakeWord(const std::string& wake_word);
    bool validatePersonalConfiguration();

    // Personal reporting
    void generatePersonalTestReport(const PersonalTestResult& result);
    void logPersonalTestSummary();
    std::string getPersonalTestStatusSummary() const;

private:
    PersonalTestConfig config_;
    std::shared_ptr<GNADeviceManager> gna_manager_;
    std::shared_ptr<GNAVoiceDetector> voice_detector_;

    // Personal test state
    bool framework_initialized_ = false;
    std::vector<PersonalTestResult> test_results_;

    // Personal test data
    std::vector<std::vector<float>> personal_test_audio_samples_;
    std::vector<std::string> personal_test_wake_words_;

    // Personal validation helpers
    bool validatePersonalGNACapability();
    bool validatePersonalPowerBudget();
    bool validatePersonalVoiceConfiguration();
    bool validatePersonalPerformanceTargets();

    // Personal test execution helpers
    bool runPersonalDeviceTest(PersonalTestResult& result);
    bool runPersonalPowerTest(PersonalTestResult& result);
    bool runPersonalVoiceTest(PersonalTestResult& result);
    bool runPersonalPerformanceTest(PersonalTestResult& result);

    // Personal measurement utilities
    float measurePersonalPowerConsumption(int duration_ms);
    float measurePersonalDetectionAccuracy(int num_samples);
    float measurePersonalProcessingLatency(const std::vector<float>& audio_sample);

    // Personal error handling
    void addPersonalError(PersonalTestResult& result, const std::string& error);
    void addPersonalWarning(PersonalTestResult& result, const std::string& warning);
    bool handlePersonalTestFailure(const std::string& test_name, const std::string& error);
};

// Personal test utility functions
namespace personal_test_utils {
    // Personal audio generation
    std::vector<float> generatePersonalSineWave(float frequency, int sample_rate, int duration_ms);
    std::vector<float> generatePersonalWhiteNoise(int sample_rate, int duration_ms, float amplitude = 0.1f);
    std::vector<float> generatePersonalSpeechLikeSignal(int sample_rate, int duration_ms);

    // Personal validation utilities
    bool validatePersonalHardwareRequirements();
    bool validatePersonalSoftwareRequirements();
    std::string getPersonalSystemInfo();

    // Personal test result utilities
    PersonalGNAIntegrationTest::PersonalTestResult createPersonalTestResult(const std::string& test_name);
    bool mergePersonalTestResults(const std::vector<PersonalGNAIntegrationTest::PersonalTestResult>& results,
                                 PersonalGNAIntegrationTest::PersonalTestResult& merged);
    void printPersonalTestResult(const PersonalGNAIntegrationTest::PersonalTestResult& result);

    // Personal Week 1 specific utilities
    bool validatePersonalWeek1Criteria(const PersonalGNAIntegrationTest::PersonalTestResult& result);
    std::string formatPersonalWeek1Report(const PersonalGNAIntegrationTest::PersonalTestResult& result);
    bool recommendPersonalOptimizations(const PersonalGNAIntegrationTest::PersonalTestResult& result);
}