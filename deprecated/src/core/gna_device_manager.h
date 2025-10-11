#pragma once

#include <string>
#include <memory>
#include <chrono>
#include <atomic>

/**
 * Personal GNA Device Manager for Dell Latitude 5450 MIL-SPEC
 * Week 1 Implementation: Personal productivity system optimization
 *
 * Focus: Individual user configuration, battery efficiency, personal patterns
 */

class GNADeviceManager {
public:
    struct PersonalConfig {
        // Personal power optimization
        double target_power_consumption = 0.05;  // <0.05W for laptop battery
        bool battery_optimization = true;

        // Personal usage patterns
        std::string user_profile = "personal_productivity";
        int detection_sensitivity = 85;  // Personal preference
        bool adaptive_learning = true;

        // Personal device settings
        std::string device_path = "/dev/accel/accel0";
        uint32_t context_size = 512;  // Personal context window
        uint32_t batch_size = 1;      // Single user optimization
    };

    struct PersonalMetrics {
        std::atomic<double> current_power_mw{0.0};
        std::atomic<uint64_t> detections_processed{0};
        std::atomic<uint64_t> wake_words_detected{0};
        std::atomic<double> detection_accuracy{0.0};
        std::chrono::steady_clock::time_point last_detection;
        std::atomic<bool> battery_efficient_mode{true};
    };

    enum class PersonalDeviceState {
        UNINITIALIZED,
        INITIALIZING,
        READY,
        ACTIVE,
        POWER_SAVING,
        ERROR
    };

    // Constructor for personal configuration
    GNADeviceManager();
    explicit GNADeviceManager(const PersonalConfig& config);
    ~GNADeviceManager();

    // Core personal GNA operations
    bool initializePersonalDevice();
    bool configurePersonalPowerProfile();
    bool enablePersonalDetection();
    void shutdown();

    // Personal power management
    bool enterBatteryOptimizedMode();
    bool exitBatteryOptimizedMode();
    double getCurrentPowerConsumption() const;
    bool isWithinPersonalPowerBudget() const;

    // Personal device status
    PersonalDeviceState getDeviceState() const { return device_state_; }
    const PersonalMetrics& getPersonalMetrics() const { return metrics_; }
    bool isPersonalDeviceReady() const;

    // Personal configuration updates
    void updatePersonalProfile(const std::string& profile);
    void adjustPersonalSensitivity(int sensitivity);
    bool validatePersonalConfiguration() const;

private:
    PersonalConfig config_;
    PersonalMetrics metrics_;
    std::atomic<PersonalDeviceState> device_state_{PersonalDeviceState::UNINITIALIZED};

    // Personal device handles
    int device_fd_ = -1;
    void* gna_context_ = nullptr;

    // Personal power management
    bool configurePowerConstraints();
    bool setupPersonalClocking();
    void monitorPersonalPowerUsage();

    // Personal error handling
    std::string last_error_;
    bool handlePersonalDeviceError(const std::string& error);
    void logPersonalMetrics() const;
};

// Personal utility functions for Week 1
namespace personal_gna_utils {
    bool detectGNACapability();
    std::string getPersonalDeviceInfo();
    bool validatePersonalPowerBudget(double target_mw);
    GNADeviceManager::PersonalConfig getOptimalPersonalConfig();
}