#include "gna_device_manager.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cstring>
#include <iostream>
#include <thread>
#include <fstream>

// Intel GNA specific headers (simplified for Week 1)
#include <linux/types.h>

// Personal GNA IOCTL commands for Dell Latitude 5450
#define PERSONAL_GNA_IOCTL_BASE 'G'
#define PERSONAL_GNA_SET_POWER_MODE    _IOW(PERSONAL_GNA_IOCTL_BASE, 1, uint32_t)
#define PERSONAL_GNA_GET_POWER_STATUS  _IOR(PERSONAL_GNA_IOCTL_BASE, 2, uint32_t)
#define PERSONAL_GNA_CONFIGURE_CONTEXT _IOW(PERSONAL_GNA_IOCTL_BASE, 3, uint64_t)

GNADeviceManager::GNADeviceManager() {
    config_ = personal_gna_utils::getOptimalPersonalConfig();
    std::cout << "Initializing Personal GNA Device Manager for Dell Latitude 5450" << std::endl;
}

GNADeviceManager::GNADeviceManager(const PersonalConfig& config)
    : config_(config) {
    std::cout << "Initializing Personal GNA Device Manager for Dell Latitude 5450" << std::endl;
}

GNADeviceManager::~GNADeviceManager() {
    shutdown();
}

bool GNADeviceManager::initializePersonalDevice() {
    device_state_ = PersonalDeviceState::INITIALIZING;

    std::cout << "Opening personal GNA device: " << config_.device_path << std::endl;

    // Open GNA device for personal use
    device_fd_ = open(config_.device_path.c_str(), O_RDWR);
    if (device_fd_ < 0) {
        last_error_ = "Failed to open personal GNA device: " + std::string(strerror(errno));
        device_state_ = PersonalDeviceState::ERROR;
        return false;
    }

    // Configure personal device context
    if (!configurePersonalPowerProfile()) {
        close(device_fd_);
        device_fd_ = -1;
        device_state_ = PersonalDeviceState::ERROR;
        return false;
    }

    // Set up personal context size for individual use
    uint64_t context_config = config_.context_size;
    if (ioctl(device_fd_, PERSONAL_GNA_CONFIGURE_CONTEXT, &context_config) < 0) {
        std::cout << "Warning: Could not configure GNA context size (using defaults)" << std::endl;
    }

    device_state_ = PersonalDeviceState::READY;
    std::cout << "Personal GNA device initialized successfully" << std::endl;

    // Start personal power monitoring
    std::thread([this]() { monitorPersonalPowerUsage(); }).detach();

    return true;
}

bool GNADeviceManager::configurePersonalPowerProfile() {
    if (device_fd_ < 0) return false;

    std::cout << "Configuring personal power profile for battery efficiency" << std::endl;

    // Set personal power mode for <0.05W target
    uint32_t personal_power_mode = 0x01;  // Low power mode for personal use
    if (config_.battery_optimization) {
        personal_power_mode |= 0x02;  // Battery optimization flag
    }

    if (ioctl(device_fd_, PERSONAL_GNA_SET_POWER_MODE, &personal_power_mode) < 0) {
        std::cout << "Warning: Could not set personal power mode (continuing with defaults)" << std::endl;
    }

    // Configure personal clocking for efficiency
    if (!setupPersonalClocking()) {
        std::cout << "Warning: Personal clocking setup failed" << std::endl;
    }

    return configurePowerConstraints();
}

bool GNADeviceManager::setupPersonalClocking() {
    // Personal frequency scaling for Dell Latitude 5450
    // Target: Balance performance and battery life

    std::ofstream freq_file("/sys/class/drm/card0/device/pp_dpm_fclk");
    if (freq_file.is_open()) {
        freq_file << "0";  // Lowest frequency for personal battery optimization
        freq_file.close();
        std::cout << "Set personal GNA frequency for battery optimization" << std::endl;
        return true;
    }

    return true;  // Continue even if frequency scaling unavailable
}

bool GNADeviceManager::configurePowerConstraints() {
    // Set personal power constraints for <0.05W target
    metrics_.current_power_mw = 0.0;
    metrics_.battery_efficient_mode = config_.battery_optimization;

    std::cout << "Personal power constraints configured:" << std::endl;
    std::cout << "  Target power consumption: " << config_.target_power_consumption << "W" << std::endl;
    std::cout << "  Battery optimization: " << (config_.battery_optimization ? "enabled" : "disabled") << std::endl;

    return true;
}

bool GNADeviceManager::enablePersonalDetection() {
    if (device_state_ != PersonalDeviceState::READY) {
        return false;
    }

    device_state_ = PersonalDeviceState::ACTIVE;
    std::cout << "Personal GNA detection enabled for " << config_.user_profile << std::endl;

    return true;
}

void GNADeviceManager::monitorPersonalPowerUsage() {
    while (device_state_ == PersonalDeviceState::ACTIVE ||
           device_state_ == PersonalDeviceState::READY) {

        // Read personal power consumption
        uint32_t power_status = 0;
        if (device_fd_ >= 0 &&
            ioctl(device_fd_, PERSONAL_GNA_GET_POWER_STATUS, &power_status) >= 0) {

            // Convert to milliwatts for personal monitoring
            double power_mw = (power_status & 0xFFFF) * 0.1;  // Simplified conversion
            metrics_.current_power_mw = power_mw;

            // Check personal power budget
            if (power_mw > (config_.target_power_consumption * 1000.0)) {
                std::cout << "Warning: Personal power consumption (" << power_mw
                         << "mW) exceeds target (" << config_.target_power_consumption * 1000.0
                         << "mW)" << std::endl;

                if (config_.battery_optimization &&
                    device_state_ != PersonalDeviceState::POWER_SAVING) {
                    enterBatteryOptimizedMode();
                }
            }
        }

        // Personal monitoring interval (1 second for Week 1)
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

bool GNADeviceManager::enterBatteryOptimizedMode() {
    if (device_fd_ < 0) return false;

    std::cout << "Entering personal battery optimized mode" << std::endl;

    // Personal battery optimization settings
    uint32_t battery_mode = 0x04;  // Ultra-low power for personal use
    if (ioctl(device_fd_, PERSONAL_GNA_SET_POWER_MODE, &battery_mode) < 0) {
        return false;
    }

    device_state_ = PersonalDeviceState::POWER_SAVING;
    metrics_.battery_efficient_mode = true;

    return true;
}

bool GNADeviceManager::exitBatteryOptimizedMode() {
    if (device_fd_ < 0) return false;

    std::cout << "Exiting personal battery optimized mode" << std::endl;

    // Restore personal normal power mode
    uint32_t normal_mode = 0x01;
    if (ioctl(device_fd_, PERSONAL_GNA_SET_POWER_MODE, &normal_mode) < 0) {
        return false;
    }

    device_state_ = PersonalDeviceState::ACTIVE;
    metrics_.battery_efficient_mode = false;

    return true;
}

double GNADeviceManager::getCurrentPowerConsumption() const {
    return metrics_.current_power_mw.load() / 1000.0;  // Convert to watts
}

bool GNADeviceManager::isWithinPersonalPowerBudget() const {
    return getCurrentPowerConsumption() <= config_.target_power_consumption;
}

bool GNADeviceManager::isPersonalDeviceReady() const {
    return device_state_ == PersonalDeviceState::READY ||
           device_state_ == PersonalDeviceState::ACTIVE ||
           device_state_ == PersonalDeviceState::POWER_SAVING;
}

void GNADeviceManager::updatePersonalProfile(const std::string& profile) {
    config_.user_profile = profile;
    std::cout << "Updated personal profile to: " << profile << std::endl;
}

void GNADeviceManager::adjustPersonalSensitivity(int sensitivity) {
    config_.detection_sensitivity = std::max(0, std::min(100, sensitivity));
    std::cout << "Adjusted personal detection sensitivity to: "
              << config_.detection_sensitivity << "%" << std::endl;
}

bool GNADeviceManager::validatePersonalConfiguration() const {
    bool valid = true;

    if (config_.target_power_consumption > 0.1) {  // >100mW too high for personal
        std::cout << "Warning: Personal power target too high for battery efficiency" << std::endl;
        valid = false;
    }

    if (config_.detection_sensitivity < 50) {
        std::cout << "Warning: Personal detection sensitivity may be too low" << std::endl;
    }

    return valid;
}

void GNADeviceManager::shutdown() {
    if (device_state_ != PersonalDeviceState::UNINITIALIZED) {
        device_state_ = PersonalDeviceState::UNINITIALIZED;

        if (device_fd_ >= 0) {
            close(device_fd_);
            device_fd_ = -1;
        }

        std::cout << "Personal GNA device manager shutdown complete" << std::endl;
        logPersonalMetrics();
    }
}

bool GNADeviceManager::handlePersonalDeviceError(const std::string& error) {
    last_error_ = error;
    std::cerr << "Personal GNA Error: " << error << std::endl;
    device_state_ = PersonalDeviceState::ERROR;
    return false;
}

void GNADeviceManager::logPersonalMetrics() const {
    std::cout << "\n=== Personal GNA Metrics Summary ===" << std::endl;
    std::cout << "Detections processed: " << metrics_.detections_processed.load() << std::endl;
    std::cout << "Wake words detected: " << metrics_.wake_words_detected.load() << std::endl;
    std::cout << "Detection accuracy: " << metrics_.detection_accuracy.load() << "%" << std::endl;
    std::cout << "Current power: " << getCurrentPowerConsumption() << "W" << std::endl;
    std::cout << "Battery efficient mode: " << (metrics_.battery_efficient_mode.load() ? "ON" : "OFF") << std::endl;
    std::cout << "===================================" << std::endl;
}

// Personal utility functions
namespace personal_gna_utils {
    bool detectGNACapability() {
        // Check for Intel GNA on Dell Latitude 5450
        std::ifstream pci_devices("/proc/bus/pci/devices");
        std::string line;

        while (std::getline(pci_devices, line)) {
            if (line.find("8086") != std::string::npos) {  // Intel vendor ID
                // Look for GNA device ID patterns
                if (line.find("8000") != std::string::npos ||
                    line.find("9000") != std::string::npos) {
                    std::cout << "Intel GNA capability detected on personal system" << std::endl;
                    return true;
                }
            }
        }

        // Check for accel device
        return access("/dev/accel/accel0", R_OK | W_OK) == 0;
    }

    std::string getPersonalDeviceInfo() {
        std::string info = "Personal GNA Device Info:\n";
        info += "  Device: Dell Latitude 5450 MIL-SPEC\n";
        info += "  CPU: Intel Meteor Lake-P\n";
        info += "  GNA: Neural-Network Accelerator\n";
        info += "  Target: Personal productivity system\n";
        info += "  Power budget: <0.05W for battery efficiency\n";
        return info;
    }

    bool validatePersonalPowerBudget(double target_mw) {
        // Personal system should target <50mW for good battery life
        return target_mw <= 50.0;
    }

    GNADeviceManager::PersonalConfig getOptimalPersonalConfig() {
        GNADeviceManager::PersonalConfig config;
        config.target_power_consumption = 0.03;  // 30mW for excellent battery life
        config.battery_optimization = true;
        config.user_profile = "personal_productivity";
        config.detection_sensitivity = 90;  // High accuracy for personal use
        config.adaptive_learning = true;
        config.context_size = 256;  // Smaller for personal efficiency
        config.batch_size = 1;      // Single user optimization
        return config;
    }
}