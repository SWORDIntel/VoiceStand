#include "security_hardware_bridge.h"
#include <iostream>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <chrono>
#include <cstring>

namespace vtt {

static SecurityHardwareBridge* bridge_instance = nullptr;

SecurityHardwareBridge::SecurityHardwareBridge()
    : rust_context_(nullptr)
    , monitoring_thread_(nullptr)
    , monitoring_active_(false)
    , rust_library_handle_(nullptr)
    , initialized_(false) {
    // Initialize function pointers to nullptr
    memset(&rust_functions_, 0, sizeof(rust_functions_));
}

SecurityHardwareBridge::~SecurityHardwareBridge() {
    cleanup();
}

bool SecurityHardwareBridge::initialize() {
    if (initialized_.load()) {
        return true;
    }

    // Load Rust library
    if (!load_rust_library()) {
        std::cerr << "Failed to load Rust security library" << std::endl;
        return false;
    }

    // Detect initial hardware capabilities
    last_capabilities_ = detect_hardware_capabilities();

    initialized_ = true;
    bridge_instance = this;

    std::cout << "SecurityHardwareBridge initialized with security level: "
              << static_cast<int>(last_capabilities_.current_level) << std::endl;

    return true;
}

void SecurityHardwareBridge::cleanup() {
    if (!initialized_.load()) {
        return;
    }

    stop_monitoring();
    unload_rust_library();

    initialized_ = false;
    bridge_instance = nullptr;
}

HardwareCapabilities SecurityHardwareBridge::detect_hardware_capabilities() {
    HardwareCapabilities caps;

    // Detect each hardware component
    caps.tpm_available = detect_tpm_availability();
    caps.intel_me_available = detect_intel_me_availability();
    caps.npu_available = detect_npu_availability();
    caps.gna_available = detect_gna_availability();
    caps.fips_compliant = check_fips_compliance();
    caps.enterprise_features = check_enterprise_capabilities();

    // Determine security level based on available hardware
    caps.current_level = determine_security_level(caps);

    // Get supported algorithms
    caps.supported_algorithms = get_supported_algorithms();

    // Update performance metrics
    if (caps.current_level > SecurityLevel::MINIMAL) {
        auto metrics = collect_performance_metrics();
        caps.crypto_operations_per_sec = metrics.operations_completed;
        caps.attestation_latency_ms = metrics.current_latency_ms;
    }

    last_capabilities_ = caps;
    return caps;
}

SecurityMetrics SecurityHardwareBridge::get_current_security_metrics() {
    return collect_performance_metrics();
}

SecurityMetrics SecurityHardwareBridge::collect_performance_metrics() {
    SecurityMetrics metrics;

    if (!initialized_.load() || !rust_library_handle_) {
        return metrics;
    }

    try {
        // Collect metrics from Rust modules
        if (rust_functions_.tpm_get_operations_count) {
            metrics.operations_completed = rust_functions_.tpm_get_operations_count();
        }

        if (rust_functions_.tpm_get_latency_ms) {
            metrics.current_latency_ms = rust_functions_.tpm_get_latency_ms();
        }

        if (rust_functions_.security_get_active_encryption) {
            const char* encryption = rust_functions_.security_get_active_encryption();
            if (encryption) {
                metrics.active_encryption = encryption;
            }
        }

        if (rust_functions_.security_get_crypto_ops_per_sec) {
            uint64_t ops_per_sec = rust_functions_.security_get_crypto_ops_per_sec();
            // Convert to operations per second for display
            metrics.operations_completed = ops_per_sec;
        }

        if (rust_functions_.security_validate_attestation) {
            metrics.attestation_valid = rust_functions_.security_validate_attestation();
        }

        if (rust_functions_.security_get_compliance_status) {
            bool compliant = rust_functions_.security_get_compliance_status();
            metrics.compliance_status = compliant ? "FIPS 140-2 Compliant" : "Not Compliant";
        }

        // Calculate average latency (simple moving average)
        static uint32_t latency_samples[10] = {0};
        static int sample_index = 0;

        latency_samples[sample_index] = metrics.current_latency_ms;
        sample_index = (sample_index + 1) % 10;

        uint32_t sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += latency_samples[i];
        }
        metrics.avg_latency_ms = sum / 10;

        // Estimate CPU usage (placeholder - would need system monitoring)
        metrics.cpu_usage_percent = std::min(5.0f, static_cast<float>(metrics.current_latency_ms) / 10.0f);

    } catch (const std::exception& e) {
        std::cerr << "Error collecting security metrics: " << e.what() << std::endl;
    }

    last_metrics_ = metrics;
    return metrics;
}

bool SecurityHardwareBridge::detect_tpm_availability() {
    if (!rust_functions_.tpm_is_available) {
        // Fallback: check for TPM device files
        return std::filesystem::exists("/dev/tpm0") || std::filesystem::exists("/dev/tpmrm0");
    }

    return rust_functions_.tpm_is_available();
}

bool SecurityHardwareBridge::detect_intel_me_availability() {
    if (!rust_functions_.me_is_available) {
        // Fallback: check for Intel ME interface
        return std::filesystem::exists("/dev/mei0") || std::filesystem::exists("/dev/mei");
    }

    return rust_functions_.me_is_available();
}

bool SecurityHardwareBridge::detect_npu_availability() {
    if (!rust_functions_.npu_is_available) {
        // Fallback: check for Intel NPU device
        return std::filesystem::exists("/dev/accel/accel0") ||
               std::filesystem::exists("/sys/class/drm/renderD128");
    }

    return rust_functions_.npu_is_available();
}

bool SecurityHardwareBridge::detect_gna_availability() {
    if (!rust_functions_.gna_is_available) {
        // Fallback: check for GNA device or driver
        return std::filesystem::exists("/dev/gna0") ||
               std::filesystem::exists("/sys/module/intel_gna");
    }

    return rust_functions_.gna_is_available();
}

bool SecurityHardwareBridge::check_fips_compliance() {
    if (!rust_functions_.security_get_compliance_status) {
        // Fallback: basic FIPS check
        return std::filesystem::exists("/proc/sys/crypto/fips_enabled");
    }

    return rust_functions_.security_get_compliance_status();
}

bool SecurityHardwareBridge::check_enterprise_capabilities() {
    // Enterprise features are available if we have TPM + policy management
    return detect_tpm_availability() && detect_intel_me_availability();
}

SecurityLevel SecurityHardwareBridge::determine_security_level(const HardwareCapabilities& caps) {
    if (caps.tpm_available && caps.intel_me_available && caps.npu_available && caps.gna_available) {
        return SecurityLevel::OPTIMAL;
    } else if (caps.tpm_available && caps.npu_available) {
        return SecurityLevel::GOOD;
    } else if (caps.npu_available) {
        return SecurityLevel::BASIC;
    } else {
        return SecurityLevel::MINIMAL;
    }
}

std::vector<std::string> SecurityHardwareBridge::get_supported_algorithms() {
    std::vector<std::string> algorithms;

    // Add algorithms based on hardware capabilities
    if (last_capabilities_.tpm_available) {
        algorithms.push_back("AES-256-GCM");
        algorithms.push_back("RSA-2048");
        algorithms.push_back("ECC-P256");
        algorithms.push_back("SHA-256");
    }

    if (last_capabilities_.intel_me_available) {
        algorithms.push_back("AES-256-XTS");
        algorithms.push_back("ChaCha20-Poly1305");
    }

    if (last_capabilities_.fips_compliant) {
        algorithms.push_back("AES-256-CBC (FIPS)");
        algorithms.push_back("SHA-3-256");
    }

    if (algorithms.empty()) {
        algorithms.push_back("Software AES");
        algorithms.push_back("Software ChaCha20");
    }

    return algorithms;
}

bool SecurityHardwareBridge::configure_security_level(SecurityLevel target_level) {
    if (!rust_functions_.config_set_security_level) {
        std::cout << "Setting security level to: " << static_cast<int>(target_level)
                  << " (software mode)" << std::endl;
        return true;
    }

    return rust_functions_.config_set_security_level(static_cast<int>(target_level));
}

bool SecurityHardwareBridge::enable_enterprise_features(bool enable) {
    if (!rust_functions_.config_enable_enterprise) {
        std::cout << "Enterprise features " << (enable ? "enabled" : "disabled")
                  << " (software mode)" << std::endl;
        return true;
    }

    return rust_functions_.config_enable_enterprise(enable);
}

bool SecurityHardwareBridge::set_encryption_algorithm(const std::string& algorithm) {
    if (!rust_functions_.config_set_encryption_algorithm) {
        std::cout << "Setting encryption algorithm to: " << algorithm
                  << " (software mode)" << std::endl;
        return true;
    }

    return rust_functions_.config_set_encryption_algorithm(algorithm.c_str());
}

void SecurityHardwareBridge::start_monitoring(std::function<void(const SecurityMetrics&)> callback) {
    if (monitoring_active_.load()) {
        return;
    }

    metrics_callback_ = callback;
    monitoring_active_ = true;

    monitoring_thread_ = std::make_unique<std::thread>(&SecurityHardwareBridge::monitoring_thread_func, this);
}

void SecurityHardwareBridge::stop_monitoring() {
    if (!monitoring_active_.load()) {
        return;
    }

    monitoring_active_ = false;

    if (monitoring_thread_ && monitoring_thread_->joinable()) {
        monitoring_thread_->join();
    }

    monitoring_thread_.reset();
}

void SecurityHardwareBridge::monitoring_thread_func() {
    while (monitoring_active_.load()) {
        try {
            if (metrics_callback_) {
                auto metrics = collect_performance_metrics();
                metrics_callback_(metrics);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        } catch (const std::exception& e) {
            std::cerr << "Error in monitoring thread: " << e.what() << std::endl;
        }
    }
}

bool SecurityHardwareBridge::load_rust_library() {
    std::string lib_path = get_library_path();

    rust_library_handle_ = dlopen(lib_path.c_str(), RTLD_LAZY);
    if (!rust_library_handle_) {
        std::cerr << "Cannot load Rust library: " << dlerror() << std::endl;
        std::cout << "Running in software-only mode" << std::endl;
        return true; // Still allow software-only operation
    }

    // Load function pointers
    rust_functions_.tpm_initialize = (bool(*)())dlsym(rust_library_handle_, "tpm_initialize");
    rust_functions_.tpm_is_available = (bool(*)())dlsym(rust_library_handle_, "tpm_is_available");
    rust_functions_.tpm_get_operations_count = (uint64_t(*)())dlsym(rust_library_handle_, "tpm_get_operations_count");
    rust_functions_.tpm_get_latency_ms = (uint32_t(*)())dlsym(rust_library_handle_, "tpm_get_latency_ms");

    rust_functions_.me_initialize = (bool(*)())dlsym(rust_library_handle_, "me_initialize");
    rust_functions_.me_is_available = (bool(*)())dlsym(rust_library_handle_, "me_is_available");

    rust_functions_.npu_initialize = (bool(*)())dlsym(rust_library_handle_, "npu_initialize");
    rust_functions_.npu_is_available = (bool(*)())dlsym(rust_library_handle_, "npu_is_available");

    rust_functions_.gna_initialize = (bool(*)())dlsym(rust_library_handle_, "gna_initialize");
    rust_functions_.gna_is_available = (bool(*)())dlsym(rust_library_handle_, "gna_is_available");

    rust_functions_.security_get_compliance_status = (bool(*)())dlsym(rust_library_handle_, "security_get_compliance_status");
    rust_functions_.security_get_active_encryption = (const char*(*)())dlsym(rust_library_handle_, "security_get_active_encryption");
    rust_functions_.security_validate_attestation = (bool(*)())dlsym(rust_library_handle_, "security_validate_attestation");
    rust_functions_.security_get_crypto_ops_per_sec = (uint64_t(*)())dlsym(rust_library_handle_, "security_get_crypto_ops_per_sec");

    rust_functions_.config_set_security_level = (bool(*)(int))dlsym(rust_library_handle_, "config_set_security_level");
    rust_functions_.config_enable_enterprise = (bool(*)(bool))dlsym(rust_library_handle_, "config_enable_enterprise");
    rust_functions_.config_set_encryption_algorithm = (bool(*)(const char*))dlsym(rust_library_handle_, "config_set_encryption_algorithm");

    // Initialize Rust modules if functions are available
    if (rust_functions_.tpm_initialize) {
        rust_functions_.tpm_initialize();
    }
    if (rust_functions_.me_initialize) {
        rust_functions_.me_initialize();
    }
    if (rust_functions_.npu_initialize) {
        rust_functions_.npu_initialize();
    }
    if (rust_functions_.gna_initialize) {
        rust_functions_.gna_initialize();
    }

    std::cout << "Rust security library loaded successfully" << std::endl;
    return true;
}

void SecurityHardwareBridge::unload_rust_library() {
    if (rust_library_handle_) {
        dlclose(rust_library_handle_);
        rust_library_handle_ = nullptr;
    }

    memset(&rust_functions_, 0, sizeof(rust_functions_));
}

std::string SecurityHardwareBridge::get_library_path() {
    // Look for the Rust library in standard locations
    std::vector<std::string> search_paths = {
        "./target/release/libvoicestand_security.so",
        "./libvoicestand_security.so",
        "/usr/local/lib/libvoicestand_security.so",
        "/usr/lib/libvoicestand_security.so"
    };

    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }

    return "libvoicestand_security.so"; // Let dlopen search system paths
}

bool SecurityHardwareBridge::validate_compliance_status() {
    if (!rust_functions_.security_get_compliance_status) {
        return false;
    }

    return rust_functions_.security_get_compliance_status();
}

bool SecurityHardwareBridge::perform_hardware_attestation() {
    if (!rust_functions_.security_validate_attestation) {
        return false;
    }

    return rust_functions_.security_validate_attestation();
}

} // namespace vtt

// C interface implementation
extern "C" {
    int security_bridge_detect_tpm() {
        if (vtt::bridge_instance) {
            return vtt::bridge_instance->detect_tpm_availability() ? 1 : 0;
        }
        return 0;
    }

    int security_bridge_detect_intel_me() {
        if (vtt::bridge_instance) {
            return vtt::bridge_instance->detect_intel_me_availability() ? 1 : 0;
        }
        return 0;
    }

    int security_bridge_detect_npu() {
        if (vtt::bridge_instance) {
            return vtt::bridge_instance->detect_npu_availability() ? 1 : 0;
        }
        return 0;
    }

    int security_bridge_detect_gna() {
        if (vtt::bridge_instance) {
            return vtt::bridge_instance->detect_gna_availability() ? 1 : 0;
        }
        return 0;
    }

    int security_bridge_get_metrics(vtt::SecurityMetricsC* metrics) {
        if (!vtt::bridge_instance || !metrics) {
            return 0;
        }

        auto cpp_metrics = vtt::bridge_instance->get_current_security_metrics();

        metrics->operations_completed = cpp_metrics.operations_completed;
        metrics->operations_failed = cpp_metrics.operations_failed;
        metrics->current_latency_ms = cpp_metrics.current_latency_ms;
        metrics->avg_latency_ms = cpp_metrics.avg_latency_ms;
        metrics->cpu_usage_percent = cpp_metrics.cpu_usage_percent;
        metrics->attestation_valid = cpp_metrics.attestation_valid ? 1 : 0;

        strncpy(metrics->active_encryption, cpp_metrics.active_encryption.c_str(), 63);
        metrics->active_encryption[63] = '\0';

        strncpy(metrics->compliance_status, cpp_metrics.compliance_status.c_str(), 63);
        metrics->compliance_status[63] = '\0';

        return 1;
    }

    int security_bridge_set_security_level(int level) {
        if (vtt::bridge_instance) {
            return vtt::bridge_instance->configure_security_level(
                static_cast<vtt::SecurityLevel>(level)) ? 1 : 0;
        }
        return 0;
    }

    int security_bridge_enable_enterprise(int enable) {
        if (vtt::bridge_instance) {
            return vtt::bridge_instance->enable_enterprise_features(enable != 0) ? 1 : 0;
        }
        return 0;
    }

    int security_bridge_set_encryption(const char* algorithm) {
        if (vtt::bridge_instance && algorithm) {
            return vtt::bridge_instance->set_encryption_algorithm(algorithm) ? 1 : 0;
        }
        return 0;
    }

    int security_bridge_initialize() {
        if (!vtt::bridge_instance) {
            vtt::bridge_instance = new vtt::SecurityHardwareBridge();
        }
        return vtt::bridge_instance->initialize() ? 1 : 0;
    }

    void security_bridge_cleanup() {
        if (vtt::bridge_instance) {
            delete vtt::bridge_instance;
            vtt::bridge_instance = nullptr;
        }
    }
}