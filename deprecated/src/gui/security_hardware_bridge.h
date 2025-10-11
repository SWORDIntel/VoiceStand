#pragma once

#include "security_interface.h"
#include <string>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>

namespace vtt {

/**
 * Bridge between C++ GUI and Rust security modules
 * Provides hardware detection and security capability assessment
 * Integrates with existing intel_me_security.rs and tpm_crypto_acceleration.rs
 */
class SecurityHardwareBridge {
public:
    SecurityHardwareBridge();
    ~SecurityHardwareBridge();

    // Initialization
    bool initialize();
    void cleanup();

    // Hardware detection interface
    HardwareCapabilities detect_hardware_capabilities();
    SecurityMetrics get_current_security_metrics();

    // Configuration interface
    bool configure_security_level(SecurityLevel target_level);
    bool enable_enterprise_features(bool enable);
    bool set_encryption_algorithm(const std::string& algorithm);

    // Real-time monitoring
    void start_monitoring(std::function<void(const SecurityMetrics&)> callback);
    void stop_monitoring();

    // Enterprise features
    bool validate_compliance_status();
    std::vector<std::string> get_supported_algorithms();
    bool perform_hardware_attestation();

private:
    // Rust FFI interface declarations
    struct RustSecurityContext;
    RustSecurityContext* rust_context_;

    // Hardware detection methods
    bool detect_tpm_availability();
    bool detect_intel_me_availability();
    bool detect_npu_availability();
    bool detect_gna_availability();
    bool check_fips_compliance();
    bool check_enterprise_capabilities();

    // Performance monitoring
    SecurityMetrics collect_performance_metrics();
    void monitoring_thread_func();

    // Thread management
    std::unique_ptr<std::thread> monitoring_thread_;
    std::atomic<bool> monitoring_active_;
    std::function<void(const SecurityMetrics&)> metrics_callback_;

    // FFI function pointers (loaded dynamically)
    struct RustFunctions {
        // TPM functions
        bool (*tpm_initialize)();
        bool (*tpm_is_available)();
        uint64_t (*tpm_get_operations_count)();
        uint32_t (*tpm_get_latency_ms)();
        bool (*tpm_perform_operation)(const char* operation);

        // Intel ME functions
        bool (*me_initialize)();
        bool (*me_is_available)();
        bool (*me_get_security_status)();
        uint32_t (*me_get_performance_metrics)();

        // NPU functions
        bool (*npu_initialize)();
        bool (*npu_is_available)();
        float (*npu_get_utilization)();

        // GNA functions
        bool (*gna_initialize)();
        bool (*gna_is_available)();
        bool (*gna_is_active)();

        // Security operations
        bool (*security_get_compliance_status)();
        const char* (*security_get_active_encryption)();
        bool (*security_validate_attestation)();
        uint64_t (*security_get_crypto_ops_per_sec)();

        // Configuration functions
        bool (*config_set_security_level)(int level);
        bool (*config_enable_enterprise)(bool enable);
        bool (*config_set_encryption_algorithm)(const char* algorithm);
    } rust_functions_;

    // Library handle for dynamic loading
    void* rust_library_handle_;

    // Internal state
    HardwareCapabilities last_capabilities_;
    SecurityMetrics last_metrics_;
    std::atomic<bool> initialized_;

    // Helper methods
    bool load_rust_library();
    void unload_rust_library();
    SecurityLevel determine_security_level(const HardwareCapabilities& caps);
    std::string get_library_path();
};

/**
 * C-style interface for integration with existing C/C++ code
 * These functions wrap the SecurityHardwareBridge class
 */
extern "C" {
    // Hardware detection
    int security_bridge_detect_tpm();
    int security_bridge_detect_intel_me();
    int security_bridge_detect_npu();
    int security_bridge_detect_gna();

    // Security metrics
    typedef struct {
        uint64_t operations_completed;
        uint64_t operations_failed;
        uint32_t current_latency_ms;
        uint32_t avg_latency_ms;
        float cpu_usage_percent;
        char active_encryption[64];
        char compliance_status[64];
        int attestation_valid;
    } SecurityMetricsC;

    int security_bridge_get_metrics(SecurityMetricsC* metrics);

    // Configuration
    int security_bridge_set_security_level(int level);
    int security_bridge_enable_enterprise(int enable);
    int security_bridge_set_encryption(const char* algorithm);

    // Initialization
    int security_bridge_initialize();
    void security_bridge_cleanup();
}

} // namespace vtt