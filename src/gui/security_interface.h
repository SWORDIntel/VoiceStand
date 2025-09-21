#pragma once

#include <gtk/gtk.h>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <json/json.h>
#include <chrono>

namespace vtt {

/**
 * Adaptive Security Interface for VoiceStand
 * Provides conditional security features based on hardware detection
 * Integrates seamlessly with existing GTK4 main window
 */

// Security levels based on hardware availability
enum class SecurityLevel {
    MINIMAL = 1,    // Software-only (no security UI elements)
    BASIC = 2,      // Software crypto + NPU (minimal security UI)
    GOOD = 3,       // TPM + NPU (security basics UI)
    OPTIMAL = 4     // TPM + ME + NPU + GNA (full security UI)
};

// Hardware component status
struct HardwareCapabilities {
    bool tpm_available = false;
    bool intel_me_available = false;
    bool npu_available = false;
    bool gna_available = false;
    bool fips_compliant = false;
    bool enterprise_features = false;
    SecurityLevel current_level = SecurityLevel::MINIMAL;
    std::vector<std::string> supported_algorithms;

    // Performance metrics
    uint64_t crypto_operations_per_sec = 0;
    uint32_t attestation_latency_ms = 0;
    bool key_rotation_active = false;
    std::chrono::steady_clock::time_point last_audit_time;
};

// Security metrics for real-time display
struct SecurityMetrics {
    uint64_t operations_completed = 0;
    uint64_t operations_failed = 0;
    uint32_t current_latency_ms = 0;
    uint32_t avg_latency_ms = 0;
    float cpu_usage_percent = 0.0f;
    std::string active_encryption = "None";
    std::string compliance_status = "Not Configured";
    bool attestation_valid = false;
    std::chrono::steady_clock::time_point last_key_rotation;
};

// Audit log entry for enterprise features
struct AuditLogEntry {
    std::chrono::system_clock::time_point timestamp;
    std::string event_type;
    std::string user_id;
    std::string resource;
    std::string action;
    std::string result;
    std::string details;
    SecurityLevel security_level;
};

class SecurityInterface {
public:
    SecurityInterface();
    ~SecurityInterface();

    // Initialization and integration
    bool initialize(GtkWidget* parent_window);
    void integrate_with_main_window(GtkWidget* main_box, GtkWidget* status_bar);
    void cleanup();

    // Hardware detection and adaptation
    void update_hardware_capabilities(const HardwareCapabilities& caps);
    void refresh_hardware_detection();
    SecurityLevel get_current_security_level() const { return capabilities_.current_level; }

    // UI state management
    void show_security_panel(bool show = true);
    void update_security_status();
    void update_real_time_metrics(const SecurityMetrics& metrics);
    void adapt_ui_to_capabilities();

    // Security configuration
    void show_security_settings_dialog();
    void show_enterprise_admin_panel();
    void show_compliance_dashboard();
    void show_audit_log_viewer();

    // Enterprise features
    void add_audit_log_entry(const AuditLogEntry& entry);
    void export_audit_logs(const std::string& filepath);
    void import_security_policy(const std::string& filepath);
    void export_security_config(const std::string& filepath);

    // Configuration callbacks
    void set_security_config_callback(std::function<void(const Json::Value&)> callback) {
        security_config_callback_ = callback;
    }

    void set_hardware_detection_callback(std::function<HardwareCapabilities()> callback) {
        hardware_detection_callback_ = callback;
    }

    void set_metrics_callback(std::function<SecurityMetrics()> callback) {
        metrics_callback_ = callback;
    }

    // Utility functions
    std::string get_security_level_string(SecurityLevel level) const;
    std::string get_security_level_color(SecurityLevel level) const;
    bool is_enterprise_mode() const { return capabilities_.enterprise_features; }

private:
    // UI creation methods
    void create_security_status_panel();
    void create_security_dashboard();
    void create_enterprise_panel();
    void create_hardware_status_indicators();
    void create_metrics_display();
    void create_audit_log_view();

    // Dialog creation
    GtkWidget* create_security_settings_dialog();
    GtkWidget* create_enterprise_admin_dialog();
    GtkWidget* create_compliance_dashboard_dialog();
    GtkWidget* create_audit_viewer_dialog();

    // UI update methods
    void update_hardware_indicators();
    void update_security_level_display();
    void update_encryption_status();
    void update_compliance_display();
    void update_performance_metrics();
    void refresh_audit_log_display();

    // Event handlers
    static void on_security_settings_clicked(GtkButton* button, gpointer user_data);
    static void on_enterprise_admin_clicked(GtkButton* button, gpointer user_data);
    static void on_compliance_dashboard_clicked(GtkButton* button, gpointer user_data);
    static void on_audit_log_clicked(GtkButton* button, gpointer user_data);
    static void on_refresh_hardware_clicked(GtkButton* button, gpointer user_data);
    static void on_export_audit_clicked(GtkButton* button, gpointer user_data);
    static void on_security_level_changed(GtkComboBox* combo, gpointer user_data);
    static void on_encryption_algorithm_changed(GtkComboBox* combo, gpointer user_data);
    static void on_key_rotation_toggled(GtkToggleButton* button, gpointer user_data);
    static gboolean on_metrics_timer(gpointer user_data);

    // Configuration management
    void load_security_config();
    void save_security_config();
    Json::Value get_current_security_config() const;
    void apply_security_config(const Json::Value& config);

    // Audit log management
    void load_audit_logs();
    void save_audit_logs();
    std::vector<AuditLogEntry> filter_audit_logs(const std::string& filter) const;

    // Member variables
    GtkWidget* parent_window_;
    GtkWidget* main_box_;
    GtkWidget* status_bar_;

    // Security panel widgets (conditional visibility)
    GtkWidget* security_panel_;
    GtkWidget* security_frame_;
    GtkWidget* security_expander_;
    bool security_panel_visible_;

    // Hardware status indicators
    GtkWidget* tpm_indicator_;
    GtkWidget* me_indicator_;
    GtkWidget* npu_indicator_;
    GtkWidget* gna_indicator_;
    GtkWidget* security_level_label_;
    GtkWidget* security_level_icon_;

    // Security dashboard widgets
    GtkWidget* encryption_status_label_;
    GtkWidget* operations_counter_label_;
    GtkWidget* latency_label_;
    GtkWidget* cpu_usage_label_;
    GtkWidget* compliance_status_label_;
    GtkWidget* last_audit_label_;

    // Enterprise widgets (conditional)
    GtkWidget* enterprise_panel_;
    GtkWidget* audit_log_view_;
    GtkWidget* compliance_dashboard_;
    GtkWidget* policy_status_label_;

    // Metrics display
    GtkWidget* metrics_grid_;
    GtkWidget* performance_chart_;
    guint metrics_timer_id_;

    // Data storage
    HardwareCapabilities capabilities_;
    SecurityMetrics current_metrics_;
    std::vector<AuditLogEntry> audit_logs_;
    Json::Value security_config_;

    // Thread safety
    mutable std::mutex capabilities_mutex_;
    mutable std::mutex metrics_mutex_;
    mutable std::mutex audit_mutex_;

    // Callbacks
    std::function<void(const Json::Value&)> security_config_callback_;
    std::function<HardwareCapabilities()> hardware_detection_callback_;
    std::function<SecurityMetrics()> metrics_callback_;

    // UI state
    bool initialized_;
    std::atomic<bool> ui_update_in_progress_;

    // Enterprise configuration
    struct EnterpriseConfig {
        bool policy_enforcement_enabled = false;
        std::string policy_server_url;
        std::string certificate_authority;
        uint32_t audit_retention_days = 90;
        bool automatic_key_rotation = false;
        uint32_t key_rotation_interval_hours = 24;
        std::vector<std::string> compliance_standards;
    } enterprise_config_;

    static SecurityInterface* instance_;
};

} // namespace vtt