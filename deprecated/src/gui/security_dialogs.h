#pragma once

#include "security_interface.h"
#include <gtk/gtk.h>
#include <vector>
#include <functional>

namespace vtt {

/**
 * Comprehensive Security Dialogs for VoiceStand
 * Provides detailed configuration and monitoring interfaces
 */

class SecurityDialogs {
public:
    SecurityDialogs(SecurityInterface* security_interface);
    ~SecurityDialogs();

    // Dialog creation methods
    void show_security_settings_dialog(GtkWidget* parent);
    void show_enterprise_admin_dialog(GtkWidget* parent);
    void show_compliance_dashboard_dialog(GtkWidget* parent);
    void show_audit_log_viewer_dialog(GtkWidget* parent);
    void show_hardware_details_dialog(GtkWidget* parent);

    // Dialog update methods
    void update_security_settings_from_capabilities(const HardwareCapabilities& caps);
    void update_enterprise_settings();
    void refresh_audit_log_display();

private:
    // Dialog creation helpers
    GtkWidget* create_security_settings_content();
    GtkWidget* create_enterprise_admin_content();
    GtkWidget* create_compliance_dashboard_content();
    GtkWidget* create_audit_log_content();
    GtkWidget* create_hardware_details_content();

    // Security settings components
    GtkWidget* create_security_level_selection();
    GtkWidget* create_encryption_algorithm_selection();
    GtkWidget* create_key_management_settings();
    GtkWidget* create_hardware_acceleration_settings();

    // Enterprise components
    GtkWidget* create_policy_management_section();
    GtkWidget* create_certificate_management_section();
    GtkWidget* create_compliance_monitoring_section();
    GtkWidget* create_audit_configuration_section();

    // Hardware details components
    GtkWidget* create_tpm_details_section();
    GtkWidget* create_intel_me_details_section();
    GtkWidget* create_npu_details_section();
    GtkWidget* create_performance_metrics_section();

    // Event handlers
    static void on_security_level_changed(GtkComboBox* combo, gpointer user_data);
    static void on_encryption_algorithm_changed(GtkComboBox* combo, gpointer user_data);
    static void on_key_rotation_toggled(GtkToggleButton* button, gpointer user_data);
    static void on_hardware_acceleration_toggled(GtkToggleButton* button, gpointer user_data);
    static void on_enterprise_features_toggled(GtkToggleButton* button, gpointer user_data);
    static void on_policy_import_clicked(GtkButton* button, gpointer user_data);
    static void on_policy_export_clicked(GtkButton* button, gpointer user_data);
    static void on_audit_export_clicked(GtkButton* button, gpointer user_data);
    static void on_audit_filter_changed(GtkEntry* entry, gpointer user_data);
    static void on_refresh_hardware_clicked(GtkButton* button, gpointer user_data);
    static void on_test_attestation_clicked(GtkButton* button, gpointer user_data);

    // Utility methods
    void populate_security_level_combo(GtkComboBox* combo);
    void populate_encryption_algorithm_combo(GtkComboBox* combo);
    void populate_audit_log_list(GtkListBox* list_box);
    void update_hardware_details_display();
    void show_file_chooser_dialog(const std::string& title, GtkFileChooserAction action,
                                 std::function<void(const std::string&)> callback);

    // Data formatting
    std::string format_hardware_info(const HardwareCapabilities& caps);
    std::string format_performance_metrics(const SecurityMetrics& metrics);
    std::string format_audit_entry(const AuditLogEntry& entry);
    std::string format_timestamp(const std::chrono::system_clock::time_point& time);

    // Member variables
    SecurityInterface* security_interface_;

    // Dialog widgets
    GtkWidget* security_settings_dialog_;
    GtkWidget* enterprise_admin_dialog_;
    GtkWidget* compliance_dashboard_dialog_;
    GtkWidget* audit_log_viewer_dialog_;
    GtkWidget* hardware_details_dialog_;

    // Security settings widgets
    GtkWidget* security_level_combo_;
    GtkWidget* encryption_algorithm_combo_;
    GtkWidget* key_rotation_toggle_;
    GtkWidget* hardware_acceleration_toggle_;
    GtkWidget* enterprise_features_toggle_;

    // Enterprise widgets
    GtkWidget* policy_status_label_;
    GtkWidget* certificate_status_label_;
    GtkWidget* compliance_progress_bar_;
    GtkWidget* audit_retention_spin_;

    // Hardware details widgets
    GtkWidget* tpm_version_label_;
    GtkWidget* tpm_algorithms_label_;
    GtkWidget* me_version_label_;
    GtkWidget* me_status_label_;
    GtkWidget* npu_model_label_;
    GtkWidget* npu_utilization_label_;
    GtkWidget* performance_chart_;

    // Audit log widgets
    GtkWidget* audit_list_box_;
    GtkWidget* audit_filter_entry_;
    GtkWidget* audit_count_label_;

    // Current state
    HardwareCapabilities current_capabilities_;
    SecurityMetrics current_metrics_;
    std::vector<AuditLogEntry> filtered_audit_logs_;
};

/**
 * Security Configuration Assistant
 * Provides guided setup for security features
 */
class SecurityConfigurationAssistant {
public:
    SecurityConfigurationAssistant(SecurityInterface* security_interface);
    ~SecurityConfigurationAssistant();

    void show_setup_wizard(GtkWidget* parent);

private:
    // Wizard pages
    GtkWidget* create_welcome_page();
    GtkWidget* create_hardware_detection_page();
    GtkWidget* create_security_level_selection_page();
    GtkWidget* create_enterprise_configuration_page();
    GtkWidget* create_completion_page();

    // Page navigation
    void next_page();
    void previous_page();
    void finish_setup();

    // Event handlers
    static void on_next_clicked(GtkButton* button, gpointer user_data);
    static void on_previous_clicked(GtkButton* button, gpointer user_data);
    static void on_finish_clicked(GtkButton* button, gpointer user_data);
    static void on_detect_hardware_clicked(GtkButton* button, gpointer user_data);

    SecurityInterface* security_interface_;
    GtkWidget* assistant_dialog_;
    GtkWidget* assistant_stack_;
    int current_page_;
    std::vector<GtkWidget*> pages_;
};

} // namespace vtt