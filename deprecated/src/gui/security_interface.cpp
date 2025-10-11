#include "security_interface.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace vtt {

SecurityInterface* SecurityInterface::instance_ = nullptr;

SecurityInterface::SecurityInterface()
    : parent_window_(nullptr)
    , main_box_(nullptr)
    , status_bar_(nullptr)
    , security_panel_(nullptr)
    , security_frame_(nullptr)
    , security_expander_(nullptr)
    , security_panel_visible_(false)
    , tpm_indicator_(nullptr)
    , me_indicator_(nullptr)
    , npu_indicator_(nullptr)
    , gna_indicator_(nullptr)
    , security_level_label_(nullptr)
    , security_level_icon_(nullptr)
    , encryption_status_label_(nullptr)
    , operations_counter_label_(nullptr)
    , latency_label_(nullptr)
    , cpu_usage_label_(nullptr)
    , compliance_status_label_(nullptr)
    , last_audit_label_(nullptr)
    , enterprise_panel_(nullptr)
    , audit_log_view_(nullptr)
    , compliance_dashboard_(nullptr)
    , policy_status_label_(nullptr)
    , metrics_grid_(nullptr)
    , performance_chart_(nullptr)
    , metrics_timer_id_(0)
    , initialized_(false)
    , ui_update_in_progress_(false) {
    instance_ = this;

    // Initialize capabilities with defaults
    capabilities_.current_level = SecurityLevel::MINIMAL;

    // Initialize enterprise config
    enterprise_config_.audit_retention_days = 90;
    enterprise_config_.key_rotation_interval_hours = 24;
}

SecurityInterface::~SecurityInterface() {
    cleanup();
    instance_ = nullptr;
}

bool SecurityInterface::initialize(GtkWidget* parent_window) {
    if (initialized_) {
        return true;
    }

    parent_window_ = parent_window;

    // Load existing configuration
    load_security_config();
    load_audit_logs();

    initialized_ = true;
    return true;
}

void SecurityInterface::integrate_with_main_window(GtkWidget* main_box, GtkWidget* status_bar) {
    if (!initialized_) {
        std::cerr << "SecurityInterface not initialized" << std::endl;
        return;
    }

    main_box_ = main_box;
    status_bar_ = status_bar;

    // Create security status panel (initially hidden)
    create_security_status_panel();

    // Create hardware status indicators for status bar
    create_hardware_status_indicators();

    // Adapt UI based on current capabilities
    adapt_ui_to_capabilities();

    // Start metrics timer if we have hardware capabilities
    if (capabilities_.current_level > SecurityLevel::MINIMAL) {
        metrics_timer_id_ = g_timeout_add(1000, on_metrics_timer, this);
    }
}

void SecurityInterface::create_security_status_panel() {
    // Create expandable security panel
    security_expander_ = gtk_expander_new("Security Status");
    gtk_expander_set_expanded(GTK_EXPANDER(security_expander_), false);

    // Create main security frame
    security_frame_ = gtk_frame_new("Hardware Security");
    gtk_expander_set_child(GTK_EXPANDER(security_expander_), security_frame_);

    // Create main security panel grid
    security_panel_ = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(security_panel_), 8);
    gtk_grid_set_column_spacing(GTK_GRID(security_panel_), 12);
    gtk_widget_set_margin_start(security_panel_, 12);
    gtk_widget_set_margin_end(security_panel_, 12);
    gtk_widget_set_margin_top(security_panel_, 8);
    gtk_widget_set_margin_bottom(security_panel_, 8);
    gtk_frame_set_child(GTK_FRAME(security_frame_), security_panel_);

    // Security level display
    auto* level_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    security_level_icon_ = gtk_image_new_from_icon_name("security-low");
    security_level_label_ = gtk_label_new("Security Level: Minimal");
    gtk_box_append(GTK_BOX(level_box), security_level_icon_);
    gtk_box_append(GTK_BOX(level_box), security_level_label_);
    gtk_grid_attach(GTK_GRID(security_panel_), level_box, 0, 0, 2, 1);

    // Hardware indicators grid
    auto* hw_frame = gtk_frame_new("Hardware Components");
    auto* hw_grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(hw_grid), 4);
    gtk_grid_set_column_spacing(GTK_GRID(hw_grid), 8);
    gtk_widget_set_margin_start(hw_grid, 8);
    gtk_widget_set_margin_end(hw_grid, 8);
    gtk_widget_set_margin_top(hw_grid, 4);
    gtk_widget_set_margin_bottom(hw_grid, 4);
    gtk_frame_set_child(GTK_FRAME(hw_frame), hw_grid);

    // TPM indicator
    auto* tpm_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    tpm_indicator_ = gtk_image_new_from_icon_name("dialog-error");
    auto* tpm_label = gtk_label_new("TPM 2.0");
    gtk_box_append(GTK_BOX(tpm_box), tpm_indicator_);
    gtk_box_append(GTK_BOX(tpm_box), tpm_label);
    gtk_grid_attach(GTK_GRID(hw_grid), tpm_box, 0, 0, 1, 1);

    // Intel ME indicator
    auto* me_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    me_indicator_ = gtk_image_new_from_icon_name("dialog-error");
    auto* me_label = gtk_label_new("Intel ME");
    gtk_box_append(GTK_BOX(me_box), me_indicator_);
    gtk_box_append(GTK_BOX(me_box), me_label);
    gtk_grid_attach(GTK_GRID(hw_grid), me_box, 1, 0, 1, 1);

    // NPU indicator
    auto* npu_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    npu_indicator_ = gtk_image_new_from_icon_name("dialog-error");
    auto* npu_label = gtk_label_new("NPU");
    gtk_box_append(GTK_BOX(npu_box), npu_indicator_);
    gtk_box_append(GTK_BOX(npu_box), npu_label);
    gtk_grid_attach(GTK_GRID(hw_grid), npu_box, 0, 1, 1, 1);

    // GNA indicator
    auto* gna_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);
    gna_indicator_ = gtk_image_new_from_icon_name("dialog-error");
    auto* gna_label = gtk_label_new("GNA");
    gtk_box_append(GTK_BOX(gna_box), gna_indicator_);
    gtk_box_append(GTK_BOX(gna_box), gna_label);
    gtk_grid_attach(GTK_GRID(hw_grid), gna_box, 1, 1, 1, 1);

    gtk_grid_attach(GTK_GRID(security_panel_), hw_frame, 0, 1, 2, 1);

    // Security metrics display
    create_metrics_display();

    // Control buttons
    auto* button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_halign(button_box, GTK_ALIGN_END);

    auto* refresh_button = gtk_button_new_with_label("Refresh Hardware");
    g_signal_connect(refresh_button, "clicked", G_CALLBACK(on_refresh_hardware_clicked), this);
    gtk_box_append(GTK_BOX(button_box), refresh_button);

    auto* settings_button = gtk_button_new_with_label("Security Settings");
    g_signal_connect(settings_button, "clicked", G_CALLBACK(on_security_settings_clicked), this);
    gtk_box_append(GTK_BOX(button_box), settings_button);

    gtk_grid_attach(GTK_GRID(security_panel_), button_box, 0, 3, 2, 1);

    // Enterprise buttons (conditional)
    auto* enterprise_button_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    gtk_widget_set_halign(enterprise_button_box, GTK_ALIGN_END);

    auto* admin_button = gtk_button_new_with_label("Enterprise Admin");
    g_signal_connect(admin_button, "clicked", G_CALLBACK(on_enterprise_admin_clicked), this);
    gtk_box_append(GTK_BOX(enterprise_button_box), admin_button);

    auto* compliance_button = gtk_button_new_with_label("Compliance Dashboard");
    g_signal_connect(compliance_button, "clicked", G_CALLBACK(on_compliance_dashboard_clicked), this);
    gtk_box_append(GTK_BOX(enterprise_button_box), compliance_button);

    auto* audit_button = gtk_button_new_with_label("Audit Logs");
    g_signal_connect(audit_button, "clicked", G_CALLBACK(on_audit_log_clicked), this);
    gtk_box_append(GTK_BOX(enterprise_button_box), audit_button);

    enterprise_panel_ = enterprise_button_box;
    gtk_grid_attach(GTK_GRID(security_panel_), enterprise_panel_, 0, 4, 2, 1);

    // Add to main window (initially hidden)
    gtk_box_append(GTK_BOX(main_box_), security_expander_);
    security_panel_visible_ = false;
    adapt_ui_to_capabilities();
}

void SecurityInterface::create_metrics_display() {
    auto* metrics_frame = gtk_frame_new("Security Metrics");
    metrics_grid_ = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(metrics_grid_), 4);
    gtk_grid_set_column_spacing(GTK_GRID(metrics_grid_), 12);
    gtk_widget_set_margin_start(metrics_grid_, 8);
    gtk_widget_set_margin_end(metrics_grid_, 8);
    gtk_widget_set_margin_top(metrics_grid_, 4);
    gtk_widget_set_margin_bottom(metrics_grid_, 4);
    gtk_frame_set_child(GTK_FRAME(metrics_frame), metrics_grid_);

    // Encryption status
    auto* enc_label = gtk_label_new("Encryption:");
    gtk_widget_set_halign(enc_label, GTK_ALIGN_START);
    encryption_status_label_ = gtk_label_new("None");
    gtk_grid_attach(GTK_GRID(metrics_grid_), enc_label, 0, 0, 1, 1);
    gtk_grid_attach(GTK_GRID(metrics_grid_), encryption_status_label_, 1, 0, 1, 1);

    // Operations counter
    auto* ops_label = gtk_label_new("Operations:");
    gtk_widget_set_halign(ops_label, GTK_ALIGN_START);
    operations_counter_label_ = gtk_label_new("0");
    gtk_grid_attach(GTK_GRID(metrics_grid_), ops_label, 0, 1, 1, 1);
    gtk_grid_attach(GTK_GRID(metrics_grid_), operations_counter_label_, 1, 1, 1, 1);

    // Latency
    auto* lat_label = gtk_label_new("Latency:");
    gtk_widget_set_halign(lat_label, GTK_ALIGN_START);
    latency_label_ = gtk_label_new("0 ms");
    gtk_grid_attach(GTK_GRID(metrics_grid_), lat_label, 0, 2, 1, 1);
    gtk_grid_attach(GTK_GRID(metrics_grid_), latency_label_, 1, 2, 1, 1);

    // CPU usage
    auto* cpu_label = gtk_label_new("CPU Usage:");
    gtk_widget_set_halign(cpu_label, GTK_ALIGN_START);
    cpu_usage_label_ = gtk_label_new("0.0%");
    gtk_grid_attach(GTK_GRID(metrics_grid_), cpu_label, 0, 3, 1, 1);
    gtk_grid_attach(GTK_GRID(metrics_grid_), cpu_usage_label_, 1, 3, 1, 1);

    // Compliance status
    auto* comp_label = gtk_label_new("Compliance:");
    gtk_widget_set_halign(comp_label, GTK_ALIGN_START);
    compliance_status_label_ = gtk_label_new("Not Configured");
    gtk_grid_attach(GTK_GRID(metrics_grid_), comp_label, 0, 4, 1, 1);
    gtk_grid_attach(GTK_GRID(metrics_grid_), compliance_status_label_, 1, 4, 1, 1);

    gtk_grid_attach(GTK_GRID(security_panel_), metrics_frame, 0, 2, 2, 1);
}

void SecurityInterface::create_hardware_status_indicators() {
    if (!status_bar_) {
        return;
    }

    // Create compact status indicators for the main status bar
    auto* status_box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 4);

    // Security level indicator
    auto* level_icon = gtk_image_new_from_icon_name("security-low");
    gtk_widget_set_tooltip_text(level_icon, "Security Level: Click to view details");

    // Make it clickable
    auto* level_button = gtk_button_new();
    gtk_button_set_child(GTK_BUTTON(level_button), level_icon);
    gtk_button_set_has_frame(GTK_BUTTON(level_button), false);
    g_signal_connect(level_button, "clicked", G_CALLBACK([](GtkButton*, gpointer user_data) {
        auto* self = static_cast<SecurityInterface*>(user_data);
        self->show_security_panel(!self->security_panel_visible_);
    }), this);

    gtk_box_append(GTK_BOX(status_box), level_button);

    // Add separator if status bar has other content
    auto* separator = gtk_separator_new(GTK_ORIENTATION_VERTICAL);
    gtk_box_append(GTK_BOX(status_box), separator);

    // Add to status bar
    gtk_box_append(GTK_BOX(status_bar_), status_box);
}

void SecurityInterface::adapt_ui_to_capabilities() {
    std::lock_guard<std::mutex> lock(capabilities_mutex_);

    // Update visibility based on security level
    bool show_security = (capabilities_.current_level > SecurityLevel::MINIMAL);
    bool show_enterprise = capabilities_.enterprise_features;

    if (security_expander_) {
        gtk_widget_set_visible(security_expander_, show_security);
        security_panel_visible_ = show_security;
    }

    if (enterprise_panel_) {
        gtk_widget_set_visible(enterprise_panel_, show_enterprise);
    }

    // Update hardware indicators
    update_hardware_indicators();
    update_security_level_display();
}

void SecurityInterface::update_hardware_indicators() {
    if (!tpm_indicator_ || !me_indicator_ || !npu_indicator_ || !gna_indicator_) {
        return;
    }

    std::lock_guard<std::mutex> lock(capabilities_mutex_);

    // Update TPM indicator
    if (capabilities_.tpm_available) {
        gtk_image_set_from_icon_name(GTK_IMAGE(tpm_indicator_), "dialog-information");
        gtk_widget_set_tooltip_text(tpm_indicator_, "TPM 2.0: Available and functional");
    } else {
        gtk_image_set_from_icon_name(GTK_IMAGE(tpm_indicator_), "dialog-error");
        gtk_widget_set_tooltip_text(tpm_indicator_, "TPM 2.0: Not available");
    }

    // Update Intel ME indicator
    if (capabilities_.intel_me_available) {
        gtk_image_set_from_icon_name(GTK_IMAGE(me_indicator_), "dialog-information");
        gtk_widget_set_tooltip_text(me_indicator_, "Intel ME: Available and functional");
    } else {
        gtk_image_set_from_icon_name(GTK_IMAGE(me_indicator_), "dialog-error");
        gtk_widget_set_tooltip_text(me_indicator_, "Intel ME: Not available");
    }

    // Update NPU indicator
    if (capabilities_.npu_available) {
        gtk_image_set_from_icon_name(GTK_IMAGE(npu_indicator_), "dialog-information");
        gtk_widget_set_tooltip_text(npu_indicator_, "NPU: Available and functional");
    } else {
        gtk_image_set_from_icon_name(GTK_IMAGE(npu_indicator_), "dialog-error");
        gtk_widget_set_tooltip_text(npu_indicator_, "NPU: Not available");
    }

    // Update GNA indicator
    if (capabilities_.gna_available) {
        gtk_image_set_from_icon_name(GTK_IMAGE(gna_indicator_), "dialog-information");
        gtk_widget_set_tooltip_text(gna_indicator_, "GNA: Available and functional");
    } else {
        gtk_image_set_from_icon_name(GTK_IMAGE(gna_indicator_), "dialog-error");
        gtk_widget_set_tooltip_text(gna_indicator_, "GNA: Not available");
    }
}

void SecurityInterface::update_security_level_display() {
    if (!security_level_label_ || !security_level_icon_) {
        return;
    }

    std::lock_guard<std::mutex> lock(capabilities_mutex_);

    std::string level_text = "Security Level: " + get_security_level_string(capabilities_.current_level);
    gtk_label_set_text(GTK_LABEL(security_level_label_), level_text.c_str());

    // Update icon based on security level
    const char* icon_name = "security-low";
    switch (capabilities_.current_level) {
        case SecurityLevel::MINIMAL:
            icon_name = "security-low";
            break;
        case SecurityLevel::BASIC:
            icon_name = "security-medium";
            break;
        case SecurityLevel::GOOD:
            icon_name = "security-high";
            break;
        case SecurityLevel::OPTIMAL:
            icon_name = "security-high";
            break;
    }
    gtk_image_set_from_icon_name(GTK_IMAGE(security_level_icon_), icon_name);

    // Set tooltip with detailed information
    std::ostringstream tooltip;
    tooltip << "Security Level: " << get_security_level_string(capabilities_.current_level) << "\n";
    tooltip << "Hardware: ";
    if (capabilities_.tpm_available) tooltip << "TPM ";
    if (capabilities_.intel_me_available) tooltip << "ME ";
    if (capabilities_.npu_available) tooltip << "NPU ";
    if (capabilities_.gna_available) tooltip << "GNA ";
    if (capabilities_.fips_compliant) tooltip << "\nFIPS 140-2 Compliant";
    if (capabilities_.enterprise_features) tooltip << "\nEnterprise Features Available";

    gtk_widget_set_tooltip_text(security_level_label_, tooltip.str().c_str());
}

void SecurityInterface::update_real_time_metrics(const SecurityMetrics& metrics) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_ = metrics;

    // Schedule UI update in main thread
    g_idle_add([](gpointer user_data) -> gboolean {
        auto* self = static_cast<SecurityInterface*>(user_data);
        self->update_performance_metrics();
        return G_SOURCE_REMOVE;
    }, this);
}

void SecurityInterface::update_performance_metrics() {
    if (ui_update_in_progress_.exchange(true)) {
        return; // Prevent concurrent updates
    }

    std::lock_guard<std::mutex> lock(metrics_mutex_);

    if (encryption_status_label_) {
        gtk_label_set_text(GTK_LABEL(encryption_status_label_), current_metrics_.active_encryption.c_str());
    }

    if (operations_counter_label_) {
        std::string ops_text = std::to_string(current_metrics_.operations_completed);
        if (current_metrics_.operations_failed > 0) {
            ops_text += " (" + std::to_string(current_metrics_.operations_failed) + " failed)";
        }
        gtk_label_set_text(GTK_LABEL(operations_counter_label_), ops_text.c_str());
    }

    if (latency_label_) {
        std::string lat_text = std::to_string(current_metrics_.current_latency_ms) + " ms";
        if (current_metrics_.avg_latency_ms > 0) {
            lat_text += " (avg: " + std::to_string(current_metrics_.avg_latency_ms) + " ms)";
        }
        gtk_label_set_text(GTK_LABEL(latency_label_), lat_text.c_str());
    }

    if (cpu_usage_label_) {
        std::ostringstream cpu_stream;
        cpu_stream << std::fixed << std::setprecision(1) << current_metrics_.cpu_usage_percent << "%";
        gtk_label_set_text(GTK_LABEL(cpu_usage_label_), cpu_stream.str().c_str());
    }

    if (compliance_status_label_) {
        std::string comp_text = current_metrics_.compliance_status;
        if (current_metrics_.attestation_valid) {
            comp_text += " (Attested)";
        }
        gtk_label_set_text(GTK_LABEL(compliance_status_label_), comp_text.c_str());
    }

    ui_update_in_progress_ = false;
}

void SecurityInterface::show_security_panel(bool show) {
    if (!security_expander_) {
        return;
    }

    security_panel_visible_ = show;
    gtk_widget_set_visible(security_expander_, show);

    if (show) {
        gtk_expander_set_expanded(GTK_EXPANDER(security_expander_), true);
        // Refresh hardware detection when panel is shown
        refresh_hardware_detection();
    }
}

void SecurityInterface::refresh_hardware_detection() {
    if (hardware_detection_callback_) {
        auto new_caps = hardware_detection_callback_();
        update_hardware_capabilities(new_caps);
    }
}

void SecurityInterface::update_hardware_capabilities(const HardwareCapabilities& caps) {
    {
        std::lock_guard<std::mutex> lock(capabilities_mutex_);
        capabilities_ = caps;
    }

    // Schedule UI update
    g_idle_add([](gpointer user_data) -> gboolean {
        auto* self = static_cast<SecurityInterface*>(user_data);
        self->adapt_ui_to_capabilities();
        return G_SOURCE_REMOVE;
    }, this);

    // Start or stop metrics timer based on capabilities
    if (caps.current_level > SecurityLevel::MINIMAL && metrics_timer_id_ == 0) {
        metrics_timer_id_ = g_timeout_add(1000, on_metrics_timer, this);
    } else if (caps.current_level == SecurityLevel::MINIMAL && metrics_timer_id_ != 0) {
        g_source_remove(metrics_timer_id_);
        metrics_timer_id_ = 0;
    }
}

std::string SecurityInterface::get_security_level_string(SecurityLevel level) const {
    switch (level) {
        case SecurityLevel::MINIMAL: return "Minimal (Software Only)";
        case SecurityLevel::BASIC: return "Basic (Software + NPU)";
        case SecurityLevel::GOOD: return "Good (TPM + NPU)";
        case SecurityLevel::OPTIMAL: return "Optimal (TPM + ME + NPU + GNA)";
        default: return "Unknown";
    }
}

std::string SecurityInterface::get_security_level_color(SecurityLevel level) const {
    switch (level) {
        case SecurityLevel::MINIMAL: return "#ff6b6b"; // Red
        case SecurityLevel::BASIC: return "#ffa726"; // Orange
        case SecurityLevel::GOOD: return "#66bb6a"; // Green
        case SecurityLevel::OPTIMAL: return "#42a5f5"; // Blue
        default: return "#757575"; // Gray
    }
}

// Event handlers
void SecurityInterface::on_security_settings_clicked(GtkButton* button, gpointer user_data) {
    auto* self = static_cast<SecurityInterface*>(user_data);
    self->show_security_settings_dialog();
}

void SecurityInterface::on_enterprise_admin_clicked(GtkButton* button, gpointer user_data) {
    auto* self = static_cast<SecurityInterface*>(user_data);
    self->show_enterprise_admin_panel();
}

void SecurityInterface::on_compliance_dashboard_clicked(GtkButton* button, gpointer user_data) {
    auto* self = static_cast<SecurityInterface*>(user_data);
    self->show_compliance_dashboard();
}

void SecurityInterface::on_audit_log_clicked(GtkButton* button, gpointer user_data) {
    auto* self = static_cast<SecurityInterface*>(user_data);
    self->show_audit_log_viewer();
}

void SecurityInterface::on_refresh_hardware_clicked(GtkButton* button, gpointer user_data) {
    auto* self = static_cast<SecurityInterface*>(user_data);
    self->refresh_hardware_detection();
}

gboolean SecurityInterface::on_metrics_timer(gpointer user_data) {
    auto* self = static_cast<SecurityInterface*>(user_data);

    if (self->metrics_callback_) {
        auto metrics = self->metrics_callback_();
        self->update_real_time_metrics(metrics);
    }

    return G_SOURCE_CONTINUE;
}

void SecurityInterface::load_security_config() {
    // Load configuration from file
    std::string config_path = std::filesystem::path(std::getenv("HOME")) / ".config" / "voice-to-text" / "security.json";

    std::ifstream file(config_path);
    if (file.is_open()) {
        Json::CharReaderBuilder builder;
        Json::Value root;
        std::string errors;

        if (Json::parseFromStream(builder, file, &root, &errors)) {
            security_config_ = root;
        }
    } else {
        // Initialize with defaults
        security_config_["version"] = "1.0";
        security_config_["auto_detect_hardware"] = true;
        security_config_["enable_enterprise_features"] = false;
        security_config_["metrics_update_interval"] = 1000;
    }
}

void SecurityInterface::save_security_config() {
    std::string config_dir = std::filesystem::path(std::getenv("HOME")) / ".config" / "voice-to-text";
    std::filesystem::create_directories(config_dir);

    std::string config_path = config_dir + "/security.json";
    std::ofstream file(config_path);

    if (file.is_open()) {
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        writer->write(security_config_, &file);
    }
}

void SecurityInterface::load_audit_logs() {
    // Load audit logs from file
    std::string log_path = std::filesystem::path(std::getenv("HOME")) / ".config" / "voice-to-text" / "audit_logs.json";

    std::ifstream file(log_path);
    if (file.is_open()) {
        Json::CharReaderBuilder builder;
        Json::Value root;
        std::string errors;

        if (Json::parseFromStream(builder, file, &root, &errors)) {
            // Parse audit log entries from JSON
            audit_logs_.clear();
            for (const auto& entry : root["entries"]) {
                AuditLogEntry log_entry;
                // Parse timestamp from ISO string
                log_entry.event_type = entry["event_type"].asString();
                log_entry.user_id = entry["user_id"].asString();
                log_entry.resource = entry["resource"].asString();
                log_entry.action = entry["action"].asString();
                log_entry.result = entry["result"].asString();
                log_entry.details = entry["details"].asString();
                log_entry.security_level = static_cast<SecurityLevel>(entry["security_level"].asInt());

                audit_logs_.push_back(log_entry);
            }
        }
    }
}

void SecurityInterface::save_audit_logs() {
    std::string config_dir = std::filesystem::path(std::getenv("HOME")) / ".config" / "voice-to-text";
    std::filesystem::create_directories(config_dir);

    std::string log_path = config_dir + "/audit_logs.json";
    std::ofstream file(log_path);

    if (file.is_open()) {
        Json::Value root;
        root["version"] = "1.0";

        Json::Value entries(Json::arrayValue);
        for (const auto& entry : audit_logs_) {
            Json::Value json_entry;
            json_entry["event_type"] = entry.event_type;
            json_entry["user_id"] = entry.user_id;
            json_entry["resource"] = entry.resource;
            json_entry["action"] = entry.action;
            json_entry["result"] = entry.result;
            json_entry["details"] = entry.details;
            json_entry["security_level"] = static_cast<int>(entry.security_level);

            entries.append(json_entry);
        }

        root["entries"] = entries;

        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        writer->write(root, &file);
    }
}

void SecurityInterface::add_audit_log_entry(const AuditLogEntry& entry) {
    std::lock_guard<std::mutex> lock(audit_mutex_);
    audit_logs_.push_back(entry);

    // Keep only recent entries based on retention policy
    auto cutoff_time = std::chrono::system_clock::now() -
                      std::chrono::hours(24 * enterprise_config_.audit_retention_days);

    audit_logs_.erase(
        std::remove_if(audit_logs_.begin(), audit_logs_.end(),
            [cutoff_time](const AuditLogEntry& e) {
                return e.timestamp < cutoff_time;
            }),
        audit_logs_.end()
    );

    // Save to disk
    save_audit_logs();
}

void SecurityInterface::cleanup() {
    if (metrics_timer_id_ != 0) {
        g_source_remove(metrics_timer_id_);
        metrics_timer_id_ = 0;
    }

    save_security_config();
    save_audit_logs();

    initialized_ = false;
}

// Placeholder implementations for dialog methods
void SecurityInterface::show_security_settings_dialog() {
    // TODO: Implement comprehensive security settings dialog
    std::cout << "Security Settings Dialog - To be implemented" << std::endl;
}

void SecurityInterface::show_enterprise_admin_panel() {
    // TODO: Implement enterprise administration panel
    std::cout << "Enterprise Admin Panel - To be implemented" << std::endl;
}

void SecurityInterface::show_compliance_dashboard() {
    // TODO: Implement compliance dashboard
    std::cout << "Compliance Dashboard - To be implemented" << std::endl;
}

void SecurityInterface::show_audit_log_viewer() {
    // TODO: Implement audit log viewer
    std::cout << "Audit Log Viewer - To be implemented" << std::endl;
}

void SecurityInterface::export_audit_logs(const std::string& filepath) {
    // TODO: Implement audit log export
    std::cout << "Export Audit Logs to: " << filepath << std::endl;
}

void SecurityInterface::import_security_policy(const std::string& filepath) {
    // TODO: Implement security policy import
    std::cout << "Import Security Policy from: " << filepath << std::endl;
}

void SecurityInterface::export_security_config(const std::string& filepath) {
    // TODO: Implement security configuration export
    std::cout << "Export Security Config to: " << filepath << std::endl;
}

} // namespace vtt