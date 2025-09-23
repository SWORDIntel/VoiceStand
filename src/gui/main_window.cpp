#include "main_window.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <filesystem>

namespace vtt {

MainWindow* MainWindow::instance_ = nullptr;

MainWindow::MainWindow()
    : app_(nullptr)
    , window_(nullptr)
    , header_bar_(nullptr)
    , main_box_(nullptr)
    , status_label_(nullptr)
    , transcription_view_(nullptr)
    , record_button_(nullptr)
    , clear_button_(nullptr)
    , settings_button_(nullptr)
    , drawing_area_(nullptr)
    , is_recording_(false)
    , window_visible_(true)
    // , security_interface_(nullptr)
    // , security_bridge_(nullptr)
    {
    instance_ = this;
}

MainWindow::~MainWindow() {
    cleanup();
    instance_ = nullptr;
}

bool MainWindow::initialize(int argc, char** argv) {
    app_ = gtk_application_new("com.vtt.voicetotext", G_APPLICATION_DEFAULT_FLAGS);
    
    g_signal_connect(app_, "activate", G_CALLBACK(on_activate), this);
    
    int status = g_application_run(G_APPLICATION(app_), argc, argv);
    
    return status == 0;
}

void MainWindow::run() {
    // GTK4 doesn't use gtk_main() anymore - the main loop is handled by g_application_run()
    // This method is now a no-op since we use GtkApplication
}

void MainWindow::start_recording() {
    if (!is_recording_) {
        is_recording_ = true;
        update_ui_state();
        
        if (recording_started_callback_) {
            recording_started_callback_();
        }
    }
}

void MainWindow::stop_recording() {
    if (is_recording_) {
        is_recording_ = false;
        update_ui_state();
        
        if (recording_stopped_callback_) {
            recording_stopped_callback_();
        }
    }
}

void MainWindow::toggle_recording() {
    if (is_recording_) {
        stop_recording();
    } else {
        start_recording();
    }
}

void MainWindow::append_transcription(const std::string& text, bool is_final) {
    if (!transcription_view_) {
        return;
    }
    
    // Create a structure to pass data to the idle callback
    struct TranscriptionData {
        MainWindow* window;
        std::string text;
        bool is_final;
    };
    
    auto* data = new TranscriptionData{this, text, is_final};
    
    g_idle_add_full(G_PRIORITY_DEFAULT, 
        [](gpointer user_data) -> gboolean {
            auto* data = static_cast<TranscriptionData*>(user_data);
            auto* window = data->window;
            
            if (!window->transcription_view_) {
                delete data;
                return G_SOURCE_REMOVE;
            }
            
            GtkTextBuffer* buffer = gtk_text_view_get_buffer(
                GTK_TEXT_VIEW(window->transcription_view_));
            
            GtkTextIter end;
            gtk_text_buffer_get_end_iter(buffer, &end);
            
            std::string formatted_text = window->format_transcription(
                data->text, data->is_final);
            
            gtk_text_buffer_insert(buffer, &end, formatted_text.c_str(), -1);
            
            // Scroll to the end of the text view
            GtkTextMark* insert_mark = gtk_text_buffer_get_insert(buffer);
            gtk_text_view_scroll_mark_onscreen(GTK_TEXT_VIEW(window->transcription_view_),
                                              insert_mark);
            
            delete data;
            return G_SOURCE_REMOVE;
        }, 
        data, 
        nullptr);
}

void MainWindow::update_waveform(const std::vector<float>& samples) {
    std::lock_guard<std::mutex> lock(waveform_mutex_);
    
    const size_t max_points = 1000;
    size_t step = std::max(size_t(1), samples.size() / max_points);
    
    waveform_data_.clear();
    for (size_t i = 0; i < samples.size(); i += step) {
        waveform_data_.push_back(samples[i]);
    }
    
    if (drawing_area_) {
        gtk_widget_queue_draw(drawing_area_);
    }
}

void MainWindow::show_notification(const std::string& title, const std::string& message) {
    GNotification* notification = g_notification_new(title.c_str());
    g_notification_set_body(notification, message.c_str());
    
    GIcon* icon = g_themed_icon_new("audio-input-microphone");
    g_notification_set_icon(notification, icon);
    g_object_unref(icon);
    
    g_application_send_notification(G_APPLICATION(app_), "vtt-notification", notification);
    g_object_unref(notification);
}

void MainWindow::cleanup() {
    // Cleanup security interface before destroying app
    /*
    if (security_interface_) {
        security_interface_->cleanup();
        security_interface_.reset();
    }

    if (security_bridge_) {
        security_bridge_->cleanup();
        security_bridge_.reset();
    }
    */

    if (app_) {
        g_object_unref(app_);
        app_ = nullptr;
    }
}

void MainWindow::on_activate(GtkApplication* app, gpointer user_data) {
    auto* window = static_cast<MainWindow*>(user_data);
    window->create_window();
}

void MainWindow::create_window() {
    window_ = gtk_application_window_new(app_);
    gtk_window_set_title(GTK_WINDOW(window_), "Voice to Text");
    gtk_window_set_default_size(GTK_WINDOW(window_), 800, 600);
    
    header_bar_ = gtk_header_bar_new();
    gtk_header_bar_set_show_title_buttons(GTK_HEADER_BAR(header_bar_), TRUE);
    gtk_window_set_titlebar(GTK_WINDOW(window_), header_bar_);
    
    record_button_ = gtk_toggle_button_new();
    GtkWidget* record_icon = gtk_image_new_from_icon_name("media-record");
    gtk_button_set_child(GTK_BUTTON(record_button_), record_icon);
    gtk_widget_set_tooltip_text(record_button_, "Start/Stop Recording (Ctrl+Alt+Space)");
    g_signal_connect(record_button_, "toggled", G_CALLBACK(on_record_toggled), this);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(header_bar_), record_button_);
    
    clear_button_ = gtk_button_new_from_icon_name("edit-clear");
    gtk_widget_set_tooltip_text(clear_button_, "Clear Transcription");
    g_signal_connect(clear_button_, "clicked", G_CALLBACK(on_clear_clicked), this);
    gtk_header_bar_pack_start(GTK_HEADER_BAR(header_bar_), clear_button_);
    
    settings_button_ = gtk_button_new_from_icon_name("preferences-system");
    gtk_widget_set_tooltip_text(settings_button_, "Settings");
    g_signal_connect(settings_button_, "clicked", G_CALLBACK(on_settings_clicked), this);
    gtk_header_bar_pack_end(GTK_HEADER_BAR(header_bar_), settings_button_);
    
    main_box_ = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_set_margin_start(main_box_, 10);
    gtk_widget_set_margin_end(main_box_, 10);
    gtk_widget_set_margin_top(main_box_, 10);
    gtk_widget_set_margin_bottom(main_box_, 10);
    gtk_window_set_child(GTK_WINDOW(window_), main_box_);
    
    drawing_area_ = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_area_, -1, 100);
    gtk_drawing_area_set_draw_func(GTK_DRAWING_AREA(drawing_area_),
                                  on_draw_waveform, this, nullptr);
    gtk_box_append(GTK_BOX(main_box_), drawing_area_);
    
    GtkWidget* scrolled_window = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scrolled_window),
                                 GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_vexpand(scrolled_window, TRUE);
    gtk_box_append(GTK_BOX(main_box_), scrolled_window);
    
    transcription_view_ = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(transcription_view_), TRUE);
    gtk_text_view_set_wrap_mode(GTK_TEXT_VIEW(transcription_view_), GTK_WRAP_WORD);
    gtk_text_view_set_left_margin(GTK_TEXT_VIEW(transcription_view_), 10);
    gtk_text_view_set_right_margin(GTK_TEXT_VIEW(transcription_view_), 10);
    gtk_text_view_set_top_margin(GTK_TEXT_VIEW(transcription_view_), 10);
    gtk_text_view_set_bottom_margin(GTK_TEXT_VIEW(transcription_view_), 10);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scrolled_window), transcription_view_);
    
    // Create status bar with status label
    GtkWidget* status_bar = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    status_label_ = gtk_label_new("Ready");
    gtk_widget_set_halign(status_label_, GTK_ALIGN_START);
    gtk_widget_set_hexpand(status_label_, TRUE);
    gtk_box_append(GTK_BOX(status_bar), status_label_);
    gtk_box_append(GTK_BOX(main_box_), status_bar);

    // Initialize security interface after main UI is created
    initialize_security_interface();

    create_tray_icon();

    g_signal_connect(window_, "close-request", G_CALLBACK(on_window_close), this);

    gtk_widget_set_visible(window_, TRUE);
}

void MainWindow::create_tray_icon() {
}

void MainWindow::update_ui_state() {
    if (!window_) {
        return;
    }
    
    g_idle_add([](gpointer data) -> gboolean {
        auto* window = static_cast<MainWindow*>(data);
        
        if (window->record_button_) {
            gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(window->record_button_),
                                        window->is_recording_);
            
            GtkWidget* icon = window->is_recording_ 
                ? gtk_image_new_from_icon_name("media-playback-stop")
                : gtk_image_new_from_icon_name("media-record");
            gtk_button_set_child(GTK_BUTTON(window->record_button_), icon);
        }
        
        if (window->status_label_) {
            const char* status = window->is_recording_ ? "Recording..." : "Ready";
            gtk_label_set_text(GTK_LABEL(window->status_label_), status);
        }
        
        return G_SOURCE_REMOVE;
    }, this);
}

std::string MainWindow::format_transcription(const std::string& text, bool is_final) {
    std::stringstream ss;
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    ss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] ";
    
    if (!is_final) {
        ss << "(interim) ";
    }
    
    ss << text;
    
    if (is_final) {
        ss << "\n";
    }
    
    return ss.str();
}

void MainWindow::on_record_toggled(GtkToggleButton* button, gpointer user_data) {
    auto* window = static_cast<MainWindow*>(user_data);
    
    if (gtk_toggle_button_get_active(button)) {
        window->start_recording();
    } else {
        window->stop_recording();
    }
}

void MainWindow::on_clear_clicked(GtkButton* button, gpointer user_data) {
    auto* window = static_cast<MainWindow*>(user_data);
    
    if (window->transcription_view_) {
        GtkTextBuffer* buffer = gtk_text_view_get_buffer(
            GTK_TEXT_VIEW(window->transcription_view_));
        gtk_text_buffer_set_text(buffer, "", -1);
    }
}

void MainWindow::on_settings_clicked(GtkButton* button, gpointer user_data) {
    auto* window = static_cast<MainWindow*>(user_data);
    window->show_settings_dialog();
}

gboolean MainWindow::on_window_close(GtkWindow* self, gpointer user_data) {
    auto* window = static_cast<MainWindow*>(user_data);
    
    window->window_visible_ = false;
    gtk_widget_set_visible(GTK_WIDGET(self), FALSE);
    
    window->show_notification("Voice to Text", 
                            "Application minimized to system tray");
    
    return TRUE;
}

void MainWindow::on_draw_waveform(GtkDrawingArea* area, cairo_t* cr,
                                 int width, int height, gpointer user_data) {
    auto* window = static_cast<MainWindow*>(user_data);
    
    cairo_set_source_rgb(cr, 0.1, 0.1, 0.1);
    cairo_paint(cr);
    
    std::lock_guard<std::mutex> lock(window->waveform_mutex_);
    
    if (window->waveform_data_.empty()) {
        return;
    }
    
    cairo_set_source_rgb(cr, 0.0, 0.8, 0.0);
    cairo_set_line_width(cr, 1.0);
    
    double x_scale = static_cast<double>(width) / window->waveform_data_.size();
    double y_mid = height / 2.0;
    double y_scale = height / 2.0;
    
    cairo_move_to(cr, 0, y_mid);
    
    for (size_t i = 0; i < window->waveform_data_.size(); ++i) {
        double x = i * x_scale;
        double y = y_mid - (window->waveform_data_[i] * y_scale);
        cairo_line_to(cr, x, y);
    }
    
    cairo_stroke(cr);
}

void MainWindow::show_settings_dialog() {
    GtkWidget* dialog = gtk_dialog_new_with_buttons(
        "Settings",
        GTK_WINDOW(window_),
        GTK_DIALOG_MODAL,
        "_Cancel", GTK_RESPONSE_CANCEL,
        "_Save", GTK_RESPONSE_OK,
        nullptr
    );
    
    gtk_window_set_default_size(GTK_WINDOW(dialog), 500, 400);
    
    GtkWidget* content = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_widget_set_margin_start(content, 20);
    gtk_widget_set_margin_end(content, 20);
    gtk_widget_set_margin_top(content, 20);
    gtk_widget_set_margin_bottom(content, 20);
    
    // Create notebook for tabbed interface
    GtkWidget* notebook = gtk_notebook_new();
    gtk_box_append(GTK_BOX(content), notebook);
    
    // Audio Settings Tab
    GtkWidget* audio_grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(audio_grid), 10);
    gtk_grid_set_column_spacing(GTK_GRID(audio_grid), 10);
    gtk_widget_set_margin_start(audio_grid, 10);
    gtk_widget_set_margin_end(audio_grid, 10);
    gtk_widget_set_margin_top(audio_grid, 10);
    gtk_widget_set_margin_bottom(audio_grid, 10);
    
    int row = 0;
    gtk_grid_attach(GTK_GRID(audio_grid), gtk_label_new("VAD Threshold:"), 0, row, 1, 1);
    vad_scale_ = gtk_scale_new_with_range(GTK_ORIENTATION_HORIZONTAL, 0.0, 1.0, 0.05);
    gtk_range_set_value(GTK_RANGE(vad_scale_), config_["audio"]["vad_threshold"].asFloat());
    gtk_scale_set_draw_value(GTK_SCALE(vad_scale_), TRUE);
    gtk_widget_set_hexpand(vad_scale_, TRUE);
    gtk_grid_attach(GTK_GRID(audio_grid), vad_scale_, 1, row++, 1, 1);
    
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), audio_grid, gtk_label_new("Audio"));
    
    // Whisper Settings Tab
    GtkWidget* whisper_grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(whisper_grid), 10);
    gtk_grid_set_column_spacing(GTK_GRID(whisper_grid), 10);
    gtk_widget_set_margin_start(whisper_grid, 10);
    gtk_widget_set_margin_end(whisper_grid, 10);
    gtk_widget_set_margin_top(whisper_grid, 10);
    gtk_widget_set_margin_bottom(whisper_grid, 10);
    
    row = 0;
    gtk_grid_attach(GTK_GRID(whisper_grid), gtk_label_new("Model Size:"), 0, row, 1, 1);
    model_combo_ = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo_), "tiny");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo_), "base");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo_), "small");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo_), "medium");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo_), "large");
    
    // Set active model based on config
    std::string model_path = config_["whisper"]["model_path"].asString();
    if (model_path.find("tiny") != std::string::npos) gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo_), 0);
    else if (model_path.find("base") != std::string::npos) gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo_), 1);
    else if (model_path.find("small") != std::string::npos) gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo_), 2);
    else if (model_path.find("medium") != std::string::npos) gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo_), 3);
    else if (model_path.find("large") != std::string::npos) gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo_), 4);
    else gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo_), 1); // default to base
    
    gtk_widget_set_hexpand(model_combo_, TRUE);
    gtk_grid_attach(GTK_GRID(whisper_grid), model_combo_, 1, row++, 1, 1);
    
    gtk_grid_attach(GTK_GRID(whisper_grid), gtk_label_new("Language:"), 0, row, 1, 1);
    language_entry_ = gtk_entry_new();
    gtk_editable_set_text(GTK_EDITABLE(language_entry_), config_["whisper"]["language"].asString().c_str());
    gtk_entry_set_placeholder_text(GTK_ENTRY(language_entry_), "auto, en, es, fr, de, etc.");
    gtk_widget_set_hexpand(language_entry_, TRUE);
    gtk_grid_attach(GTK_GRID(whisper_grid), language_entry_, 1, row++, 1, 1);
    
    gtk_grid_attach(GTK_GRID(whisper_grid), gtk_label_new("Threads:"), 0, row, 1, 1);
    threads_spin_ = gtk_spin_button_new_with_range(1, 16, 1);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(threads_spin_), config_["whisper"]["num_threads"].asInt());
    gtk_grid_attach(GTK_GRID(whisper_grid), threads_spin_, 1, row++, 1, 1);
    
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), whisper_grid, gtk_label_new("Whisper"));
    
    // Hotkeys Tab
    GtkWidget* hotkey_grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(hotkey_grid), 10);
    gtk_grid_set_column_spacing(GTK_GRID(hotkey_grid), 10);
    gtk_widget_set_margin_start(hotkey_grid, 10);
    gtk_widget_set_margin_end(hotkey_grid, 10);
    gtk_widget_set_margin_top(hotkey_grid, 10);
    gtk_widget_set_margin_bottom(hotkey_grid, 10);
    
    row = 0;
    gtk_grid_attach(GTK_GRID(hotkey_grid), gtk_label_new("Toggle Recording:"), 0, row, 1, 1);
    hotkey_entry_ = gtk_entry_new();
    gtk_editable_set_text(GTK_EDITABLE(hotkey_entry_), config_["hotkeys"]["toggle_recording"].asString().c_str());
    gtk_entry_set_placeholder_text(GTK_ENTRY(hotkey_entry_), "e.g. Ctrl+Alt+Space");
    gtk_widget_set_hexpand(hotkey_entry_, TRUE);
    gtk_grid_attach(GTK_GRID(hotkey_grid), hotkey_entry_, 1, row++, 1, 1);
    
    GtkWidget* hotkey_note = gtk_label_new("Note: Restart required for hotkey changes to take effect");
    gtk_widget_set_halign(hotkey_note, GTK_ALIGN_START);
    gtk_grid_attach(GTK_GRID(hotkey_grid), hotkey_note, 0, row++, 2, 1);
    
    gtk_notebook_append_page(GTK_NOTEBOOK(notebook), hotkey_grid, gtk_label_new("Hotkeys"));
    
    // Connect response signal for dialog
    g_signal_connect(dialog, "response", 
        G_CALLBACK(+[](GtkDialog* dialog, gint response_id, gpointer user_data) {
            auto* window = static_cast<MainWindow*>(user_data);
            if (response_id == GTK_RESPONSE_OK) {
                window->save_settings_from_dialog(GTK_WIDGET(dialog));
            }
            gtk_window_destroy(GTK_WINDOW(dialog));
        }), this);
    
    // Show dialog (non-blocking in GTK4)
    gtk_widget_set_visible(dialog, TRUE);
}

void MainWindow::save_settings_from_dialog(GtkWidget* dialog) {
    // Update config from dialog widgets
    config_["audio"]["vad_threshold"] = gtk_range_get_value(GTK_RANGE(vad_scale_));
    
    // Get selected model and update path
    const char* model_size = gtk_combo_box_text_get_active_text(GTK_COMBO_BOX_TEXT(model_combo_));
    if (model_size) {
        std::filesystem::path config_dir = std::filesystem::path(getenv("HOME")) / ".config" / "voice-to-text";
        std::filesystem::path models_dir = config_dir / "models";
        std::string filename = std::string("ggml-") + model_size + ".bin";
        config_["whisper"]["model_path"] = (models_dir / filename).string();
    }
    
    config_["whisper"]["language"] = gtk_editable_get_text(GTK_EDITABLE(language_entry_));
    config_["whisper"]["num_threads"] = static_cast<int>(gtk_spin_button_get_value(GTK_SPIN_BUTTON(threads_spin_)));
    config_["hotkeys"]["toggle_recording"] = gtk_editable_get_text(GTK_EDITABLE(hotkey_entry_));
    
    // Save config to file
    std::filesystem::path config_dir = std::filesystem::path(getenv("HOME")) / ".config" / "voice-to-text";
    std::filesystem::path config_file = config_dir / "config.json";
    
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    
    std::ofstream file(config_file);
    if (file.is_open()) {
        writer->write(config_, &file);
        file.close();
        
        show_notification("Settings Saved", "Configuration has been saved successfully");
        
        // Notify the app about config changes
        if (config_update_callback_) {
            config_update_callback_(config_);
        }
    } else {
        std::cerr << "[ERROR] Failed to save configuration file\n";
        show_notification("Error", "Failed to save settings");
    }
}

// Security interface integration methods
void MainWindow::initialize_security_interface() {
    /*
    try {
        // Initialize security bridge first
        security_bridge_ = std::make_unique<SecurityHardwareBridge>();
        if (!security_bridge_->initialize()) {
            std::cerr << "Warning: Security bridge initialization failed, running in software-only mode" << std::endl;
        }

        // Initialize security interface
        security_interface_ = std::make_unique<SecurityInterface>();
        if (!security_interface_->initialize(window_)) {
            std::cerr << "Warning: Security interface initialization failed" << std::endl;
            return;
        }

        // Find status bar widget for integration
        GtkWidget* status_bar = nullptr;
        GtkWidget* child = gtk_widget_get_first_child(main_box_);
        while (child) {
            if (GTK_IS_BOX(child)) {
                // Check if this is our status bar (last box in main_box_)
                GtkWidget* next = gtk_widget_get_next_sibling(child);
                if (!next) {
                    status_bar = child;
                    break;
                }
            }
            child = gtk_widget_get_next_sibling(child);
        }

        // Integrate with main window
        security_interface_->integrate_with_main_window(main_box_, status_bar);

        // Set up callbacks for hardware detection and metrics
        security_interface_->set_hardware_detection_callback(
            [this]() -> HardwareCapabilities {
                return get_hardware_capabilities();
            }
        );

        security_interface_->set_metrics_callback(
            [this]() -> SecurityMetrics {
                return get_security_metrics();
            }
        );

        security_interface_->set_security_config_callback(
            [this](const Json::Value& security_config) {
                // Integrate security config with main config
                config_["security"] = security_config;
                if (config_update_callback_) {
                    config_update_callback_(config_);
                }
            }
        );

        // Initial hardware detection
        update_security_status();

        std::cout << "Security interface initialized successfully" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing security interface: " << e.what() << std::endl;
    }
    */
}

void MainWindow::update_security_status() {
    /*
    if (!security_bridge_ || !security_interface_) {
        return;
    }

    try {
        // Get current hardware capabilities
        auto capabilities = security_bridge_->detect_hardware_capabilities();
        security_interface_->update_hardware_capabilities(capabilities);

        // Log security status change
        if (security_interface_->is_enterprise_mode()) {
            AuditLogEntry log_entry;
            log_entry.timestamp = std::chrono::system_clock::now();
            log_entry.event_type = "SECURITY_STATUS_UPDATE";
            log_entry.user_id = std::getenv("USER") ? std::getenv("USER") : "unknown";
            log_entry.resource = "VoiceStand GUI";
            log_entry.action = "Hardware Detection";
            log_entry.result = "SUCCESS";
            log_entry.details = "Security level: " + security_interface_->get_security_level_string(capabilities.current_level);
            log_entry.security_level = capabilities.current_level;

            security_interface_->add_audit_log_entry(log_entry);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error updating security status: " << e.what() << std::endl;
    }
    */
}

/*
HardwareCapabilities MainWindow::get_hardware_capabilities() {
    if (!security_bridge_) {
        return HardwareCapabilities{}; // Return default capabilities
    }

    return security_bridge_->detect_hardware_capabilities();
}

SecurityMetrics MainWindow::get_security_metrics() {
    if (!security_bridge_) {
        return SecurityMetrics{}; // Return default metrics
    }

    return security_bridge_->get_current_security_metrics();
}
*/

}