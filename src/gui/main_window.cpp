#include "main_window.h"
#include <iostream>
#include <sstream>
#include <iomanip>

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
    , tray_icon_(nullptr)
    , is_recording_(false)
    , window_visible_(true) {
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
    gtk_main();
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
    
    g_idle_add([](gpointer data) -> gboolean {
        auto* window = static_cast<MainWindow*>(data);
        
        GtkTextBuffer* buffer = gtk_text_view_get_buffer(
            GTK_TEXT_VIEW(window->transcription_view_));
        
        GtkTextIter end;
        gtk_text_buffer_get_end_iter(buffer, &end);
        
        std::string formatted_text = window->format_transcription(
            static_cast<const char*>(g_object_get_data(G_OBJECT(data), "text")),
            GPOINTER_TO_INT(g_object_get_data(G_OBJECT(data), "is_final"))
        );
        
        gtk_text_buffer_insert(buffer, &end, formatted_text.c_str(), -1);
        
        gtk_text_view_scroll_to_iter(GTK_TEXT_VIEW(window->transcription_view_),
                                    &end, 0.0, FALSE, 0.0, 0.0);
        
        return G_SOURCE_REMOVE;
    }, this);
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
    if (tray_icon_) {
        g_object_unref(tray_icon_);
        tray_icon_ = nullptr;
    }
    
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
    
    status_label_ = gtk_label_new("Ready");
    gtk_widget_set_halign(status_label_, GTK_ALIGN_START);
    gtk_box_append(GTK_BOX(main_box_), status_label_);
    
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
        "_OK", GTK_RESPONSE_OK,
        nullptr
    );
    
    GtkWidget* content = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
    gtk_widget_set_margin_start(content, 20);
    gtk_widget_set_margin_end(content, 20);
    gtk_widget_set_margin_top(content, 20);
    gtk_widget_set_margin_bottom(content, 20);
    
    GtkWidget* grid = gtk_grid_new();
    gtk_grid_set_row_spacing(GTK_GRID(grid), 10);
    gtk_grid_set_column_spacing(GTK_GRID(grid), 10);
    gtk_box_append(GTK_BOX(content), grid);
    
    gtk_grid_attach(GTK_GRID(grid), gtk_label_new("Model:"), 0, 0, 1, 1);
    GtkWidget* model_combo = gtk_combo_box_text_new();
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo), "tiny");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo), "base");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo), "small");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo), "medium");
    gtk_combo_box_text_append_text(GTK_COMBO_BOX_TEXT(model_combo), "large");
    gtk_combo_box_set_active(GTK_COMBO_BOX(model_combo), 1);
    gtk_grid_attach(GTK_GRID(grid), model_combo, 1, 0, 1, 1);
    
    gtk_grid_attach(GTK_GRID(grid), gtk_label_new("Language:"), 0, 1, 1, 1);
    GtkWidget* lang_entry = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(lang_entry), "auto");
    gtk_grid_attach(GTK_GRID(grid), lang_entry, 1, 1, 1, 1);
    
    gtk_grid_attach(GTK_GRID(grid), gtk_label_new("Hotkey:"), 0, 2, 1, 1);
    GtkWidget* hotkey_entry = gtk_entry_new();
    gtk_entry_set_text(GTK_ENTRY(hotkey_entry), "Ctrl+Alt+Space");
    gtk_grid_attach(GTK_GRID(grid), hotkey_entry, 1, 2, 1, 1);
    
    gtk_widget_set_visible(dialog, TRUE);
    
    g_signal_connect(dialog, "response", G_CALLBACK(gtk_window_destroy), nullptr);
}

}