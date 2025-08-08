#pragma once

#include <gtk/gtk.h>
#include <string>
#include <vector>
#include <functional>
#include <mutex>

namespace vtt {

class MainWindow {
public:
    MainWindow();
    ~MainWindow();
    
    bool initialize(int argc, char** argv);
    void run();
    
    void start_recording();
    void stop_recording();
    void toggle_recording();
    
    void append_transcription(const std::string& text, bool is_final = false);
    void update_waveform(const std::vector<float>& samples);
    void show_notification(const std::string& title, const std::string& message);
    
    void set_recording_started_callback(std::function<void()> callback) {
        recording_started_callback_ = callback;
    }
    
    void set_recording_stopped_callback(std::function<void()> callback) {
        recording_stopped_callback_ = callback;
    }
    
    bool is_recording() const { return is_recording_; }
    
private:
    void cleanup();
    static void on_activate(GtkApplication* app, gpointer user_data);
    void create_window();
    void create_tray_icon();
    void update_ui_state();
    std::string format_transcription(const std::string& text, bool is_final);
    
    static void on_record_toggled(GtkToggleButton* button, gpointer user_data);
    static void on_clear_clicked(GtkButton* button, gpointer user_data);
    static void on_settings_clicked(GtkButton* button, gpointer user_data);
    static gboolean on_window_close(GtkWindow* self, gpointer user_data);
    static void on_draw_waveform(GtkDrawingArea* area, cairo_t* cr, 
                                int width, int height, gpointer user_data);
    
    void show_settings_dialog();
    
    GtkApplication* app_;
    GtkWidget* window_;
    GtkWidget* header_bar_;
    GtkWidget* main_box_;
    GtkWidget* status_label_;
    GtkWidget* transcription_view_;
    GtkWidget* record_button_;
    GtkWidget* clear_button_;
    GtkWidget* settings_button_;
    GtkWidget* drawing_area_;
    GtkStatusIcon* tray_icon_;
    
    bool is_recording_;
    bool window_visible_;
    
    std::vector<float> waveform_data_;
    std::mutex waveform_mutex_;
    
    std::function<void()> recording_started_callback_;
    std::function<void()> recording_stopped_callback_;
    
    static MainWindow* instance_;
};

}