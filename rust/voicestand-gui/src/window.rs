use crate::{WaveformWidget, SettingsDialog, EventHandler, DefaultEventHandler};
use voicestand_core::{Result, VoiceStandError, GuiEvent};

use gtk4::prelude::*;
use gtk4::{
    Application, ApplicationWindow, Box as GtkBox, Button, Label, ScrolledWindow,
    TextView, TextBuffer, Orientation, Align, HeaderBar, MenuButton, PopoverMenu,
    ToggleButton, Separator, ProgressBar, Paned, Frame
};
use glib::clone;
use parking_lot::Mutex;
use std::sync::Arc;

/// Main application window
pub struct MainWindow {
    window: ApplicationWindow,
    transcription_view: TextView,
    transcription_buffer: TextBuffer,
    recording_button: ToggleButton,
    status_label: Label,
    progress_bar: ProgressBar,
    waveform_widget: Arc<Mutex<WaveformWidget>>,
    event_handler: Arc<dyn EventHandler + Send + Sync>,
}

impl MainWindow {
    /// Create new main window
    pub fn new(app: &Application) -> Result<Self> {
        let window = ApplicationWindow::builder()
            .application(app)
            .title("VoiceStand - Voice to Text")
            .default_width(800)
            .default_height(600)
            .build();

        // Create header bar
        let header_bar = HeaderBar::new();
        header_bar.set_title_widget(Some(&Label::new(Some("VoiceStand"))));
        window.set_titlebar(Some(&header_bar));

        // Create main layout
        let main_box = GtkBox::new(Orientation::Vertical, 0);
        window.set_child(Some(&main_box));

        // Create toolbar
        let toolbar = Self::create_toolbar()?;
        main_box.append(&toolbar);

        // Create content area with paned layout
        let paned = Paned::new(Orientation::Vertical);
        paned.set_vexpand(true);
        main_box.append(&paned);

        // Create waveform area
        let waveform_frame = Frame::new(Some("Audio Waveform"));
        let waveform_widget = WaveformWidget::new()?;
        waveform_frame.set_child(Some(waveform_widget.widget()));
        paned.set_start_child(Some(&waveform_frame));

        // Create transcription area
        let transcription_area = Self::create_transcription_area()?;
        paned.set_end_child(Some(&transcription_area));

        // Set initial paned position (30% waveform, 70% transcription)
        paned.set_position(180);

        // Create status bar
        let status_bar = Self::create_status_bar()?;
        main_box.append(&status_bar);

        // Get references to important widgets
        let transcription_view = transcription_area
            .first_child()
            .and_then(|child| child.downcast::<ScrolledWindow>().ok())
            .and_then(|scrolled| scrolled.child())
            .and_then(|child| child.downcast::<TextView>().ok())
            .ok_or_else(|| VoiceStandError::gui("Failed to find transcription TextView"))?;

        let transcription_buffer = transcription_view.buffer();

        let recording_button = toolbar
            .first_child()
            .and_then(|child| child.downcast::<ToggleButton>().ok())
            .ok_or_else(|| VoiceStandError::gui("Failed to find recording ToggleButton"))?;

        let status_label = status_bar
            .first_child()
            .and_then(|child| child.downcast::<Label>().ok())
            .ok_or_else(|| VoiceStandError::gui("Failed to find status Label"))?;

        let progress_bar = status_bar
            .last_child()
            .and_then(|child| child.downcast::<ProgressBar>().ok())
            .ok_or_else(|| VoiceStandError::gui("Failed to find ProgressBar"))?;

        // Create window instance
        let main_window = Self {
            window,
            transcription_view,
            transcription_buffer,
            recording_button,
            status_label,
            progress_bar,
            waveform_widget: Arc::new(Mutex::new(waveform_widget)),
            event_handler: Arc::new(DefaultEventHandler),
        };

        // Setup event handlers
        main_window.setup_event_handlers()?;

        Ok(main_window)
    }

    /// Create toolbar with recording controls
    fn create_toolbar() -> Result<GtkBox> {
        let toolbar = GtkBox::new(Orientation::Horizontal, 12);
        toolbar.set_margin_top(12);
        toolbar.set_margin_bottom(12);
        toolbar.set_margin_start(12);
        toolbar.set_margin_end(12);

        // Recording toggle button
        let recording_button = ToggleButton::with_label("🎤 Start Recording");
        recording_button.set_tooltip_text(Some("Toggle recording (Ctrl+Alt+Space)"));
        toolbar.append(&recording_button);

        // Separator
        let separator = Separator::new(Orientation::Vertical);
        toolbar.append(&separator);

        // Clear button
        let clear_button = Button::with_label("🗑️ Clear");
        clear_button.set_tooltip_text(Some("Clear transcription"));
        toolbar.append(&clear_button);

        // Export button
        let export_button = Button::with_label("💾 Export");
        export_button.set_tooltip_text(Some("Export transcription"));
        toolbar.append(&export_button);

        // Settings button
        let settings_button = Button::with_label("⚙️ Settings");
        settings_button.set_tooltip_text(Some("Open settings"));
        toolbar.append(&settings_button);

        // Spacer
        let spacer = Label::new(None);
        spacer.set_hexpand(true);
        toolbar.append(&spacer);

        // Menu button
        let menu_button = MenuButton::new();
        menu_button.set_icon_name("open-menu-symbolic");
        toolbar.append(&menu_button);

        Ok(toolbar)
    }

    /// Create transcription text area
    fn create_transcription_area() -> Result<Frame> {
        let frame = Frame::new(Some("Transcription"));

        let scrolled = ScrolledWindow::new();
        scrolled.set_policy(gtk4::PolicyType::Automatic, gtk4::PolicyType::Automatic);
        scrolled.set_vexpand(true);

        let text_view = TextView::new();
        text_view.set_editable(true);
        text_view.set_wrap_mode(gtk4::WrapMode::Word);
        text_view.set_margin_top(12);
        text_view.set_margin_bottom(12);
        text_view.set_margin_start(12);
        text_view.set_margin_end(12);

        // Set font
        let font_desc = pango::FontDescription::from_string("Monospace 11");
        text_view.override_font(Some(&font_desc));

        scrolled.set_child(Some(&text_view));
        frame.set_child(Some(&scrolled));

        Ok(frame)
    }

    /// Create status bar
    fn create_status_bar() -> Result<GtkBox> {
        let status_bar = GtkBox::new(Orientation::Horizontal, 12);
        status_bar.set_margin_top(6);
        status_bar.set_margin_bottom(6);
        status_bar.set_margin_start(12);
        status_bar.set_margin_end(12);

        // Status label
        let status_label = Label::new(Some("Ready"));
        status_label.set_halign(Align::Start);
        status_bar.append(&status_label);

        // Spacer
        let spacer = Label::new(None);
        spacer.set_hexpand(true);
        status_bar.append(&spacer);

        // Progress bar
        let progress_bar = ProgressBar::new();
        progress_bar.set_show_text(true);
        progress_bar.set_text(Some("0%"));
        progress_bar.set_size_request(200, -1);
        status_bar.append(&progress_bar);

        Ok(status_bar)
    }

    /// Setup event handlers for UI elements
    fn setup_event_handlers(&self) -> Result<()> {
        let event_handler = self.event_handler.clone();

        // Recording button
        self.recording_button.connect_toggled(clone!(@weak self.recording_button as button => move |_| {
            if button.is_active() {
                button.set_label("🛑 Stop Recording");
                event_handler.handle_recording_toggle();
            } else {
                button.set_label("🎤 Start Recording");
                event_handler.handle_recording_toggle();
            }
        }));

        // Find and connect other buttons
        if let Some(toolbar) = self.window.child()
            .and_then(|child| child.downcast::<GtkBox>().ok())
            .and_then(|main_box| main_box.first_child())
            .and_then(|child| child.downcast::<GtkBox>().ok()) {

            // Connect clear button
            if let Some(clear_button) = Self::find_button_by_label(&toolbar, "🗑️ Clear") {
                clear_button.connect_clicked(clone!(@weak self.transcription_buffer as buffer, @strong event_handler => move |_| {
                    buffer.set_text("");
                    event_handler.handle_clear_transcription();
                }));
            }

            // Connect export button
            if let Some(export_button) = Self::find_button_by_label(&toolbar, "💾 Export") {
                export_button.connect_clicked(clone!(@weak self.transcription_buffer as buffer, @strong event_handler => move |_| {
                    let text = buffer.text(&buffer.start_iter(), &buffer.end_iter(), false);
                    event_handler.handle_export_transcription(&text);
                }));
            }

            // Connect settings button
            if let Some(settings_button) = Self::find_button_by_label(&toolbar, "⚙️ Settings") {
                settings_button.connect_clicked(clone!(@strong event_handler => move |_| {
                    event_handler.handle_settings_open();
                }));
            }
        }

        Ok(())
    }

    /// Helper to find button by label text
    fn find_button_by_label(container: &GtkBox, label: &str) -> Option<Button> {
        let mut child = container.first_child();
        while let Some(widget) = child {
            if let Ok(button) = widget.downcast::<Button>() {
                if let Some(button_label) = button.label() {
                    if button_label == label {
                        return Some(button);
                    }
                }
            }
            child = widget.next_sibling();
        }
        None
    }

    /// Update transcription text
    pub fn update_transcription(&self, text: &str, is_final: bool, confidence: f32) {
        let mut end_iter = self.transcription_buffer.end_iter();

        if is_final {
            self.transcription_buffer.insert(&mut end_iter, &format!("{}\n", text));

            // Auto-scroll to bottom
            let mark = self.transcription_buffer.insert_mark(None, &end_iter, false);
            self.transcription_view.scroll_mark_onscreen(&mark);
        } else {
            // For partial results, show in status
            self.status_label.set_text(&format!("Partial: {} ({}%)", text, (confidence * 100.0) as i32));
        }
    }

    /// Update waveform display
    pub fn update_waveform(&self, samples: &[f32]) {
        if let Ok(mut waveform) = self.waveform_widget.try_lock() {
            waveform.update_samples(samples);
        }
    }

    /// Update recording status
    pub fn update_recording_status(&self, recording: bool) {
        self.recording_button.set_active(recording);

        if recording {
            self.status_label.set_text("Recording...");
            self.progress_bar.pulse();
        } else {
            self.status_label.set_text("Ready");
            self.progress_bar.set_fraction(0.0);
        }
    }

    /// Show error message
    pub fn show_error(&self, message: &str) {
        let dialog = gtk4::AlertDialog::builder()
            .modal(true)
            .message("Error")
            .detail(message)
            .build();

        dialog.show(Some(&self.window));
    }

    /// Handle GUI events
    pub fn handle_gui_event(&self, event: GuiEvent) {
        match event {
            GuiEvent::ToggleWindow => {
                if self.window.is_visible() {
                    self.window.hide();
                } else {
                    self.window.show();
                }
            }
            GuiEvent::ShowSettings => {
                if let Ok(settings) = SettingsDialog::new(&self.window) {
                    settings.show();
                }
            }
            GuiEvent::UpdateTranscription { text, is_final, confidence } => {
                self.update_transcription(&text, is_final, confidence);
            }
            GuiEvent::UpdateWaveform(samples) => {
                self.update_waveform(&samples);
            }
            GuiEvent::UpdateRecordingStatus(recording) => {
                self.update_recording_status(recording);
            }
            GuiEvent::ClearTranscription => {
                self.transcription_buffer.set_text("");
            }
            GuiEvent::ExportTranscription(text) => {
                self.export_transcription(&text);
            }
        }
    }

    /// Export transcription to file
    fn export_transcription(&self, text: &str) {
        let dialog = gtk4::FileDialog::builder()
            .title("Export Transcription")
            .modal(true)
            .build();

        dialog.save(Some(&self.window), None::<&gtk4::gio::Cancellable>, clone!(@strong text => move |result| {
            match result {
                Ok(file) => {
                    if let Some(path) = file.path() {
                        if let Err(e) = std::fs::write(&path, &text) {
                            tracing::error!("Failed to export transcription: {}", e);
                        } else {
                            tracing::info!("Transcription exported to: {:?}", path);
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Export dialog error: {}", e);
                }
            }
        }));
    }

    /// Set custom event handler
    pub fn set_event_handler(&mut self, handler: Arc<dyn EventHandler + Send + Sync>) {
        self.event_handler = handler;
    }

    /// Show the window
    pub fn show(&self) {
        self.window.show();
    }

    /// Close the window
    pub fn close(&self) {
        self.window.close();
    }

    /// Get the GTK window
    pub fn window(&self) -> &ApplicationWindow {
        &self.window
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_creation() {
        // This test would require GTK to be initialized
        // In a real test environment, you would need to set this up properly
    }
}