use voicestand_core::{VoiceStandConfig, AudioConfig, SpeechConfig, GuiConfig, HotkeyConfig, Result};
use gtk4::prelude::*;
use gtk4::{
    Window, Dialog, Box as GtkBox, Grid, Label, Entry, SpinButton, Switch,
    ComboBoxText, Button, Orientation, ResponseType, FileChooserButton,
    Adjustment, Scale
};
use glib::clone;

/// Settings dialog for configuration
pub struct SettingsDialog {
    dialog: Dialog,
    config: VoiceStandConfig,
}

impl SettingsDialog {
    /// Create new settings dialog
    pub fn new(parent: &Window) -> Result<Self> {
        let dialog = Dialog::builder()
            .title("VoiceStand Settings")
            .modal(true)
            .transient_for(parent)
            .default_width(500)
            .default_height(400)
            .build();

        // Add buttons
        dialog.add_button("Cancel", ResponseType::Cancel);
        dialog.add_button("Apply", ResponseType::Apply);
        dialog.add_button("OK", ResponseType::Ok);

        // Load current configuration
        let config = VoiceStandConfig::load().unwrap_or_default();

        let settings_dialog = Self { dialog, config };

        // Build UI
        settings_dialog.build_ui()?;

        Ok(settings_dialog)
    }

    /// Build the settings UI
    fn build_ui(&self) -> Result<()> {
        let content_area = self.dialog.content_area();
        content_area.set_spacing(12);
        content_area.set_margin_top(12);
        content_area.set_margin_bottom(12);
        content_area.set_margin_start(12);
        content_area.set_margin_end(12);

        // Create notebook for tabbed interface
        let notebook = gtk4::Notebook::new();
        content_area.append(&notebook);

        // Audio settings tab
        let audio_page = self.create_audio_settings()?;
        notebook.append_page(&audio_page, Some(&Label::new(Some("Audio"))));

        // Speech settings tab
        let speech_page = self.create_speech_settings()?;
        notebook.append_page(&speech_page, Some(&Label::new(Some("Speech"))));

        // GUI settings tab
        let gui_page = self.create_gui_settings()?;
        notebook.append_page(&gui_page, Some(&Label::new(Some("Interface"))));

        // Hotkey settings tab
        let hotkey_page = self.create_hotkey_settings()?;
        notebook.append_page(&hotkey_page, Some(&Label::new(Some("Hotkeys"))));

        // Connect response signal
        self.dialog.connect_response(clone!(@weak self.dialog as dialog => move |_, response| {
            match response {
                ResponseType::Ok | ResponseType::Apply => {
                    // Apply settings
                    // In real implementation, would extract values from widgets and save
                    tracing::info!("Settings applied");
                    if response == ResponseType::Ok {
                        dialog.close();
                    }
                }
                ResponseType::Cancel => {
                    dialog.close();
                }
                _ => {}
            }
        }));

        Ok(())
    }

    /// Create audio settings page
    fn create_audio_settings(&self) -> Result<GtkBox> {
        let page = GtkBox::new(Orientation::Vertical, 12);
        page.set_margin_top(12);
        page.set_margin_bottom(12);
        page.set_margin_start(12);
        page.set_margin_end(12);

        let grid = Grid::new();
        grid.set_column_spacing(12);
        grid.set_row_spacing(12);
        page.append(&grid);

        let mut row = 0;

        // Sample rate
        grid.attach(&Label::new(Some("Sample Rate:")), 0, row, 1, 1);
        let sample_rate_combo = ComboBoxText::new();
        sample_rate_combo.append_text("8000 Hz");
        sample_rate_combo.append_text("16000 Hz");
        sample_rate_combo.append_text("22050 Hz");
        sample_rate_combo.append_text("44100 Hz");
        sample_rate_combo.append_text("48000 Hz");
        sample_rate_combo.set_active_id(Some(&self.config.audio.sample_rate.to_string()));
        grid.attach(&sample_rate_combo, 1, row, 1, 1);
        row += 1;

        // Channels
        grid.attach(&Label::new(Some("Channels:")), 0, row, 1, 1);
        let channels_spin = SpinButton::with_range(1.0, 2.0, 1.0);
        channels_spin.set_value(self.config.audio.channels as f64);
        grid.attach(&channels_spin, 1, row, 1, 1);
        row += 1;

        // VAD threshold
        grid.attach(&Label::new(Some("Voice Activity Threshold:")), 0, row, 1, 1);
        let vad_adjustment = Adjustment::new(
            self.config.audio.vad_threshold as f64,
            0.0, 1.0, 0.01, 0.1, 0.0
        );
        let vad_scale = Scale::new(Orientation::Horizontal, Some(&vad_adjustment));
        vad_scale.set_digits(2);
        vad_scale.set_hexpand(true);
        grid.attach(&vad_scale, 1, row, 1, 1);
        row += 1;

        // Device selection
        grid.attach(&Label::new(Some("Input Device:")), 0, row, 1, 1);
        let device_combo = ComboBoxText::new();
        device_combo.append_text("Default");
        device_combo.append_text("System Microphone");
        device_combo.append_text("USB Microphone");
        device_combo.set_active(0);
        grid.attach(&device_combo, 1, row, 1, 1);
        row += 1;

        // Buffer size
        grid.attach(&Label::new(Some("Buffer Size (frames):")), 0, row, 1, 1);
        let buffer_spin = SpinButton::with_range(256.0, 4096.0, 256.0);
        buffer_spin.set_value(self.config.audio.frames_per_buffer as f64);
        grid.attach(&buffer_spin, 1, row, 1, 1);

        Ok(page)
    }

    /// Create speech settings page
    fn create_speech_settings(&self) -> Result<GtkBox> {
        let page = GtkBox::new(Orientation::Vertical, 12);
        page.set_margin_top(12);
        page.set_margin_bottom(12);
        page.set_margin_start(12);
        page.set_margin_end(12);

        let grid = Grid::new();
        grid.set_column_spacing(12);
        grid.set_row_spacing(12);
        page.append(&grid);

        let mut row = 0;

        // Model file
        grid.attach(&Label::new(Some("Model File:")), 0, row, 1, 1);
        let model_chooser = FileChooserButton::new("Select Model File", gtk4::FileChooserAction::Open);
        model_chooser.set_hexpand(true);
        grid.attach(&model_chooser, 1, row, 1, 1);
        row += 1;

        // Language
        grid.attach(&Label::new(Some("Language:")), 0, row, 1, 1);
        let language_combo = ComboBoxText::new();
        language_combo.append_text("Auto-detect");
        language_combo.append_text("English");
        language_combo.append_text("Spanish");
        language_combo.append_text("French");
        language_combo.append_text("German");
        language_combo.append_text("Italian");
        language_combo.append_text("Portuguese");
        language_combo.append_text("Russian");
        language_combo.append_text("Japanese");
        language_combo.append_text("Chinese");
        language_combo.set_active(0);
        grid.attach(&language_combo, 1, row, 1, 1);
        row += 1;

        // Number of threads
        grid.attach(&Label::new(Some("Processing Threads:")), 0, row, 1, 1);
        let threads_spin = SpinButton::with_range(1.0, 16.0, 1.0);
        threads_spin.set_value(self.config.speech.num_threads as f64);
        grid.attach(&threads_spin, 1, row, 1, 1);
        row += 1;

        // Use GPU
        grid.attach(&Label::new(Some("Use GPU Acceleration:")), 0, row, 1, 1);
        let gpu_switch = Switch::new();
        gpu_switch.set_active(self.config.speech.use_gpu);
        grid.attach(&gpu_switch, 1, row, 1, 1);
        row += 1;

        // Max tokens
        grid.attach(&Label::new(Some("Max Tokens:")), 0, row, 1, 1);
        let tokens_spin = SpinButton::with_range(64.0, 2048.0, 64.0);
        tokens_spin.set_value(self.config.speech.max_tokens as f64);
        grid.attach(&tokens_spin, 1, row, 1, 1);
        row += 1;

        // Beam size
        grid.attach(&Label::new(Some("Beam Size:")), 0, row, 1, 1);
        let beam_spin = SpinButton::with_range(1.0, 10.0, 1.0);
        beam_spin.set_value(self.config.speech.beam_size as f64);
        grid.attach(&beam_spin, 1, row, 1, 1);

        Ok(page)
    }

    /// Create GUI settings page
    fn create_gui_settings(&self) -> Result<GtkBox> {
        let page = GtkBox::new(Orientation::Vertical, 12);
        page.set_margin_top(12);
        page.set_margin_bottom(12);
        page.set_margin_start(12);
        page.set_margin_end(12);

        let grid = Grid::new();
        grid.set_column_spacing(12);
        grid.set_row_spacing(12);
        page.append(&grid);

        let mut row = 0;

        // Theme
        grid.attach(&Label::new(Some("Theme:")), 0, row, 1, 1);
        let theme_combo = ComboBoxText::new();
        theme_combo.append_text("System");
        theme_combo.append_text("Light");
        theme_combo.append_text("Dark");
        theme_combo.set_active_id(Some(&self.config.gui.theme));
        grid.attach(&theme_combo, 1, row, 1, 1);
        row += 1;

        // Show waveform
        grid.attach(&Label::new(Some("Show Waveform:")), 0, row, 1, 1);
        let waveform_switch = Switch::new();
        waveform_switch.set_active(self.config.gui.show_waveform);
        grid.attach(&waveform_switch, 1, row, 1, 1);
        row += 1;

        // Auto scroll
        grid.attach(&Label::new(Some("Auto Scroll:")), 0, row, 1, 1);
        let scroll_switch = Switch::new();
        scroll_switch.set_active(self.config.gui.auto_scroll);
        grid.attach(&scroll_switch, 1, row, 1, 1);
        row += 1;

        // Window size
        grid.attach(&Label::new(Some("Window Width:")), 0, row, 1, 1);
        let width_spin = SpinButton::with_range(400.0, 2000.0, 50.0);
        width_spin.set_value(self.config.gui.window_width as f64);
        grid.attach(&width_spin, 1, row, 1, 1);
        row += 1;

        grid.attach(&Label::new(Some("Window Height:")), 0, row, 1, 1);
        let height_spin = SpinButton::with_range(300.0, 1500.0, 50.0);
        height_spin.set_value(self.config.gui.window_height as f64);
        grid.attach(&height_spin, 1, row, 1, 1);

        Ok(page)
    }

    /// Create hotkey settings page
    fn create_hotkey_settings(&self) -> Result<GtkBox> {
        let page = GtkBox::new(Orientation::Vertical, 12);
        page.set_margin_top(12);
        page.set_margin_bottom(12);
        page.set_margin_start(12);
        page.set_margin_end(12);

        let grid = Grid::new();
        grid.set_column_spacing(12);
        grid.set_row_spacing(12);
        page.append(&grid);

        let mut row = 0;

        // Toggle recording
        grid.attach(&Label::new(Some("Toggle Recording:")), 0, row, 1, 1);
        let toggle_entry = Entry::new();
        toggle_entry.set_text(&self.config.hotkeys.toggle_recording);
        toggle_entry.set_hexpand(true);
        grid.attach(&toggle_entry, 1, row, 1, 1);

        let toggle_button = Button::with_label("Set");
        grid.attach(&toggle_button, 2, row, 1, 1);
        row += 1;

        // Push to talk
        grid.attach(&Label::new(Some("Push to Talk:")), 0, row, 1, 1);
        let ptt_entry = Entry::new();
        ptt_entry.set_text(&self.config.hotkeys.push_to_talk);
        ptt_entry.set_hexpand(true);
        grid.attach(&ptt_entry, 1, row, 1, 1);

        let ptt_button = Button::with_label("Set");
        grid.attach(&ptt_button, 2, row, 1, 1);

        // Add instructions
        let instructions = Label::new(Some(
            "Click 'Set' button and press the desired key combination.\n\
             Supported modifiers: Ctrl, Alt, Shift, Super"
        ));
        instructions.set_margin_top(20);
        instructions.set_justify(gtk4::Justification::Left);
        page.append(&instructions);

        Ok(page)
    }

    /// Show the dialog
    pub fn show(&self) {
        self.dialog.show();
    }

    /// Get the current configuration
    pub fn get_config(&self) -> &VoiceStandConfig {
        &self.config
    }

    /// Set configuration values from UI elements
    fn extract_config_from_ui(&mut self) -> Result<VoiceStandConfig> {
        // In a real implementation, this would extract values from all UI elements
        // and construct a new VoiceStandConfig

        // For now, return the current config
        Ok(self.config.clone())
    }
}

/// Widget utilities for settings
pub struct SettingsWidgets;

impl SettingsWidgets {
    /// Create a labeled spin button
    pub fn create_spin_button(
        grid: &Grid,
        row: i32,
        label: &str,
        value: f64,
        min: f64,
        max: f64,
        step: f64,
    ) -> SpinButton {
        grid.attach(&Label::new(Some(label)), 0, row, 1, 1);
        let spin = SpinButton::with_range(min, max, step);
        spin.set_value(value);
        grid.attach(&spin, 1, row, 1, 1);
        spin
    }

    /// Create a labeled switch
    pub fn create_switch(grid: &Grid, row: i32, label: &str, active: bool) -> Switch {
        grid.attach(&Label::new(Some(label)), 0, row, 1, 1);
        let switch = Switch::new();
        switch.set_active(active);
        grid.attach(&switch, 1, row, 1, 1);
        switch
    }

    /// Create a labeled combo box
    pub fn create_combo_box(grid: &Grid, row: i32, label: &str, items: &[&str]) -> ComboBoxText {
        grid.attach(&Label::new(Some(label)), 0, row, 1, 1);
        let combo = ComboBoxText::new();
        for item in items {
            combo.append_text(item);
        }
        grid.attach(&combo, 1, row, 1, 1);
        combo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_widget_creation() {
        // This test would require GTK to be initialized
        // In a real test environment, you would need to set this up properly
    }
}