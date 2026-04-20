use glib::clone;
use gtk4::prelude::*;
use gtk4::{
    Adjustment, Box as GtkBox, Button, ComboBoxText, Dialog, Entry, FileChooserAction,
    FileChooserButton, Frame, Grid, Label, Notebook, Orientation, ResponseType, Scale, SpinButton,
    Switch, Window,
};
use std::path::Path;
use voicestand_core::{Result, VoiceStandConfig};

#[derive(Clone)]
struct SettingsWidgets {
    sample_rate_combo: ComboBoxText,
    channels_spin: SpinButton,
    vad_scale: Scale,
    buffer_spin: SpinButton,
    model_combo: ComboBoxText,
    model_chooser: FileChooserButton,
    language_combo: ComboBoxText,
    threads_spin: SpinButton,
    gpu_switch: Switch,
    tokens_spin: SpinButton,
    beam_spin: SpinButton,
    theme_combo: ComboBoxText,
    waveform_switch: Switch,
    scroll_switch: Switch,
    width_spin: SpinButton,
    height_spin: SpinButton,
    toggle_entry: Entry,
    ptt_entry: Entry,
}

/// Settings dialog for configuration.
pub struct SettingsDialog {
    dialog: Dialog,
    config: VoiceStandConfig,
    widgets: Option<SettingsWidgets>,
}

impl SettingsDialog {
    /// Create new settings dialog.
    pub fn new(parent: &Window) -> Result<Self> {
        let dialog = Dialog::builder()
            .title("VoiceStand Settings")
            .modal(true)
            .transient_for(parent)
            .default_width(760)
            .default_height(560)
            .build();

        dialog.add_button("Defaults", ResponseType::Other(1));
        dialog.add_button("Cancel", ResponseType::Cancel);
        dialog.add_button("Apply", ResponseType::Apply);
        dialog.add_button("OK", ResponseType::Ok);

        let config = VoiceStandConfig::load().unwrap_or_default();
        let mut settings_dialog = Self {
            dialog,
            config,
            widgets: None,
        };
        settings_dialog.build_ui()?;

        Ok(settings_dialog)
    }

    /// Build the settings UI.
    fn build_ui(&mut self) -> Result<()> {
        let content_area = self.dialog.content_area();
        content_area.set_spacing(16);
        content_area.set_margin_top(14);
        content_area.set_margin_bottom(14);
        content_area.set_margin_start(14);
        content_area.set_margin_end(14);

        let title = Label::builder()
            .label("Configure accuracy, performance, and accessibility for transcription")
            .xalign(0.0)
            .css_classes(["title-4"])
            .build();
        content_area.append(&title);

        let notebook = Notebook::new();
        notebook.set_hexpand(true);
        notebook.set_vexpand(true);
        content_area.append(&notebook);

        let (audio_page, sample_rate_combo, channels_spin, vad_scale, buffer_spin) =
            self.create_audio_settings()?;
        let (
            speech_page,
            model_combo,
            model_chooser,
            language_combo,
            threads_spin,
            gpu_switch,
            tokens_spin,
            beam_spin,
        ) = self.create_speech_settings()?;
        let (gui_page, theme_combo, waveform_switch, scroll_switch, width_spin, height_spin) =
            self.create_gui_settings()?;
        let (hotkey_page, toggle_entry, ptt_entry) = self.create_hotkey_settings()?;

        notebook.append_page(&audio_page, Some(&Label::new(Some("Audio"))));
        notebook.append_page(&speech_page, Some(&Label::new(Some("Speech"))));
        notebook.append_page(&gui_page, Some(&Label::new(Some("Interface"))));
        notebook.append_page(&hotkey_page, Some(&Label::new(Some("Hotkeys"))));

        let widgets = SettingsWidgets {
            sample_rate_combo,
            channels_spin,
            vad_scale,
            buffer_spin,
            model_combo,
            model_chooser,
            language_combo,
            threads_spin,
            gpu_switch,
            tokens_spin,
            beam_spin,
            theme_combo,
            waveform_switch,
            scroll_switch,
            width_spin,
            height_spin,
            toggle_entry,
            ptt_entry,
        };

        self.widgets = Some(widgets.clone());

        self.dialog
            .connect_response(clone!(@weak self.dialog as dialog => move |_, response| {
                match response {
                    ResponseType::Ok | ResponseType::Apply => {
                        if let Err(err) = Self::persist_from_widgets(&widgets) {
                            tracing::error!("Failed to save settings: {}", err);
                            Self::show_feedback_dialog(&dialog, "Settings Error", &format!("Unable to save settings: {}", err));
                        } else {
                            tracing::info!("Settings applied and saved");
                            if response == ResponseType::Apply {
                                Self::show_feedback_dialog(&dialog, "Settings Saved", "Settings were validated and saved successfully.");
                            }
                        }
                        if response == ResponseType::Ok {
                            dialog.close();
                        }
                    }
                    ResponseType::Other(1) => {
                        if let Err(err) = Self::restore_defaults() {
                            tracing::error!("Failed to restore defaults: {}", err);
                            Self::show_feedback_dialog(&dialog, "Defaults Error", &format!("Unable to restore defaults: {}", err));
                        } else {
                            Self::show_feedback_dialog(&dialog, "Defaults Restored", "Default configuration restored. Re-open settings to refresh visible fields.");
                        }
                    }
                    ResponseType::Cancel => dialog.close(),
                    _ => {}
                }
            }));

        Ok(())
    }

    fn show_feedback_dialog(parent: &Dialog, title: &str, message: &str) {
        let info = Dialog::builder()
            .title(title)
            .modal(true)
            .transient_for(parent)
            .default_width(420)
            .default_height(120)
            .build();
        info.add_button("OK", ResponseType::Ok);
        let area = info.content_area();
        let label = Label::new(Some(message));
        label.set_wrap(true);
        label.set_xalign(0.0);
        area.append(&label);
        info.connect_response(|dialog, _| dialog.close());
        info.present();
    }

    fn restore_defaults() -> Result<()> {
        let defaults = VoiceStandConfig::default();
        defaults.save()
    }

    fn persist_from_widgets(w: &SettingsWidgets) -> Result<()> {
        let mut config = VoiceStandConfig::load().unwrap_or_default();

        if let Some(id) = w.sample_rate_combo.active_id() {
            config.audio.sample_rate = id.parse::<u32>().unwrap_or(16_000);
        }
        config.audio.channels = w.channels_spin.value_as_int().max(1) as u16;
        config.audio.vad_threshold = w.vad_scale.value() as f32;
        config.audio.frames_per_buffer = w.buffer_spin.value_as_int().max(128) as u32;

        if let Some(id) = w.model_combo.active_id() {
            config.speech.model_path = id.to_string();
        }
        if let Some(file) = w.model_chooser.file().and_then(|f| f.path()) {
            if file.is_file() {
                let resolved = file.canonicalize().unwrap_or(file);
                config.speech.model_path = resolved.to_string_lossy().to_string();
            }
        }

        if !Path::new(&config.speech.model_path).exists() {
            tracing::warn!(
                "Selected model path does not exist yet: {}",
                config.speech.model_path
            );
        }

        if let Some(lang) = w.language_combo.active_id() {
            config.speech.language = lang.to_string();
        }
        config.speech.num_threads = w.threads_spin.value_as_int().max(1) as usize;
        config.speech.use_gpu = w.gpu_switch.is_active();
        config.speech.max_tokens = w.tokens_spin.value_as_int().max(64) as usize;
        config.speech.beam_size = w.beam_spin.value_as_int().max(1) as usize;

        if let Some(theme) = w.theme_combo.active_id() {
            config.gui.theme = theme.to_string();
        }
        config.gui.show_waveform = w.waveform_switch.is_active();
        config.gui.auto_scroll = w.scroll_switch.is_active();
        config.gui.window_width = w.width_spin.value_as_int().max(400);
        config.gui.window_height = w.height_spin.value_as_int().max(300);

        let toggle = w.toggle_entry.text().trim().to_string();
        let ptt = w.ptt_entry.text().trim().to_string();
        config.hotkeys.toggle_recording = if toggle.is_empty() {
            "Ctrl+Alt+Space".to_string()
        } else {
            toggle
        };
        config.hotkeys.push_to_talk = if ptt.is_empty() {
            "Ctrl+Alt+V".to_string()
        } else {
            ptt
        };

        config.validate()?;
        config.save()
    }

    fn create_section(title: &str) -> (Frame, Grid) {
        let frame = Frame::new(Some(title));
        frame.set_hexpand(true);
        frame.set_margin_bottom(10);

        let grid = Grid::new();
        grid.set_column_spacing(12);
        grid.set_row_spacing(10);
        grid.set_margin_top(12);
        grid.set_margin_bottom(12);
        grid.set_margin_start(12);
        grid.set_margin_end(12);

        frame.set_child(Some(&grid));
        (frame, grid)
    }

    /// Create audio settings page.
    fn create_audio_settings(
        &self,
    ) -> Result<(GtkBox, ComboBoxText, SpinButton, Scale, SpinButton)> {
        let page = GtkBox::new(Orientation::Vertical, 10);
        page.set_margin_top(8);
        page.set_margin_bottom(8);
        page.set_margin_start(8);
        page.set_margin_end(8);

        let (audio_frame, grid) = Self::create_section("Capture & Voice Activity Detection");
        page.append(&audio_frame);

        let mut row = 0;

        grid.attach(&Label::new(Some("Sample Rate")), 0, row, 1, 1);
        let sample_rate_combo = ComboBoxText::new();
        for rate in [8000_u32, 16000, 22050, 44100, 48000] {
            let label = format!("{} Hz", rate);
            sample_rate_combo.append(Some(&rate.to_string()), &label);
        }
        sample_rate_combo.set_active_id(Some(&self.config.audio.sample_rate.to_string()));
        grid.attach(&sample_rate_combo, 1, row, 1, 1);
        row += 1;

        grid.attach(&Label::new(Some("Channels")), 0, row, 1, 1);
        let channels_spin = SpinButton::with_range(1.0, 2.0, 1.0);
        channels_spin.set_value(self.config.audio.channels as f64);
        grid.attach(&channels_spin, 1, row, 1, 1);
        row += 1;

        grid.attach(&Label::new(Some("VAD Threshold")), 0, row, 1, 1);
        let vad_adjustment = Adjustment::new(
            self.config.audio.vad_threshold as f64,
            0.0,
            1.0,
            0.01,
            0.1,
            0.0,
        );
        let vad_scale = Scale::new(Orientation::Horizontal, Some(&vad_adjustment));
        vad_scale.set_digits(2);
        vad_scale.set_hexpand(true);
        grid.attach(&vad_scale, 1, row, 1, 1);
        row += 1;

        grid.attach(&Label::new(Some("Buffer Size (frames)")), 0, row, 1, 1);
        let buffer_spin = SpinButton::with_range(256.0, 4096.0, 256.0);
        buffer_spin.set_value(self.config.audio.frames_per_buffer as f64);
        grid.attach(&buffer_spin, 1, row, 1, 1);

        Ok((
            page,
            sample_rate_combo,
            channels_spin,
            vad_scale,
            buffer_spin,
        ))
    }

    /// Create speech settings page with model preset options.
    fn create_speech_settings(
        &self,
    ) -> Result<(
        GtkBox,
        ComboBoxText,
        FileChooserButton,
        ComboBoxText,
        SpinButton,
        Switch,
        SpinButton,
        SpinButton,
    )> {
        let page = GtkBox::new(Orientation::Vertical, 10);
        page.set_margin_top(8);
        page.set_margin_bottom(8);
        page.set_margin_start(8);
        page.set_margin_end(8);

        let (model_frame, model_grid) = Self::create_section("Model Selection & Accuracy");
        page.append(&model_frame);

        let mut row = 0;
        model_grid.attach(&Label::new(Some("Model Preset")), 0, row, 1, 1);
        let model_combo = ComboBoxText::new();
        model_combo.append(
            Some("models/ggml-tiny.bin"),
            "Tiny (39MB) • Lowest RAM • Fastest",
        );
        model_combo.append(
            Some("models/ggml-base.bin"),
            "Base (142MB) • Balanced (Recommended)",
        );
        model_combo.append(
            Some("models/ggml-small.bin"),
            "Small (244MB) • Higher accuracy",
        );
        model_combo.append(
            Some("models/ggml-medium.bin"),
            "Medium (769MB) • Professional accuracy",
        );
        model_combo.append(
            Some("models/ggml-large.bin"),
            "Large (1.5GB+) • Max accuracy",
        );
        model_combo.set_active_id(Some(&self.config.speech.model_path));
        if model_combo.active_id().is_none() {
            model_combo.set_active_id(Some("models/ggml-base.bin"));
        }
        model_grid.attach(&model_combo, 1, row, 1, 1);
        row += 1;

        model_grid.attach(&Label::new(Some("Custom Model File")), 0, row, 1, 1);
        let model_chooser = FileChooserButton::new("Select Model File", FileChooserAction::Open);
        model_chooser.set_hexpand(true);
        model_grid.attach(&model_chooser, 1, row, 1, 1);
        row += 1;

        let model_hint = Label::builder()
            .label("Tip: custom file overrides preset when selected and valid.")
            .xalign(0.0)
            .wrap(true)
            .build();
        model_grid.attach(&model_hint, 0, row, 2, 1);

        let (engine_frame, engine_grid) = Self::create_section("Transcription Engine");
        page.append(&engine_frame);

        let mut engine_row = 0;
        engine_grid.attach(&Label::new(Some("Language")), 0, engine_row, 1, 1);
        let language_combo = ComboBoxText::new();
        language_combo.append(Some("auto"), "Auto-detect");
        language_combo.append(Some("en"), "English");
        language_combo.append(Some("es"), "Spanish");
        language_combo.append(Some("fr"), "French");
        language_combo.append(Some("de"), "German");
        language_combo.append(Some("it"), "Italian");
        language_combo.append(Some("pt"), "Portuguese");
        language_combo.append(Some("ru"), "Russian");
        language_combo.append(Some("ja"), "Japanese");
        language_combo.append(Some("zh"), "Chinese");
        language_combo.set_active_id(Some(&self.config.speech.language));
        if language_combo.active_id().is_none() {
            language_combo.set_active_id(Some("auto"));
        }
        engine_grid.attach(&language_combo, 1, engine_row, 1, 1);
        engine_row += 1;

        engine_grid.attach(&Label::new(Some("Processing Threads")), 0, engine_row, 1, 1);
        let threads_spin = SpinButton::with_range(1.0, 32.0, 1.0);
        threads_spin.set_value(self.config.speech.num_threads as f64);
        engine_grid.attach(&threads_spin, 1, engine_row, 1, 1);
        engine_row += 1;

        engine_grid.attach(
            &Label::new(Some("Use GPU Acceleration")),
            0,
            engine_row,
            1,
            1,
        );
        let gpu_switch = Switch::new();
        gpu_switch.set_active(self.config.speech.use_gpu);
        engine_grid.attach(&gpu_switch, 1, engine_row, 1, 1);
        engine_row += 1;

        engine_grid.attach(&Label::new(Some("Max Tokens")), 0, engine_row, 1, 1);
        let tokens_spin = SpinButton::with_range(64.0, 2048.0, 64.0);
        tokens_spin.set_value(self.config.speech.max_tokens as f64);
        engine_grid.attach(&tokens_spin, 1, engine_row, 1, 1);
        engine_row += 1;

        engine_grid.attach(&Label::new(Some("Beam Size")), 0, engine_row, 1, 1);
        let beam_spin = SpinButton::with_range(1.0, 10.0, 1.0);
        beam_spin.set_value(self.config.speech.beam_size as f64);
        engine_grid.attach(&beam_spin, 1, engine_row, 1, 1);
        engine_row += 1;

        engine_grid.attach(&Label::new(Some("Voice Calibration")), 0, engine_row, 1, 1);
        let calibrate_button = Button::with_label("Start 3-second Calibration");
        calibrate_button.connect_clicked(|_| {
            tracing::info!("Calibration requested by user; capture workflow should start here");
        });
        engine_grid.attach(&calibrate_button, 1, engine_row, 1, 1);

        Ok((
            page,
            model_combo,
            model_chooser,
            language_combo,
            threads_spin,
            gpu_switch,
            tokens_spin,
            beam_spin,
        ))
    }

    /// Create GUI settings page.
    fn create_gui_settings(
        &self,
    ) -> Result<(GtkBox, ComboBoxText, Switch, Switch, SpinButton, SpinButton)> {
        let page = GtkBox::new(Orientation::Vertical, 10);
        page.set_margin_top(8);
        page.set_margin_bottom(8);
        page.set_margin_start(8);
        page.set_margin_end(8);

        let (frame, grid) = Self::create_section("Look & Feel");
        page.append(&frame);

        let mut row = 0;

        grid.attach(&Label::new(Some("Theme")), 0, row, 1, 1);
        let theme_combo = ComboBoxText::new();
        theme_combo.append(Some("system"), "System");
        theme_combo.append(Some("light"), "Light");
        theme_combo.append(Some("dark"), "Dark");
        theme_combo.set_active_id(Some(&self.config.gui.theme));
        if theme_combo.active_id().is_none() {
            theme_combo.set_active_id(Some("system"));
        }
        grid.attach(&theme_combo, 1, row, 1, 1);
        row += 1;

        grid.attach(&Label::new(Some("Show Waveform")), 0, row, 1, 1);
        let waveform_switch = Switch::new();
        waveform_switch.set_active(self.config.gui.show_waveform);
        grid.attach(&waveform_switch, 1, row, 1, 1);
        row += 1;

        grid.attach(&Label::new(Some("Auto Scroll")), 0, row, 1, 1);
        let scroll_switch = Switch::new();
        scroll_switch.set_active(self.config.gui.auto_scroll);
        grid.attach(&scroll_switch, 1, row, 1, 1);
        row += 1;

        grid.attach(&Label::new(Some("Window Width")), 0, row, 1, 1);
        let width_spin = SpinButton::with_range(400.0, 3840.0, 10.0);
        width_spin.set_value(self.config.gui.window_width as f64);
        grid.attach(&width_spin, 1, row, 1, 1);
        row += 1;

        grid.attach(&Label::new(Some("Window Height")), 0, row, 1, 1);
        let height_spin = SpinButton::with_range(300.0, 2160.0, 10.0);
        height_spin.set_value(self.config.gui.window_height as f64);
        grid.attach(&height_spin, 1, row, 1, 1);

        Ok((
            page,
            theme_combo,
            waveform_switch,
            scroll_switch,
            width_spin,
            height_spin,
        ))
    }

    /// Create hotkey settings page.
    fn create_hotkey_settings(&self) -> Result<(GtkBox, Entry, Entry)> {
        let page = GtkBox::new(Orientation::Vertical, 10);
        page.set_margin_top(8);
        page.set_margin_bottom(8);
        page.set_margin_start(8);
        page.set_margin_end(8);

        let (frame, grid) = Self::create_section("Keyboard Shortcuts");
        page.append(&frame);

        grid.attach(&Label::new(Some("Toggle Recording")), 0, 0, 1, 1);
        let toggle_entry = Entry::new();
        toggle_entry.set_text(&self.config.hotkeys.toggle_recording);
        grid.attach(&toggle_entry, 1, 0, 1, 1);

        grid.attach(&Label::new(Some("Push-to-Talk")), 0, 1, 1, 1);
        let ptt_entry = Entry::new();
        ptt_entry.set_text(&self.config.hotkeys.push_to_talk);
        grid.attach(&ptt_entry, 1, 1, 1, 1);

        Ok((page, toggle_entry, ptt_entry))
    }

    /// Show the settings dialog.
    pub fn show(&self) {
        self.dialog.present();
    }
}
