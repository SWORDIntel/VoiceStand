use voicestand_core::Result;
use gtk4::prelude::*;
use gtk4::{
    Widget, Box as GtkBox, Label, Button, ToggleButton, ProgressBar,
    Entry, TextView, TextBuffer, ScrolledWindow, Orientation, Align
};

/// Custom widget implementations for VoiceStand GUI

/// Status indicator widget
pub struct StatusIndicator {
    container: GtkBox,
    status_label: Label,
    activity_indicator: ProgressBar,
}

impl StatusIndicator {
    pub fn new() -> Result<Self> {
        let container = GtkBox::new(Orientation::Horizontal, 6);

        let status_label = Label::new(Some("Ready"));
        status_label.set_halign(Align::Start);
        container.append(&status_label);

        let activity_indicator = ProgressBar::new();
        activity_indicator.set_size_request(100, -1);
        activity_indicator.set_show_text(false);
        container.append(&activity_indicator);

        Ok(Self {
            container,
            status_label,
            activity_indicator,
        })
    }

    pub fn set_status(&self, status: &str) {
        self.status_label.set_text(status);
    }

    pub fn set_activity(&self, active: bool) {
        if active {
            self.activity_indicator.pulse();
        } else {
            self.activity_indicator.set_fraction(0.0);
        }
    }

    pub fn set_progress(&self, fraction: f64) {
        self.activity_indicator.set_fraction(fraction);
    }

    pub fn widget(&self) -> &Widget {
        self.container.upcast_ref()
    }
}

/// Recording control widget
pub struct RecordingControl {
    container: GtkBox,
    record_button: ToggleButton,
    status_label: Label,
}

impl RecordingControl {
    pub fn new() -> Result<Self> {
        let container = GtkBox::new(Orientation::Horizontal, 12);

        let record_button = ToggleButton::with_label("🎤 Start Recording");
        record_button.set_tooltip_text(Some("Toggle recording"));
        container.append(&record_button);

        let status_label = Label::new(Some("Ready"));
        status_label.set_halign(Align::Start);
        status_label.set_hexpand(true);
        container.append(&status_label);

        Ok(Self {
            container,
            record_button,
            status_label,
        })
    }

    pub fn set_recording(&self, recording: bool) {
        self.record_button.set_active(recording);
        if recording {
            self.record_button.set_label("🛑 Stop Recording");
            self.status_label.set_text("Recording...");
        } else {
            self.record_button.set_label("🎤 Start Recording");
            self.status_label.set_text("Ready");
        }
    }

    pub fn connect_toggled<F>(&self, callback: F)
    where
        F: Fn(bool) + 'static,
    {
        self.record_button.connect_toggled(move |button| {
            callback(button.is_active());
        });
    }

    pub fn widget(&self) -> &Widget {
        self.container.upcast_ref()
    }
}

/// Transcription display widget
pub struct TranscriptionDisplay {
    container: ScrolledWindow,
    text_view: TextView,
    text_buffer: TextBuffer,
}

impl TranscriptionDisplay {
    pub fn new() -> Result<Self> {
        let container = ScrolledWindow::new();
        container.set_policy(gtk4::PolicyType::Automatic, gtk4::PolicyType::Automatic);
        container.set_vexpand(true);

        let text_view = TextView::new();
        text_view.set_editable(true);
        text_view.set_wrap_mode(gtk4::WrapMode::Word);
        text_view.set_margin_top(12);
        text_view.set_margin_bottom(12);
        text_view.set_margin_start(12);
        text_view.set_margin_end(12);

        // Set monospace font
        let font_desc = pango::FontDescription::from_string("Monospace 11");
        text_view.override_font(Some(&font_desc));

        let text_buffer = text_view.buffer();
        container.set_child(Some(&text_view));

        Ok(Self {
            container,
            text_view,
            text_buffer,
        })
    }

    pub fn append_text(&self, text: &str) {
        let mut end_iter = self.text_buffer.end_iter();
        self.text_buffer.insert(&mut end_iter, text);

        // Auto-scroll to bottom
        let mark = self.text_buffer.insert_mark(None, &end_iter, false);
        self.text_view.scroll_mark_onscreen(&mark);
    }

    pub fn append_line(&self, text: &str) {
        self.append_text(&format!("{}\n", text));
    }

    pub fn clear(&self) {
        self.text_buffer.set_text("");
    }

    pub fn get_text(&self) -> String {
        self.text_buffer.text(&self.text_buffer.start_iter(), &self.text_buffer.end_iter(), false).to_string()
    }

    pub fn set_text(&self, text: &str) {
        self.text_buffer.set_text(text);
    }

    pub fn widget(&self) -> &Widget {
        self.container.upcast_ref()
    }
}

/// Audio level meter widget
pub struct AudioLevelMeter {
    container: GtkBox,
    level_bar: ProgressBar,
    peak_label: Label,
    rms_label: Label,
}

impl AudioLevelMeter {
    pub fn new() -> Result<Self> {
        let container = GtkBox::new(Orientation::Vertical, 6);

        let level_bar = ProgressBar::new();
        level_bar.set_orientation(Orientation::Horizontal);
        level_bar.set_show_text(true);
        level_bar.set_text(Some("0 dB"));
        container.append(&level_bar);

        let info_box = GtkBox::new(Orientation::Horizontal, 12);
        container.append(&info_box);

        let peak_label = Label::new(Some("Peak: 0 dB"));
        peak_label.set_halign(Align::Start);
        info_box.append(&peak_label);

        let rms_label = Label::new(Some("RMS: 0 dB"));
        rms_label.set_halign(Align::End);
        rms_label.set_hexpand(true);
        info_box.append(&rms_label);

        Ok(Self {
            container,
            level_bar,
            peak_label,
            rms_label,
        })
    }

    pub fn update_levels(&self, rms: f32, peak: f32) {
        // Convert to dB (with floor at -60 dB)
        let rms_db = if rms > 0.0 {
            20.0 * rms.log10().max(-60.0)
        } else {
            -60.0
        };

        let peak_db = if peak > 0.0 {
            20.0 * peak.log10().max(-60.0)
        } else {
            -60.0
        };

        // Update progress bar (normalize -60 to 0 dB to 0.0 to 1.0)
        let normalized = ((rms_db + 60.0) / 60.0).clamp(0.0, 1.0);
        self.level_bar.set_fraction(normalized as f64);
        self.level_bar.set_text(Some(&format!("{:.1} dB", rms_db)));

        // Update labels
        self.peak_label.set_text(&format!("Peak: {:.1} dB", peak_db));
        self.rms_label.set_text(&format!("RMS: {:.1} dB", rms_db));

        // Color coding
        if peak_db > -6.0 {
            // Red zone (clipping risk)
            self.level_bar.add_css_class("level-critical");
            self.level_bar.remove_css_class("level-warning");
            self.level_bar.remove_css_class("level-normal");
        } else if peak_db > -12.0 {
            // Yellow zone (loud)
            self.level_bar.add_css_class("level-warning");
            self.level_bar.remove_css_class("level-critical");
            self.level_bar.remove_css_class("level-normal");
        } else {
            // Green zone (normal)
            self.level_bar.add_css_class("level-normal");
            self.level_bar.remove_css_class("level-critical");
            self.level_bar.remove_css_class("level-warning");
        }
    }

    pub fn widget(&self) -> &Widget {
        self.container.upcast_ref()
    }
}

/// Quick action toolbar
pub struct QuickActionToolbar {
    container: GtkBox,
    clear_button: Button,
    export_button: Button,
    settings_button: Button,
}

impl QuickActionToolbar {
    pub fn new() -> Result<Self> {
        let container = GtkBox::new(Orientation::Horizontal, 6);

        let clear_button = Button::with_label("🗑️ Clear");
        clear_button.set_tooltip_text(Some("Clear transcription"));
        container.append(&clear_button);

        let export_button = Button::with_label("💾 Export");
        export_button.set_tooltip_text(Some("Export transcription"));
        container.append(&export_button);

        let settings_button = Button::with_label("⚙️ Settings");
        settings_button.set_tooltip_text(Some("Open settings"));
        container.append(&settings_button);

        Ok(Self {
            container,
            clear_button,
            export_button,
            settings_button,
        })
    }

    pub fn connect_clear<F>(&self, callback: F)
    where
        F: Fn() + 'static,
    {
        self.clear_button.connect_clicked(move |_| callback());
    }

    pub fn connect_export<F>(&self, callback: F)
    where
        F: Fn() + 'static,
    {
        self.export_button.connect_clicked(move |_| callback());
    }

    pub fn connect_settings<F>(&self, callback: F)
    where
        F: Fn() + 'static,
    {
        self.settings_button.connect_clicked(move |_| callback());
    }

    pub fn widget(&self) -> &Widget {
        self.container.upcast_ref()
    }
}

/// Search and replace widget for transcription editing
pub struct SearchReplace {
    container: GtkBox,
    search_entry: Entry,
    replace_entry: Entry,
    find_button: Button,
    replace_button: Button,
    replace_all_button: Button,
}

impl SearchReplace {
    pub fn new() -> Result<Self> {
        let container = GtkBox::new(Orientation::Horizontal, 6);

        let search_entry = Entry::new();
        search_entry.set_placeholder_text(Some("Search..."));
        search_entry.set_size_request(150, -1);
        container.append(&search_entry);

        let replace_entry = Entry::new();
        replace_entry.set_placeholder_text(Some("Replace with..."));
        replace_entry.set_size_request(150, -1);
        container.append(&replace_entry);

        let find_button = Button::with_label("Find");
        container.append(&find_button);

        let replace_button = Button::with_label("Replace");
        container.append(&replace_button);

        let replace_all_button = Button::with_label("Replace All");
        container.append(&replace_all_button);

        Ok(Self {
            container,
            search_entry,
            replace_entry,
            find_button,
            replace_button,
            replace_all_button,
        })
    }

    pub fn connect_find<F>(&self, callback: F)
    where
        F: Fn(&str) + 'static,
    {
        let search_entry = self.search_entry.clone();
        self.find_button.connect_clicked(move |_| {
            callback(&search_entry.text());
        });
    }

    pub fn connect_replace<F>(&self, callback: F)
    where
        F: Fn(&str, &str) + 'static,
    {
        let search_entry = self.search_entry.clone();
        let replace_entry = self.replace_entry.clone();
        self.replace_button.connect_clicked(move |_| {
            callback(&search_entry.text(), &replace_entry.text());
        });
    }

    pub fn connect_replace_all<F>(&self, callback: F)
    where
        F: Fn(&str, &str) + 'static,
    {
        let search_entry = self.search_entry.clone();
        let replace_entry = self.replace_entry.clone();
        self.replace_all_button.connect_clicked(move |_| {
            callback(&search_entry.text(), &replace_entry.text());
        });
    }

    pub fn widget(&self) -> &Widget {
        self.container.upcast_ref()
    }
}

/// CSS styles for custom widgets
pub const WIDGET_CSS: &str = r#"
.level-normal {
    color: #00ff00;
}

.level-warning {
    color: #ffff00;
}

.level-critical {
    color: #ff0000;
}

.recording-active {
    background-color: #ff4444;
    color: white;
}

.transcription-view {
    font-family: monospace;
    background-color: #2e2e2e;
    color: #ffffff;
}

.waveform-widget {
    border: 1px solid #666666;
    background-color: #1a1a1a;
}

.status-ready {
    color: #00aa00;
}

.status-recording {
    color: #ff6600;
}

.status-error {
    color: #ff0000;
}
"#;

/// Load custom CSS styles
pub fn load_css() -> Result<()> {
    let provider = gtk4::CssProvider::new();
    provider.load_from_data(WIDGET_CSS);

    gtk4::style_context_add_provider_for_display(
        &gtk4::gdk::Display::default().unwrap(),
        &provider,
        gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_creation() {
        // These tests would require GTK to be initialized
        // In a real test environment, you would need to set this up properly
    }
}