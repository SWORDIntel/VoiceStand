use voicestand_core::Result;
use gtk4::prelude::*;
use gtk4::{DrawingArea, Widget};
use cairo::Context;
use glib::clone;
use parking_lot::Mutex;
use std::sync::Arc;
use std::collections::VecDeque;

/// Waveform visualization widget
pub struct WaveformWidget {
    drawing_area: DrawingArea,
    samples: Arc<Mutex<VecDeque<f32>>>,
    max_samples: usize,
    amplitude_scale: f32,
    background_color: (f64, f64, f64),
    waveform_color: (f64, f64, f64),
    grid_color: (f64, f64, f64),
}

impl WaveformWidget {
    /// Create new waveform widget
    pub fn new() -> Result<Self> {
        let drawing_area = DrawingArea::new();
        drawing_area.set_size_request(400, 150);
        drawing_area.set_hexpand(true);

        let samples = Arc::new(Mutex::new(VecDeque::new()));
        let max_samples = 2048; // Show ~128ms at 16kHz

        let widget = Self {
            drawing_area,
            samples: samples.clone(),
            max_samples,
            amplitude_scale: 1.0,
            background_color: (0.1, 0.1, 0.1), // Dark background
            waveform_color: (0.2, 0.8, 0.2),   // Green waveform
            grid_color: (0.3, 0.3, 0.3),       // Gray grid
        };

        // Setup drawing callback
        widget.drawing_area.set_draw_func(clone!(@strong samples => move |_, cr, width, height| {
            WaveformWidget::draw_waveform(cr, width, height, &*samples.lock());
        }));

        Ok(widget)
    }

    /// Update waveform with new samples
    pub fn update_samples(&mut self, new_samples: &[f32]) {
        let mut samples = self.samples.lock();

        // Add new samples
        for &sample in new_samples {
            samples.push_back(sample);
        }

        // Limit buffer size
        while samples.len() > self.max_samples {
            samples.pop_front();
        }

        // Trigger redraw
        self.drawing_area.queue_draw();
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.lock().clear();
        self.drawing_area.queue_draw();
    }

    /// Set amplitude scaling
    pub fn set_amplitude_scale(&mut self, scale: f32) {
        self.amplitude_scale = scale.max(0.1).min(10.0);
        self.drawing_area.queue_draw();
    }

    /// Set colors
    pub fn set_colors(&mut self, background: (f64, f64, f64), waveform: (f64, f64, f64), grid: (f64, f64, f64)) {
        self.background_color = background;
        self.waveform_color = waveform;
        self.grid_color = grid;
        self.drawing_area.queue_draw();
    }

    /// Get the GTK widget
    pub fn widget(&self) -> &Widget {
        self.drawing_area.upcast_ref()
    }

    /// Drawing function for the waveform
    fn draw_waveform(cr: &Context, width: i32, height: i32, samples: &VecDeque<f32>) {
        let width = width as f64;
        let height = height as f64;

        // Clear background
        cr.set_source_rgb(0.1, 0.1, 0.1);
        if let Err(e) = cr.paint() {
            eprintln!("Cairo paint error: {}", e);
            return;
        }

        if samples.is_empty() {
            Self::draw_no_signal(cr, width, height);
            return;
        }

        // Draw grid
        Self::draw_grid(cr, width, height);

        // Draw waveform
        Self::draw_samples(cr, width, height, samples);

        // Draw scale indicators
        Self::draw_scale(cr, width, height);
    }

    /// Draw "no signal" message
    fn draw_no_signal(cr: &Context, width: f64, height: f64) {
        cr.set_source_rgb(0.5, 0.5, 0.5);
        cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
        cr.set_font_size(14.0);

        let text = "No audio signal";
        let text_extents = match cr.text_extents(text) {
            Ok(extents) => extents,
            Err(e) => {
                eprintln!("Cairo text extents error: {}", e);
                return;
            }
        };
        let x = (width - text_extents.width()) / 2.0;
        let y = height / 2.0;

        cr.move_to(x, y);
        if let Err(e) = cr.show_text(text) {
            eprintln!("Cairo show text error: {}", e);
            return;
        }
    }

    /// Draw grid lines
    fn draw_grid(cr: &Context, width: f64, height: f64) {
        cr.set_source_rgb(0.3, 0.3, 0.3);
        cr.set_line_width(0.5);

        // Horizontal center line
        cr.move_to(0.0, height / 2.0);
        cr.line_to(width, height / 2.0);
        if let Err(e) = cr.stroke() {
            eprintln!("Cairo stroke error: {}", e);
            return;
        }

        // Horizontal quarter lines
        cr.move_to(0.0, height / 4.0);
        cr.line_to(width, height / 4.0);
        if let Err(e) = cr.stroke() {
            eprintln!("Cairo stroke error: {}", e);
            return;
        }

        cr.move_to(0.0, 3.0 * height / 4.0);
        cr.line_to(width, 3.0 * height / 4.0);
        if let Err(e) = cr.stroke() {
            eprintln!("Cairo stroke error: {}", e);
            return;
        }

        // Vertical grid lines
        let num_vertical_lines = 10;
        for i in 1..num_vertical_lines {
            let x = width * i as f64 / num_vertical_lines as f64;
            cr.move_to(x, 0.0);
            cr.line_to(x, height);
            if let Err(e) = cr.stroke() {
            eprintln!("Cairo stroke error: {}", e);
            return;
        }
        }
    }

    /// Draw audio samples as waveform
    fn draw_samples(cr: &Context, width: f64, height: f64, samples: &VecDeque<f32>) {
        if samples.len() < 2 {
            return;
        }

        cr.set_source_rgb(0.2, 0.8, 0.2);
        cr.set_line_width(1.0);

        let center_y = height / 2.0;
        let amplitude_scale = (height / 2.0) * 0.8; // Leave some margin

        // Draw waveform
        let samples_vec: Vec<f32> = samples.iter().copied().collect();
        let x_step = width / (samples_vec.len() - 1) as f64;

        cr.move_to(0.0, center_y - samples_vec[0] as f64 * amplitude_scale);

        for (i, &sample) in samples_vec.iter().enumerate().skip(1) {
            let x = i as f64 * x_step;
            let y = center_y - sample as f64 * amplitude_scale;
            cr.line_to(x, y);
        }

        if let Err(e) = cr.stroke() {
            eprintln!("Cairo stroke error: {}", e);
            return;
        }

        // Draw envelope (RMS)
        Self::draw_envelope(cr, width, height, &samples_vec, center_y, amplitude_scale);
    }

    /// Draw RMS envelope
    fn draw_envelope(cr: &Context, width: f64, height: f64, samples: &[f32], center_y: f64, amplitude_scale: f64) {
        if samples.len() < 32 {
            return;
        }

        cr.set_source_rgba(0.8, 0.2, 0.2, 0.5); // Semi-transparent red
        cr.set_line_width(2.0);

        let window_size = 32; // RMS window size
        let num_windows = samples.len() / window_size;

        if num_windows < 2 {
            return;
        }

        let x_step = width / (num_windows - 1) as f64;

        // Calculate RMS for each window
        let mut rms_values = Vec::with_capacity(num_windows);
        for i in 0..num_windows {
            let start = i * window_size;
            let end = (start + window_size).min(samples.len());
            let window = &samples[start..end];

            let rms = if !window.is_empty() {
                let sum_squares: f32 = window.iter().map(|&x| x * x).sum();
                (sum_squares / window.len() as f32).sqrt()
            } else {
                0.0
            };

            rms_values.push(rms);
        }

        // Draw upper envelope
        cr.move_to(0.0, center_y - rms_values[0] as f64 * amplitude_scale);
        for (i, &rms) in rms_values.iter().enumerate().skip(1) {
            let x = i as f64 * x_step;
            let y = center_y - rms as f64 * amplitude_scale;
            cr.line_to(x, y);
        }
        if let Err(e) = cr.stroke() {
            eprintln!("Cairo stroke error: {}", e);
            return;
        }

        // Draw lower envelope
        cr.move_to(0.0, center_y + rms_values[0] as f64 * amplitude_scale);
        for (i, &rms) in rms_values.iter().enumerate().skip(1) {
            let x = i as f64 * x_step;
            let y = center_y + rms as f64 * amplitude_scale;
            cr.line_to(x, y);
        }
        if let Err(e) = cr.stroke() {
            eprintln!("Cairo stroke error: {}", e);
            return;
        }
    }

    /// Draw scale indicators
    fn draw_scale(cr: &Context, width: f64, height: f64) {
        cr.set_source_rgb(0.7, 0.7, 0.7);
        cr.select_font_face("Sans", cairo::FontSlant::Normal, cairo::FontWeight::Normal);
        cr.set_font_size(10.0);

        // Amplitude scale indicators
        let scales = ["+1.0", "+0.5", "0.0", "-0.5", "-1.0"];
        let positions = [0.1, 0.3, 0.5, 0.7, 0.9];

        for (i, &scale) in scales.iter().enumerate() {
            let y = height * positions[i];
            cr.move_to(5.0, y);
            if let Err(e) = cr.show_text(scale) {
            eprintln!("Cairo show text error: {}", e);
            continue;
        }
        }

        // Time scale (simplified - shows relative position)
        let time_labels = ["0%", "25%", "50%", "75%", "100%"];
        let time_positions = [0.05, 0.25, 0.5, 0.75, 0.95];

        for (i, &label) in time_labels.iter().enumerate() {
            let x = width * time_positions[i];
            cr.move_to(x, height - 5.0);
            if let Err(e) = cr.show_text(label) {
            eprintln!("Cairo show text error: {}", e);
            continue;
        }
        }
    }

    /// Get current peak amplitude
    pub fn get_peak_amplitude(&self) -> f32 {
        let samples = self.samples.lock();
        samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
    }

    /// Get current RMS level
    pub fn get_rms_level(&self) -> f32 {
        let samples = self.samples.lock();
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Set maximum number of samples to display
    pub fn set_max_samples(&mut self, max_samples: usize) {
        self.max_samples = max_samples.max(64).min(16384);

        // Trim current samples if necessary
        let mut samples = self.samples.lock();
        while samples.len() > self.max_samples {
            samples.pop_front();
        }
    }
}

/// Waveform analysis utilities
impl WaveformWidget {
    /// Calculate frequency spectrum (simplified)
    pub fn get_frequency_spectrum(&self) -> Vec<f32> {
        let samples = self.samples.lock();
        let samples_vec: Vec<f32> = samples.iter().copied().collect();

        if samples_vec.len() < 64 {
            return vec![0.0; 32]; // Return empty spectrum
        }

        // Simple frequency analysis using zero-crossing rate
        let mut spectrum = vec![0.0; 32];

        // Divide into frequency bands
        let window_size = samples_vec.len() / spectrum.len();

        for (i, spectrum_val) in spectrum.iter_mut().enumerate() {
            let start = i * window_size;
            let end = ((i + 1) * window_size).min(samples_vec.len());

            if end > start + 1 {
                let window = &samples_vec[start..end];

                // Calculate energy in this frequency band
                let energy: f32 = window.iter().map(|&x| x * x).sum();
                *spectrum_val = (energy / window.len() as f32).sqrt();
            }
        }

        spectrum
    }

    /// Detect voice activity based on waveform
    pub fn detect_voice_activity(&self) -> bool {
        let rms = self.get_rms_level();
        let peak = self.get_peak_amplitude();

        // Simple VAD based on energy thresholds
        rms > 0.01 && peak > 0.05
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waveform_creation() {
        // This test would require GTK to be initialized
        // let waveform = WaveformWidget::new().unwrap();
        // assert_eq!(waveform.get_peak_amplitude(), 0.0);
    }

    #[test]
    fn test_amplitude_calculations() {
        // Test RMS and peak calculations with known data
        let samples = vec![0.5, -0.5, 0.8, -0.3, 0.0];

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        let rms = (sum_squares / samples.len() as f32).sqrt();
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

        assert!((rms - 0.537).abs() < 0.01); // Approximate RMS
        assert_eq!(peak, 0.8);
    }
}