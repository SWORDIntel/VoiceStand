pub mod window;
pub mod widgets;
pub mod settings;
pub mod waveform;

pub use window::*;
pub use widgets::*;
pub use settings::*;
pub use waveform::*;

use voicestand_core::{Result, VoiceStandError, AppEvent};
use gtk4::prelude::*;
use gtk4::{Application, ApplicationWindow};
use crossbeam_channel::Receiver;
use std::sync::Arc;

/// Initialize GTK4 GUI subsystem
pub fn initialize_gui() -> Result<Application> {
    tracing::info!("Initializing GUI subsystem");

    let app = Application::builder()
        .application_id("com.voicestand.app")
        .build();

    tracing::info!("GUI subsystem initialized successfully");
    Ok(app)
}

/// Main GUI application controller
pub struct GuiController {
    app: Application,
    main_window: Option<Arc<MainWindow>>,
    event_receiver: Option<Receiver<AppEvent>>,
}

impl GuiController {
    pub fn new(app: Application) -> Self {
        Self {
            app,
            main_window: None,
            event_receiver: None,
        }
    }

    /// Set event receiver for handling application events
    pub fn set_event_receiver(&mut self, receiver: Receiver<AppEvent>) {
        self.event_receiver = Some(receiver);
    }

    /// Run the GUI application
    pub fn run(&mut self, args: &[String]) -> Result<()> {
        let app = self.app.clone();

        app.connect_activate(move |app| {
            tracing::info!("GUI application activated");
        });

        let exit_code = app.run_with_args(args);
        if exit_code != 0 {
            return Err(VoiceStandError::gui(format!("Application exited with code: {}", exit_code)));
        }

        Ok(())
    }

    /// Create and show main window
    pub fn create_main_window(&mut self) -> Result<Arc<MainWindow>> {
        let window = Arc::new(MainWindow::new(&self.app)?);
        self.main_window = Some(window.clone());

        // Setup event handling if receiver is available
        if let Some(receiver) = &self.event_receiver {
            self.setup_event_handling(window.clone(), receiver.clone())?;
        }

        window.show();
        Ok(window)
    }

    /// Setup event handling between core and GUI
    fn setup_event_handling(&self, window: Arc<MainWindow>, receiver: Receiver<AppEvent>) -> Result<()> {
        // Clone for the closure
        let window_clone = window.clone();

        // Spawn event handling task
        glib::spawn_future_local(async move {
            while let Ok(event) = receiver.recv() {
                match event {
                    AppEvent::TranscriptionReceived(result) => {
                        window_clone.update_transcription(&result.text, result.is_final, result.confidence);
                    }
                    AppEvent::AudioDataReceived(data) => {
                        window_clone.update_waveform(&data.samples);
                    }
                    AppEvent::RecordingStateChanged(recording) => {
                        window_clone.update_recording_status(recording);
                    }
                    AppEvent::Error(message) => {
                        window_clone.show_error(&message);
                    }
                    AppEvent::GuiEvent(gui_event) => {
                        window_clone.handle_gui_event(gui_event);
                    }
                    AppEvent::Shutdown => {
                        window_clone.close();
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }

    /// Get main window reference
    pub fn main_window(&self) -> Option<Arc<MainWindow>> {
        self.main_window.clone()
    }

    /// Shutdown GUI
    pub fn shutdown(&self) {
        if let Some(window) = &self.main_window {
            window.close();
        }
        self.app.quit();
    }
}

/// GUI event handler trait
pub trait EventHandler {
    fn handle_recording_toggle(&self);
    fn handle_settings_open(&self);
    fn handle_export_transcription(&self, text: &str);
    fn handle_clear_transcription(&self);
}

/// Default event handler implementation
pub struct DefaultEventHandler;

impl EventHandler for DefaultEventHandler {
    fn handle_recording_toggle(&self) {
        tracing::info!("Recording toggle requested");
    }

    fn handle_settings_open(&self) {
        tracing::info!("Settings dialog requested");
    }

    fn handle_export_transcription(&self, text: &str) {
        tracing::info!("Export transcription requested: {} chars", text.len());
    }

    fn handle_clear_transcription(&self) {
        tracing::info!("Clear transcription requested");
    }
}