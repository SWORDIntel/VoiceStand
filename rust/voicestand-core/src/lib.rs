pub mod config;
pub mod error;
pub mod types;
pub mod events;

pub use config::*;
pub use error::*;
pub use types::*;
pub use events::*;

use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam_channel::{Receiver, Sender};

/// Core application state with thread-safe access
#[derive(Debug, Clone)]
pub struct AppState {
    pub config: Arc<RwLock<VoiceStandConfig>>,
    pub is_recording: Arc<parking_lot::Mutex<bool>>,
    pub event_sender: Sender<AppEvent>,
    pub event_receiver: Arc<parking_lot::Mutex<Receiver<AppEvent>>>,
}

impl AppState {
    pub fn new(config: VoiceStandConfig) -> Result<Self> {
        let (event_sender, event_receiver) = crossbeam_channel::unbounded();

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            is_recording: Arc::new(parking_lot::Mutex::new(false)),
            event_sender,
            event_receiver: Arc::new(parking_lot::Mutex::new(event_receiver)),
        })
    }

    pub fn is_recording(&self) -> bool {
        *self.is_recording.lock()
    }

    pub fn set_recording(&self, recording: bool) {
        *self.is_recording.lock() = recording;
        let _ = self.event_sender.send(AppEvent::RecordingStateChanged(recording));
    }

    pub fn send_event(&self, event: AppEvent) -> Result<()> {
        self.event_sender.send(event)
            .map_err(|_| VoiceStandError::EventSendFailed)?;
        Ok(())
    }
}