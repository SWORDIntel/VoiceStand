pub mod asr_runtime;
pub mod config;
pub mod error;
pub mod events;
pub mod integration;
pub mod performance;
pub mod streaming_asr_runtime;
pub mod transcript_merge;
pub mod transcript_stabilizer;
pub mod types;

pub use asr_runtime::*;
pub use config::*;
pub use error::{AudioError, Result, VoiceStandError};
pub use events::*;
pub use integration::*;
pub use performance::*;
pub use streaming_asr_runtime::*;
pub use types::*;

use crossbeam_channel::{Receiver, Sender};
use parking_lot::RwLock;
use std::sync::Arc;

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
        let _ = self
            .event_sender
            .send(AppEvent::RecordingStateChanged(recording));
    }

    pub fn send_event(&self, event: AppEvent) -> Result<()> {
        self.event_sender
            .send(event)
            .map_err(|_| VoiceStandError::EventSendFailed)?;
        Ok(())
    }
}
