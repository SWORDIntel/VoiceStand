use std::sync::Arc;

use parking_lot::Mutex;
use tokio::task;
use voicestand_asr::{
    SherpaOnlineSession, SherpaZipformerBackend, SherpaZipformerConfig, Transcript,
};
use voicestand_types::{Result, VoiceStandError};

const PRE_ROLL_SAMPLES: usize = 16_000;

/// Resident streaming recognizer with exactly one ordered utterance session.
#[derive(Clone)]
pub struct StreamingAsrRuntime {
    backend: Arc<SherpaZipformerBackend>,
    session: Arc<Mutex<Option<SherpaOnlineSession>>>,
}

impl StreamingAsrRuntime {
    pub fn load(config: &SherpaZipformerConfig) -> Result<Self> {
        Ok(Self {
            backend: Arc::new(SherpaZipformerBackend::load(config)?),
            session: Arc::new(Mutex::new(None)),
        })
    }

    /// Start a fresh utterance and prime the tightly cropped streaming model.
    pub fn begin(&self) {
        let mut session = self.backend.start_session();
        // The upstream model clips initial phonemes without left context. Silence
        // is fed into the recognizer only; captured utterance audio is unchanged.
        let pre_roll = vec![0.0; PRE_ROLL_SAMPLES];
        let pre_roll_result = session.accept_audio(&pre_roll);
        debug_assert!(pre_roll_result.is_ok());
        *self.session.lock() = Some(session);
    }

    pub async fn accept_audio(&self, audio: Vec<f32>) -> Result<Option<Transcript>> {
        let session = Arc::clone(&self.session);
        task::spawn_blocking(move || {
            session
                .lock()
                .as_mut()
                .ok_or_else(|| VoiceStandError::speech("streaming ASR session is not active"))?
                .accept_audio(&audio)
        })
        .await
        .map_err(|error| VoiceStandError::speech(format!("streaming ASR worker failed: {error}")))?
    }

    pub async fn finish(&self) -> Result<Option<Transcript>> {
        let session = Arc::clone(&self.session);
        task::spawn_blocking(move || {
            let mut active = session
                .lock()
                .take()
                .ok_or_else(|| VoiceStandError::speech("streaming ASR session is not active"))?;
            active.finish()
        })
        .await
        .map_err(|error| VoiceStandError::speech(format!("streaming ASR worker failed: {error}")))?
    }

    pub fn cancel(&self) {
        self.session.lock().take();
    }
}
