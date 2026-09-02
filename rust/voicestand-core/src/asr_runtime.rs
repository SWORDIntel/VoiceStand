use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use tokio::task;
use voicestand_asr::{
    CancellationToken, DecodeOptions, ModelSpec, SpeechBackend, Transcript, WhisperCppBackend,
};
use voicestand_types::{Result, SpeechConfig, VoiceStandError};

/// Resident speech backend executed away from the asynchronous control loop.
#[derive(Clone)]
pub struct AsrRuntime {
    backend: Arc<Mutex<Box<dyn SpeechBackend>>>,
    options: DecodeOptions,
    active: Arc<Mutex<Option<CancellationToken>>>,
    metrics: Arc<AsrMetricCounters>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AsrMetrics {
    pub started: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub average_latency_ms: f64,
}

#[derive(Default)]
struct AsrMetricCounters {
    started: AtomicU64,
    completed: AtomicU64,
    cancelled: AtomicU64,
    failed: AtomicU64,
    total_latency_us: AtomicU64,
}

impl AsrRuntime {
    /// Construct and load the configured production backend.
    pub fn from_config(config: &SpeechConfig) -> Result<Self> {
        let mut backend = WhisperCppBackend::with_gpu(config.use_gpu);
        let model_id = Path::new(&config.model_path)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("configured-model");
        backend.load(&ModelSpec::new(model_id, &config.model_path))?;

        Self::from_loaded_backend(Box::new(backend), Self::decode_options(config))
    }

    /// Construct a runtime around an already-loaded backend.
    ///
    /// This is also the injection point used by deterministic orchestration tests.
    pub fn from_loaded_backend(
        backend: Box<dyn SpeechBackend>,
        options: DecodeOptions,
    ) -> Result<Self> {
        options.validate()?;
        if !backend.is_loaded() {
            return Err(VoiceStandError::speech(format!(
                "{} backend must be loaded before runtime creation",
                backend.name()
            )));
        }

        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
            options,
            active: Arc::new(Mutex::new(None)),
            metrics: Arc::new(AsrMetricCounters::default()),
        })
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.lock().name()
    }

    pub fn metrics(&self) -> AsrMetrics {
        let completed = self.metrics.completed.load(Ordering::Relaxed);
        AsrMetrics {
            started: self.metrics.started.load(Ordering::Relaxed),
            completed,
            cancelled: self.metrics.cancelled.load(Ordering::Relaxed),
            failed: self.metrics.failed.load(Ordering::Relaxed),
            average_latency_ms: if completed == 0 {
                0.0
            } else {
                self.metrics.total_latency_us.load(Ordering::Relaxed) as f64
                    / completed as f64
                    / 1_000.0
            },
        }
    }

    /// Transcribe canonical 16 kHz mono PCM without blocking Tokio worker threads.
    pub async fn transcribe(&self, audio: Vec<f32>) -> Result<Transcript> {
        self.transcribe_latest(audio).await
    }

    /// Cancel an older decode and make this request the only publishable result.
    pub async fn transcribe_latest(&self, audio: Vec<f32>) -> Result<Transcript> {
        self.metrics.started.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let backend = Arc::clone(&self.backend);
        let options = self.options.clone();
        let cancellation = CancellationToken::default();
        if let Some(previous) = self.active.lock().replace(cancellation.clone()) {
            previous.cancel();
        }

        let worker_token = cancellation.clone();
        let result = task::spawn_blocking(move || {
            backend
                .lock()
                .transcribe_cancellable(&audio, &options, worker_token)
        })
        .await
        .map_err(|error| VoiceStandError::speech(format!("ASR worker task failed: {error}")))?;
        if cancellation.is_cancelled() {
            self.metrics.cancelled.fetch_add(1, Ordering::Relaxed);
            return Err(VoiceStandError::speech("transcription superseded"));
        }
        match result {
            Ok(transcript) => {
                self.metrics.completed.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .total_latency_us
                    .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                Ok(transcript)
            }
            Err(error) => {
                self.metrics.failed.fetch_add(1, Ordering::Relaxed);
                Err(error)
            }
        }
    }

    fn decode_options(config: &SpeechConfig) -> DecodeOptions {
        DecodeOptions {
            language: match config.language.trim() {
                "" | "auto" => None,
                language => Some(language.to_string()),
            },
            thread_count: config.num_threads,
            translate: false,
            emit_partials: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;

    struct MockBackend;

    impl SpeechBackend for MockBackend {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn load(&mut self, _model: &ModelSpec) -> Result<()> {
            Ok(())
        }

        fn is_loaded(&self) -> bool {
            true
        }

        fn transcribe(&mut self, audio: &[f32], _options: &DecodeOptions) -> Result<Transcript> {
            Transcript::final_result(
                "runtime transcript",
                Some(0.95),
                Duration::from_secs_f64(audio.len() as f64 / 16_000.0),
            )
        }
    }

    #[tokio::test]
    async fn runs_loaded_backend_on_blocking_worker() {
        let runtime =
            AsrRuntime::from_loaded_backend(Box::new(MockBackend), DecodeOptions::default())
                .expect("create runtime");

        let transcript = runtime
            .transcribe(vec![0.0; 1_600])
            .await
            .expect("transcribe audio");

        assert_eq!(runtime.backend_name(), "mock");
        assert_eq!(transcript.text, "runtime transcript");
        assert_eq!(transcript.audio_duration, Duration::from_millis(100));
        assert_eq!(runtime.metrics().started, 1);
        assert_eq!(runtime.metrics().completed, 1);
        assert_eq!(runtime.metrics().failed, 0);
    }

    #[test]
    fn rejects_unloaded_backend() {
        struct UnloadedBackend;
        impl SpeechBackend for UnloadedBackend {
            fn name(&self) -> &'static str {
                "unloaded"
            }
            fn load(&mut self, _model: &ModelSpec) -> Result<()> {
                Ok(())
            }
            fn is_loaded(&self) -> bool {
                false
            }
            fn transcribe(
                &mut self,
                _audio: &[f32],
                _options: &DecodeOptions,
            ) -> Result<Transcript> {
                unreachable!()
            }
        }

        assert!(AsrRuntime::from_loaded_backend(
            Box::new(UnloadedBackend),
            DecodeOptions::default()
        )
        .is_err());
    }

    #[tokio::test]
    async fn newer_decode_cancels_older_work() {
        struct CancellableBackend;
        impl SpeechBackend for CancellableBackend {
            fn name(&self) -> &'static str {
                "cancellable"
            }
            fn load(&mut self, _: &ModelSpec) -> Result<()> {
                Ok(())
            }
            fn is_loaded(&self) -> bool {
                true
            }
            fn transcribe(&mut self, _: &[f32], _: &DecodeOptions) -> Result<Transcript> {
                unreachable!()
            }
            fn transcribe_cancellable(
                &mut self,
                audio: &[f32],
                _: &DecodeOptions,
                token: CancellationToken,
            ) -> Result<Transcript> {
                for _ in 0..100 {
                    if token.is_cancelled() {
                        return Err(VoiceStandError::speech("cancelled"));
                    }
                    thread::sleep(Duration::from_millis(1));
                }
                Transcript::final_result(format!("{}", audio.len()), Some(1.0), Duration::ZERO)
            }
        }

        let runtime =
            AsrRuntime::from_loaded_backend(Box::new(CancellableBackend), DecodeOptions::default())
                .unwrap();
        let older_runtime = runtime.clone();
        let older =
            tokio::spawn(async move { older_runtime.transcribe_latest(vec![0.0; 10]).await });
        tokio::time::sleep(Duration::from_millis(5)).await;
        let newer = runtime.transcribe_latest(vec![0.0; 20]).await.unwrap();

        assert!(older.await.unwrap().is_err());
        assert_eq!(newer.text, "20");
        let metrics = runtime.metrics();
        assert_eq!(metrics.started, 2);
        assert_eq!(metrics.completed, 1);
        assert_eq!(metrics.cancelled, 1);
    }
}
