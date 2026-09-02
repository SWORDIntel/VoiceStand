use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use serde_json::json;
use voicestand_asr::{
    benchmark::{percentile, word_error_rate},
    read_wav_16khz_mono, DecodeOptions, ModelSpec, SherpaZipformerConfig, SpeechBackend,
    Transcript,
};
use voicestand_core::{
    AsrRuntime, IntegrationEvent, StreamingAsrRuntime, VoiceStandConfig, VoiceStandIntegration,
};
use voicestand_types::{Result, VoiceStandError};

const SAMPLE_RATE: usize = 16_000;

struct UnusedFallback;

impl SpeechBackend for UnusedFallback {
    fn name(&self) -> &'static str {
        "benchmark-fallback"
    }

    fn load(&mut self, _model: &ModelSpec) -> Result<()> {
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        true
    }

    fn transcribe(&mut self, _audio: &[f32], _options: &DecodeOptions) -> Result<Transcript> {
        Err(VoiceStandError::speech(
            "streaming benchmark unexpectedly used its fallback",
        ))
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("VoiceStand streaming PTT benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: streaming_ptt_benchmark MODEL_DIR WAV [ITERATIONS] [CHUNK_MS] [REFERENCE]";
    let model_dir = PathBuf::from(args.next().ok_or_else(|| VoiceStandError::config(usage))?);
    let wav_path = PathBuf::from(args.next().ok_or_else(|| VoiceStandError::config(usage))?);
    let iterations = parse_arg(args.next(), "iterations", 5usize)?;
    let chunk_ms = parse_arg(args.next(), "chunk milliseconds", 100usize)?;
    let reference = args
        .next()
        .map(|value| value.to_string_lossy().into_owned());
    if iterations == 0 || chunk_ms == 0 {
        return Err(VoiceStandError::config(
            "iterations and chunk milliseconds must be positive",
        ));
    }

    let audio = read_wav_16khz_mono(&wav_path)?;
    let chunk_samples = chunk_ms * SAMPLE_RATE / 1_000;
    let fallback =
        AsrRuntime::from_loaded_backend(Box::new(UnusedFallback), DecodeOptions::default())?;
    let mut model_config = SherpaZipformerConfig::new(&model_dir);
    model_config.thread_count = 2;
    let load_started = Instant::now();
    let streaming = StreamingAsrRuntime::load(&model_config)?;
    let model_load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
    let (integration, mut events) = VoiceStandIntegration::for_offline_benchmark(
        VoiceStandConfig::default(),
        fallback,
        streaming,
    )?;

    let mut first_partial_ms = Vec::with_capacity(iterations);
    let mut release_ms = Vec::with_capacity(iterations);
    let mut total_compute_ms = Vec::with_capacity(iterations);
    let mut transcripts = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        integration.begin_ptt()?;
        let iteration_started = Instant::now();
        let mut first_partial = None;
        for chunk in audio.chunks(chunk_samples) {
            integration.process_audio_frame(chunk).await?;
            while let Ok(event) = events.try_recv() {
                if matches!(event, IntegrationEvent::PartialTranscription { .. }) {
                    first_partial.get_or_insert(iteration_started.elapsed());
                }
            }
        }
        let release_started = Instant::now();
        let result = integration
            .end_ptt()
            .await?
            .ok_or_else(|| VoiceStandError::speech("PTT produced no final transcript"))?;
        release_ms.push(milliseconds(release_started.elapsed()));
        total_compute_ms.push(milliseconds(iteration_started.elapsed()));
        if let Some(latency) = first_partial {
            first_partial_ms.push(milliseconds(latency));
        }
        transcripts.push(result.text);
        while events.try_recv().is_ok() {}
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "metric_scope": "complete_rust_ptt_orchestration",
            "model_dir": model_dir,
            "wav": wav_path,
            "iterations": iterations,
            "chunk_ms": chunk_ms,
            "audio_seconds": audio.len() as f64 / SAMPLE_RATE as f64,
            "model_load_ms": model_load_ms,
            "first_partial_ms": summary(&first_partial_ms),
            "release_to_final_ms": summary(&release_ms),
            "total_compute_ms": summary(&total_compute_ms),
            "compute_rtf_p50": percentile(&total_compute_ms, 50.0)
                .map(|value| value / 1_000.0 / (audio.len() as f64 / SAMPLE_RATE as f64)),
            "reference": reference,
            "word_error_rates": reference.as_ref().map(|reference| transcripts
                .iter()
                .filter_map(|transcript| word_error_rate(reference, transcript))
                .collect::<Vec<_>>()),
            "transcripts": transcripts,
        }))?
    );
    Ok(())
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn summary(samples: &[f64]) -> serde_json::Value {
    json!({
        "p50": percentile(samples, 50.0),
        "p95": percentile(samples, 95.0),
        "samples": samples,
    })
}

fn parse_arg<T>(value: Option<std::ffi::OsString>, name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse()
                .map_err(|error| VoiceStandError::config(format!("invalid {name}: {error}")))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}
