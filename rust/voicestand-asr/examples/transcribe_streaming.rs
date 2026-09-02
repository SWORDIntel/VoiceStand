use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use voicestand_asr::{read_wav_16khz_mono, SherpaZipformerBackend, SherpaZipformerConfig};
use voicestand_types::{Result, VoiceStandError};

const SAMPLE_RATE: usize = 16_000;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("VoiceStand streaming benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: transcribe_streaming MODEL_DIR WAV [THREADS] [PRE_ROLL_MS] [CHUNK_MS]";
    let model_dir = PathBuf::from(args.next().ok_or_else(|| VoiceStandError::config(usage))?);
    let wav_path = PathBuf::from(args.next().ok_or_else(|| VoiceStandError::config(usage))?);
    let thread_count =
        args.next()
            .map(|value| {
                value.to_string_lossy().parse::<usize>().map_err(|error| {
                    VoiceStandError::config(format!("invalid thread count: {error}"))
                })
            })
            .transpose()?
            .unwrap_or(2);
    let pre_roll_ms = args
        .next()
        .map(|value| {
            value.to_string_lossy().parse::<usize>().map_err(|error| {
                VoiceStandError::config(format!("invalid pre-roll milliseconds: {error}"))
            })
        })
        .transpose()?
        .unwrap_or(300);
    let chunk_ms = args
        .next()
        .map(|value| {
            value.to_string_lossy().parse::<usize>().map_err(|error| {
                VoiceStandError::config(format!("invalid chunk milliseconds: {error}"))
            })
        })
        .transpose()?
        .unwrap_or(100);
    if chunk_ms == 0 {
        return Err(VoiceStandError::config(
            "chunk milliseconds must be positive",
        ));
    }
    let chunk_samples = chunk_ms * SAMPLE_RATE / 1_000;

    let audio = read_wav_16khz_mono(&wav_path)?;
    let audio_seconds = audio.len() as f64 / SAMPLE_RATE as f64;
    let mut config = SherpaZipformerConfig::new(&model_dir);
    config.thread_count = thread_count;
    let load_started = Instant::now();
    let backend = SherpaZipformerBackend::load(&config)?;
    let model_load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
    let mut session = backend.start_session();
    let mut compute_ms = 0.0;
    let mut partial_updates = 0usize;
    let mut partial_texts = Vec::new();
    let mut first_partial_audio_ms = None;

    if pre_roll_ms > 0 {
        let pre_roll = vec![0.0; pre_roll_ms * SAMPLE_RATE / 1_000];
        let started = Instant::now();
        let _ = session.accept_audio(&pre_roll)?;
        compute_ms += started.elapsed().as_secs_f64() * 1_000.0;
    }

    for (index, chunk) in audio.chunks(chunk_samples).enumerate() {
        let started = Instant::now();
        let update = session.accept_audio(chunk)?;
        compute_ms += started.elapsed().as_secs_f64() * 1_000.0;
        if let Some(update) = update {
            partial_updates += 1;
            partial_texts.push(update.text);
            first_partial_audio_ms.get_or_insert(
                ((index * chunk_samples + chunk.len()) as f64 / SAMPLE_RATE as f64) * 1_000.0,
            );
        }
    }

    let flush_started = Instant::now();
    let final_result = session.finish()?;
    let final_flush_ms = flush_started.elapsed().as_secs_f64() * 1_000.0;
    let transcript = final_result.map(|result| result.text);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "metric_scope": "rust_streaming_asr_component",
            "backend": "sherpa-onnx",
            "model_dir": model_dir,
            "wav": wav_path,
            "threads": thread_count,
            "chunk_ms": chunk_ms,
            "pre_roll_ms": pre_roll_ms,
            "audio_seconds": audio_seconds,
            "model_load_ms": model_load_ms,
            "stream_compute_ms": compute_ms,
            "stream_compute_rtf": compute_ms / 1_000.0 / audio_seconds,
            "final_flush_ms": final_flush_ms,
            "first_partial_audio_ms": first_partial_audio_ms,
            "partial_updates": partial_updates,
            "partial_texts": partial_texts,
            "transcript": transcript,
        }))?
    );
    Ok(())
}
