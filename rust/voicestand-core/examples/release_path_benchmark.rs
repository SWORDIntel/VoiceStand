use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use voicestand_asr::{
    benchmark::percentile, read_wav_16khz_mono, DecodeOptions, ModelSpec, SpeechBackend,
    WhisperCppBackend,
};
use voicestand_core::transcript_merge::merge_word_overlap;
use voicestand_types::{Result, VoiceStandError};

const SAMPLE_RATE: usize = 16_000;
const OVERLAP_SAMPLES: usize = SAMPLE_RATE;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("VoiceStand release-path benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: release_path_benchmark MODEL WAV [THREADS] [ITERATIONS] [UNCOVERED_MS]";
    let model_path = PathBuf::from(args.next().ok_or_else(|| VoiceStandError::config(usage))?);
    let wav_path = PathBuf::from(args.next().ok_or_else(|| VoiceStandError::config(usage))?);
    let threads = parse_arg(args.next(), "thread count", 4usize)?;
    let iterations = parse_arg(args.next(), "iteration count", 1usize)?;
    let uncovered_ms = parse_arg(args.next(), "uncovered milliseconds", 1_000usize)?;
    if iterations == 0 || uncovered_ms == 0 || uncovered_ms > 2_000 {
        return Err(VoiceStandError::config(
            "iterations must be positive and uncovered milliseconds must be 1..=2000",
        ));
    }

    let audio = read_wav_16khz_mono(&wav_path)?;
    let uncovered_samples = uncovered_ms * SAMPLE_RATE / 1_000;
    if audio.len() <= uncovered_samples + OVERLAP_SAMPLES {
        return Err(VoiceStandError::config(
            "WAV must be longer than the uncovered tail plus one-second overlap",
        ));
    }
    let covered_samples = audio.len() - uncovered_samples;
    let tail_start = covered_samples - OVERLAP_SAMPLES;
    let options = DecodeOptions {
        thread_count: threads,
        ..DecodeOptions::default()
    };
    let mut backend = WhisperCppBackend::cpu();
    backend.load(&ModelSpec::new("release-path-benchmark", &model_path))?;

    // Prepare the state that would already exist when the user releases PTT.
    let cached = backend.transcribe(&audio[..covered_samples], &options)?;
    let mut exact_ms = Vec::with_capacity(iterations);
    let mut tail_ms = Vec::with_capacity(iterations);
    let mut fallback_ms = Vec::with_capacity(iterations);
    let mut merged_text = None;
    let mut fallback_text = None;
    for _ in 0..iterations {
        let started = Instant::now();
        let _exact = cached.clone();
        exact_ms.push(started.elapsed().as_secs_f64() * 1_000.0);

        let started = Instant::now();
        let tail = backend.transcribe(&audio[tail_start..], &options)?;
        tail_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        merged_text = merge_word_overlap(&cached.text, &tail.text);

        let started = Instant::now();
        let fallback = backend.transcribe(&audio, &options)?;
        fallback_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
        fallback_text = Some(fallback.text);
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "metric_scope": "ptt_release_to_transcript_component",
            "backend": backend.name(),
            "model": model_path,
            "wav": wav_path,
            "threads": threads,
            "measured_iterations": iterations,
            "audio_seconds": audio.len() as f64 / SAMPLE_RATE as f64,
            "covered_seconds": covered_samples as f64 / SAMPLE_RATE as f64,
            "uncovered_ms": uncovered_ms,
            "overlap_ms": 1_000,
            "exact_cache_ms": summary(&exact_ms),
            "bounded_tail_ms": summary(&tail_ms),
            "full_fallback_ms": summary(&fallback_ms),
            "tail_merge_succeeded": merged_text.is_some(),
            "merged_transcript": merged_text,
            "fallback_transcript": fallback_text,
        }))?
    );
    Ok(())
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
