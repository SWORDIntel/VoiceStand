use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde_json::json;
use voicestand_asr::{
    benchmark::percentile, read_wav_16khz_mono, DecodeOptions, ModelSpec, SpeechBackend,
    WhisperCppBackend,
};
use voicestand_types::{Result, VoiceStandError};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("VoiceStand ASR benchmark failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let usage = "usage: transcribe_wav MODEL WAV [THREADS] [ITERATIONS] [MAX_P95_MS]";
    let model_path = PathBuf::from(args.next().ok_or_else(|| VoiceStandError::config(usage))?);
    let wav_path = PathBuf::from(args.next().ok_or_else(|| VoiceStandError::config(usage))?);
    let thread_count = parse_arg(args.next(), "thread count", 2usize)?;
    let iterations = parse_arg(args.next(), "iteration count", 10usize)?;
    let max_p95_ms = args
        .next()
        .map(|value| parse_number(&value.to_string_lossy(), "maximum p95 milliseconds"))
        .transpose()?;
    if iterations == 0 {
        return Err(VoiceStandError::config("iteration count must be positive"));
    }

    let audio = read_wav_16khz_mono(&wav_path)?;
    let audio_seconds = audio.len() as f64 / 16_000.0;
    let options = DecodeOptions {
        thread_count,
        ..DecodeOptions::default()
    };
    let mut backend = WhisperCppBackend::cpu();
    let load_started = Instant::now();
    backend.load(&ModelSpec::new("benchmark", &model_path))?;
    let model_load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;

    // Exclude model load and the first warm-up decode from warm-path percentiles.
    backend.transcribe(&audio, &options)?;
    let mut latencies_ms = Vec::with_capacity(iterations);
    let mut rtfs = Vec::with_capacity(iterations);
    let mut transcript = None;
    for _ in 0..iterations {
        let started = Instant::now();
        let result = backend.transcribe(&audio, &options)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        latencies_ms.push(elapsed_ms);
        rtfs.push(elapsed_ms / 1_000.0 / audio_seconds.max(f64::EPSILON));
        transcript = Some(result);
    }

    let p50_ms = percentile(&latencies_ms, 50.0).expect("non-empty measurements");
    let p95_ms = percentile(&latencies_ms, 95.0).expect("non-empty measurements");
    let rtf_p50 = percentile(&rtfs, 50.0).expect("non-empty measurements");
    let rtf_p95 = percentile(&rtfs, 95.0).expect("non-empty measurements");
    let transcript = transcript.expect("positive iteration count");
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema_version": 1,
            "metric_scope": "warm_decode_component_of_release_to_final",
            "backend": backend.name(),
            "model": model_path,
            "wav": wav_path,
            "threads": thread_count,
            "warmup_iterations": 1,
            "measured_iterations": iterations,
            "audio_seconds": audio_seconds,
            "model_load_ms": model_load_ms,
            "decode_ms": { "p50": p50_ms, "p95": p95_ms, "samples": latencies_ms },
            "rtf": { "p50": rtf_p50, "p95": rtf_p95 },
            "resident_memory_kib": resident_memory_kib(),
            "transcript": transcript.text,
            "mean_token_confidence": transcript.confidence,
            "p95_gate_ms": max_p95_ms,
            "p95_gate_passed": max_p95_ms.map(|limit| p95_ms <= limit),
        }))?
    );

    if let Some(limit) = max_p95_ms {
        if p95_ms > limit {
            return Err(VoiceStandError::system(format!(
                "warm decode p95 {p95_ms:.1} ms exceeds {limit:.1} ms gate"
            )));
        }
    }
    Ok(())
}

fn parse_arg<T>(value: Option<std::ffi::OsString>, name: &str, default: T) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .map(|value| parse_number(&value.to_string_lossy(), name))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_number<T>(value: &str, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| VoiceStandError::config(format!("invalid {name}: {error}")))
}

fn resident_memory_kib() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}
