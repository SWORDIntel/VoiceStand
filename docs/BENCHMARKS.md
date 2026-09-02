# VoiceStand benchmark evidence

**Updated:** 2026-09-02

## Metric boundaries

- **RTF** is inference compute time divided by audio duration. Values below 1.0 are mandatory for live recognition; below 0.5 is preferred.
- **Final flush** is compute after the last audio frame until the streaming recognizer returns its final result.
- **Release-to-commit** additionally includes orchestration and text insertion. It is not interchangeable with decode or final-flush time.

| Metric | Target |
|---|---:|
| Activation to capture | <30 ms |
| Speech to useful partial | approximately 100–300 ms |
| Release to final commit p50 | <300 ms |
| Release to final commit p95 | <700 ms |
| RTF | <1.0 mandatory; <0.5 preferred |

## Reproducible corpus

The committed fixtures are under `benchmarks/corpus/`. `SOURCES.md` records their upstream provenance and duration; `SHA256SUMS` detects replacement or corruption.

```bash
sha256sum -c benchmarks/corpus/SHA256SUMS
```

The corpus contains canonical whisper.cpp JFK audio plus two English sherpa-onnx fixtures. All are 16-bit mono PCM at 16 kHz.

## whisper.cpp baseline

Backend: whisper.cpp through `whisper-rs`

Model: `ggml-tiny.en.bin`

Input: 11-second JFK fixture

Threads: 4

| Metric | Result |
|---|---:|
| Decode p50/p95 (one measured iteration) | 20,764.9 ms |
| RTF p50/p95 | 1.888 |
| Model load | 185.3 ms |
| Resident memory | 91,896 KiB |

This fails the interactive requirement. The result measures the decode component, not complete release-to-commit latency.

```bash
cd rust
cargo run --release -p voicestand-asr --example transcribe_wav -- \
  /path/to/ggml-tiny.en.bin ../benchmarks/corpus/jfk.wav 4 1 700
```

## Release-path optimization

The implementation supports exact completed-partial reuse, one-second acoustic overlap with at most two seconds of uncovered tail, and a full-utterance correctness fallback when reconciliation is ambiguous.

```bash
cd rust
cargo run --release -p voicestand-core --example release_path_benchmark -- \
  /path/to/ggml-tiny.en.bin ../benchmarks/corpus/jfk.wav 4 1 1000
```

These optimizations avoid redundant work but do not make whisper.cpp suitable for the normal live path on this host.

## sherpa-onnx streaming candidate

Runtime: sherpa-onnx 1.13.7 native CPU engine

Model: `sherpa-onnx-streaming-zipformer-en-20M-2023-02-17`

Precision tested: fp32 and int8

Input cadence: 100 ms frames

| Metric | Result |
|---|---:|
| Compute RTF | 0.11–0.15 |
| Final flush | 63–103 ms |
| JFK native decode | 1.6–1.7 s for approximately 11 s audio |

JFK with 300 ms leading pre-roll produced the complete sentence structure but recognized “and so” as “and saw” and one “ask” as “asked.” This is sufficient to promote the engine into Rust integration, not sufficient to declare the model accuracy-qualified.

## Interpretation and next experiment

The backend decision is no longer blocked on speed: Zipformer passes and whisper.cpp fails the live latency gate. The next benchmark must run through the Rust online backend and report first-partial p50/p95, final-flush p50/p95, release-to-commit p50/p95, WER, peak RSS, and a longer dictation corpus.

Hosted CI is not used for iterative performance work. `scripts/ci-local.sh` is the authoritative pre-push correctness/build/package gate; performance results are gathered locally on an identified host and are not presented as portable hardware guarantees.
