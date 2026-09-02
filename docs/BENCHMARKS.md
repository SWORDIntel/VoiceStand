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

The first `voicestand-asr` Rust session run established an additional integration constraint. Forwarding 100 ms chunks with only 300 ms pre-roll clipped the opening speech; one second of pre-roll preserved the complete JFK phrase structure. Debug runs vary with host load: after core integration, a two-thread run produced 22 progressive updates, 0.46 RTF, a 356 ms final flush, and its first partial after 1.1 seconds of fed audio. A four-thread comparison regressed to 0.93 RTF, so the 20M runtime is capped at two threads. Release-mode percentiles remain to be recorded.

## Interpretation and next experiment

The backend decision is no longer blocked on speed: Zipformer passes and whisper.cpp fails the live latency gate. The next benchmark must run through the Rust online backend and report first-partial p50/p95, final-flush p50/p95, release-to-commit p50/p95, WER, peak RSS, and a longer dictation corpus.

Hosted CI is not used for iterative performance work. `scripts/ci-local.sh` is the authoritative pre-push correctness/build/package gate; performance results are gathered locally on an identified host and are not presented as portable hardware guarantees.

Run the complete three-file corpus locally with:

```bash
./scripts/benchmark-streaming.sh /path/to/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 5
```

The release-mode harness uses the production audio/PTT orchestration rather than calling the recognizer directly. Its first JFK run exposed that the one-second pre-roll had accidentally lived inside `debug_assert!` and was therefore removed from optimized builds; the harness now protects against repeating that class of debug/release mismatch. A final-padding sweep showed that 400 ms preserved the same transcripts as 800 ms while removing unnecessary release work. Across ten JFK runs at the new default, first-partial p50/p95 was 415/794 ms, release-to-final p50/p95 was 57/157 ms, compute RTF p50 was 0.39, and all outputs had 4.55% WER. Three-run checks of the other fixed fixtures measured release p95 values of 131 ms and 73 ms with unchanged WER. The 700 ms release p95 gate now passes.
