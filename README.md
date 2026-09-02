# VoiceStand

**Local, CPU-first voice-to-text input for Linux.**

VoiceStand is being redesigned as a lightweight Linux dictation utility: activate it with a button or hotkey, speak, and commit locally transcribed text into the currently focused application.

> **Project status:** midpoint implementation. The X11/XWayland path now has real microphone capture, global hold/toggle activation, local whisper.cpp transcription, focused-window insertion, release packaging, and local CI. Whisper.cpp is retained as the correctness fallback but fails interactive latency on this host. A sherpa-onnx Zipformer candidate has passed the latency gate and is the next live backend to integrate. Manual desktop acceptance and recovery testing remain before production readiness.

## Direction

VoiceStand is no longer designed around Intel Meteor Lake NPU/GNA hardware.

The baseline product must work with:

```text
CPU + RAM
no discrete GPU
no NPU
no cloud API
no network connection during transcription
```

GPU, NPU, Vulkan, CUDA, OpenVINO, or other acceleration may be added as optional backends after the CPU path meets production requirements.

The canonical product flow is intentionally narrow:

```text
BUTTON
  ↓
LISTEN
  ↓
TRANSCRIBE
  ↓
TYPE
```

## Target Architecture

```text
                      ┌────────────────────────┐
                      │    Hotkey / PTT        │
                      └───────────┬────────────┘
                                  │
                                  ▼
┌──────────────┐        ┌──────────────────────┐
│ CPAL/PipeWire│───────▶│ Audio ring buffer    │
└──────────────┘        └───────────┬──────────┘
                                    │
                           ┌────────▼─────────┐
                           │ VAD / endpoint   │
                           └────────┬─────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │       ASR backend         │
                    │                           │
                    │ CPU              baseline │
                    │ GPU/NPU          optional │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ Text normalization     │
                     └────────────┬───────────┘
                                  │
                     ┌────────────▼────────────┐
                     │ Linux text-input backend│
                     │ IBus/Fcitx/X11/Wayland  │
                     └─────────────────────────┘
```

## CPU quick start

```bash
sudo apt-get install -y build-essential pkg-config libasound2-dev curl xdotool
./scripts/install-model.sh
cd rust
cargo build -p voicestand --release
./target/release/voicestand --check
./target/release/voicestand
```

The current installer downloads `ggml-tiny.en.bin` for the implemented whisper.cpp fallback, verifies its SHA-256 digest, and atomically installs it under the platform configuration directory. Set `VOICESTAND_MODEL_DIR` to override the installation directory. The streaming Zipformer installer/configuration will land with its Rust backend integration; it is not yet selected by the application.

The first production desktop path uses X11/XWayland global hotkeys and focused-window text insertion. Hold `Ctrl+Alt+V`, speak, and release to finalize and insert the transcript, or press `Ctrl+Alt+Space` to toggle recording on and off. Native Wayland input-method support remains a planned backend; VoiceStand reports an explicit error instead of pretending global activation works when neither X11 nor XWayland is available.

To validate real inference without a microphone:

```bash
./target/release/voicestand --smoke-test /path/to/16khz-mono.wav
```

To measure repeatable warm-decode latency and enforce an optional p95 gate:

```bash
cd rust
cargo run --release -p voicestand-asr --example transcribe_wav -- \
  MODEL WAV THREADS ITERATIONS MAX_P95_MS
```

The command performs one unmeasured warm-up, emits versioned JSON containing decode p50/p95, RTF p50/p95, samples, load time, and resident memory, and exits nonzero when the optional p95 limit is exceeded. Its latency scope is explicitly the warm decode component of release-to-final; it does not include focused-application insertion.

To compare the three PTT release paths without spending hosted CI time:

```bash
cd rust
cargo run --release -p voicestand-core --example release_path_benchmark -- \
  /path/to/model.bin /path/to/16khz-mono.wav 4 1 1000
```

The final argument is the simulated uncovered tail in milliseconds (`1..=2000`). The JSON report separates exact-cache reuse, bounded overlapping-tail decoding, and full fallback latency and records whether word-overlap reconciliation succeeded.

An optional second argument overrides the configured model path. The command prints the transcript and reports model-load time, decode time, real-time factor, and confidence.

Before pushing CI changes, run the same gate used by GitHub Actions:

```bash
./scripts/ci-local.sh
```

The gate uses the committed lockfile, tests the complete workspace, applies strict linting to the production ASR, activation, and text-output boundaries, and builds the release executable. GitHub CI contains one parity job that calls this script directly and cancels superseded runs.

## Architecture Rules

### CPU is normal operation

CPU inference is not a degraded fallback mode. A system is considered fully supported when VoiceStand works correctly without accelerator hardware.

### VoiceStand does not implement its own transformer

The application should own audio capture, session state, VAD, backend selection, text processing, desktop integration, configuration, and recovery.

It should use a proven inference runtime rather than maintaining a bespoke Whisper encoder/decoder/tokenizer/quantization stack.

### ASR is backend-driven

The intended interface is approximately:

```rust
trait SpeechBackend {
    fn load(&mut self, model: &ModelSpec) -> Result<()>;
    fn transcribe(
        &mut self,
        audio: &[f32],
        options: &DecodeOptions,
    ) -> Result<Transcript>;
}
```

Measured backend roles at this midpoint:

- [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) — implemented offline/fallback backend. The tiny.en warm decode measured 20.8 s for an 11 s JFK fixture (RTF 1.89), so it is not acceptable for interactive dictation on this host.
- [`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) — selected streaming candidate. The 20M English Zipformer measured 0.11–0.15 RTF and 63–103 ms final flush locally. It passes latency but still needs Rust integration and broader WER testing.

The backend choice is not permanent and should be decided from measurements rather than architecture lock-in.

## Model Strategy

Expose a small number of validated product profiles instead of every model variant.

| Profile | Initial candidate | Intent |
|---|---|---|
| **Live** | sherpa-onnx streaming Zipformer 20M, int8 | Immediate partials and low release latency |
| **Fallback** | whisper.cpp tiny.en | Offline compatibility and accuracy comparison |
| **Accurate** | To be selected from corpus measurements | Optional second pass or faster hardware |

FUTO Voice Input is a useful feasibility reference: practical local Whisper-class dictation already runs on phone-class hardware. VoiceStand should therefore be able to provide good CPU-only dictation on ordinary x86-64 Linux machines without treating a GPU as mandatory.

References:

- <https://github.com/futo-org/voice-input>
- <https://github.com/futo-org/voice-input-models>

## Performance Philosophy

The old global `<10 ms` transcription target is retired.

VoiceStand will measure user-visible latency instead:

```text
activation → capture          target < 30 ms
speech → useful partial       target ~100–300 ms
release → final commit p50    target < 300 ms
release → final commit p95    target < 700 ms
RTF                           mandatory < 1.0
RTF                           preferred < 0.5
```

Every backend/model profile must be benchmarked for:

- WER;
- short-utterance WER;
- first-partial p50/p95;
- release-to-final p50/p95;
- real-time factor;
- RSS / peak RSS;
- model load time;
- 1/2/4/8-thread scaling;
- CPU utilization;
- repetition/hallucination rate;
- silence false-positive rate.

## Linux Desktop Integration

VoiceStand is not complete when it can print a transcript to stdout. It is complete when dictation works across normal Linux applications.

Two first-class abstractions are required:

```rust
trait ActivationBackend {
    fn register(&mut self, binding: &Binding) -> Result<()>;
    fn events(&mut self) -> Result<ActivationEventStream>;
}
```

```rust
trait TextSink {
    fn begin(&mut self) -> Result<()>;
    fn partial(&mut self, text: &str) -> Result<()>;
    fn commit(&mut self, text: &str) -> Result<()>;
    fn cancel(&mut self) -> Result<()>;
}
```

Target desktop paths include:

- IBus;
- Fcitx5;
- Wayland/compositor-supported integration;
- X11;
- carefully implemented clipboard/paste fallback.

The preferred long-term model is for VoiceStand to behave like a Linux input method whose source is speech, rather than merely pretending to type keys.

## Current Repository

Useful pieces already exist and should be retained where practical:

```text
rust/voicestand-audio/       CPAL audio capture, buffering, VAD
rust/voicestand-state/       state/activation scaffolding
rust/voicestand-core/        orchestration/integration concepts
rust/voicestand-speech/      experimental speech work; not the production backend
rust/voicestand-gui/         optional UI work
rust/voicestand-hardware/    legacy/optional hardware abstraction
```

The immediate redesign removes Intel-specific hardware assumptions from the required dependency path.

## Immediate Work Order

1. Integrate sherpa-onnx's Rust online recognizer behind a streaming backend contract.
2. Feed live 100 ms frames into one persistent recognition stream and publish accumulated partials.
3. Make the streaming backend the default only after fixed-corpus accuracy and release-latency gates pass.
4. Keep whisper.cpp as an explicit offline/fallback path; remove its full-utterance work from normal live dictation.
5. Run the browser/editor/terminal X11 acceptance matrix.
6. Add audio-device loss/recovery, suspend/resume, and live microphone soak tests.
7. Implement native Wayland/input-method backend selection.
8. Add optional GPU/NPU acceleration only after the CPU path passes production gates.

## Production Baseline

A release candidate must be validated on an intentionally accelerator-free Linux system, for example:

```text
Wayland Linux desktop
4 CPU cores
8 GB RAM
no usable NPU
no discrete GPU requirement
no network during transcription
```

Required end-to-end validation includes dictation into a browser, terminal, editor/IDE, office application, microphone disconnect/reconnect, suspend/resume, and at least 500 consecutive activation cycles with bounded memory use and no stuck microphone/input state.

## Privacy / Security

Core requirements:

- fully local transcription;
- no telemetry by default;
- no persistent raw audio by default;
- transcription text excluded from normal logs;
- model integrity checking;
- no root requirement for normal operation;
- least-privilege desktop integration;
- bounded audio/session buffers;
- conservative handling of secure/password fields;
- no hidden network dependency.

## Build / Development

The current tree is undergoing refactoring, so successful compilation should not be interpreted as end-to-end product readiness.

Rust workspace:

```bash
cd rust
cargo build --release
```

Run checks while redesign work proceeds:

```bash
cd rust
cargo test --workspace
cargo clippy --workspace --all-targets --all-features
```

Some legacy/experimental crates may remain disabled from the active workspace until their responsibilities are either removed or reintegrated behind the new interfaces.

## Authoritative Documentation

The complete architecture, migration phases, production gates, benchmark plan, session-state design, backend evaluation criteria, and implementation order are maintained here:

- **[Current Architecture](docs/CURRENT_ARCHITECTURE.md)** — implemented data flow, backend decision, boundaries, and next migration.
- **[Project Status](docs/PROJECT_STATUS.md)** — phase ledger and remaining production gates.
- **[Benchmark Evidence](docs/BENCHMARKS.md)** — corpus, commands, measurements, and interpretation.
- **[CPU-First Production Redesign](docs/CPU_FIRST_PRODUCTION_REDESIGN.md)** — broader design requirements.

## License

See [LICENSE](LICENSE).
