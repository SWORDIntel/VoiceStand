# VoiceStand

**Local, CPU-first voice-to-text input for Linux.**

VoiceStand is being redesigned as a lightweight Linux dictation utility: activate it with a button or hotkey, speak, and commit locally transcribed text into the currently focused application.

> **Project status:** active architectural redesign. The current repository contains useful Rust audio/state infrastructure, but the end-to-end production path is not complete. In particular, real CPU ASR, Linux-wide activation, and focused-application text insertion are the immediate implementation priorities.

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

Initial candidates:

- [`whisper.cpp`](https://github.com/ggml-org/whisper.cpp) — first CPU production backend to integrate and benchmark.
- [`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) — streaming/INT8 alternative to benchmark empirically.

The backend choice is not permanent and should be decided from measurements rather than architecture lock-in.

## Model Strategy

Expose a small number of validated product profiles instead of every model variant.

| Profile | Initial candidate | Intent |
|---|---|---|
| **Fast** | Whisper tiny-class, quantized | Older / low-spec CPUs |
| **Balanced** | Whisper base-class, quantized | Default interactive dictation |
| **Accurate** | Whisper small-class | Faster CPUs or optional GPU |

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

1. Introduce `SpeechBackend`.
2. Integrate real CPU ASR through `whisper.cpp`.
3. Add a fixed WAV/corpus benchmark harness.
4. Replace the placeholder CPU transcription path.
5. Add explicit transcription session IDs/state machine.
6. Implement real global PTT/toggle activation.
7. Implement the first Linux text sink.
8. Benchmark IBus/Fcitx5/Wayland/X11 integration.
9. Optimize endpointing and release-to-final latency.
10. Benchmark tiny/base/small and thread counts.
11. Benchmark a streaming `sherpa-onnx` backend.
12. Run soak, device-recovery, suspend/resume, and desktop-session recovery tests.
13. Add optional GPU/NPU acceleration only after the CPU path passes production gates.

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

## Redesign Specification

The complete architecture, migration phases, production gates, benchmark plan, session-state design, backend evaluation criteria, and implementation order are maintained here:

**[CPU-First Production Redesign](docs/CPU_FIRST_PRODUCTION_REDESIGN.md)**

That document is the authoritative design direction for the current redesign.

## License

See [LICENSE](LICENSE).
