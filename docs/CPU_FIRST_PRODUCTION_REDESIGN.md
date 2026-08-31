# VoiceStand CPU-First Production Redesign

**Status:** Architecture direction / implementation plan  
**Date:** 2026-08-31  
**Target:** Linux local voice-to-text input utility  
**Baseline hardware:** CPU + RAM only  
**Optional acceleration:** GPU, NPU, other hardware backends

---

## 1. Objective

VoiceStand is being refocused from an Intel Meteor Lake NPU/GNA experiment into a practical Linux-wide local dictation tool.

The production objective is deliberately narrow:

> Press or toggle a configured button anywhere in the Linux desktop, speak, and have locally transcribed text inserted into the currently focused application with low perceived latency.

The product must work well without a discrete GPU, Intel NPU, GNA, cloud API, or network connection.

The canonical v1 flow is:

```text
BUTTON
  ↓
LISTEN
  ↓
TRANSCRIBE
  ↓
TYPE
```

Everything else is secondary.

---

## 2. Revised Design Principles

### 2.1 CPU is the baseline, not a fallback

A machine is considered fully supported when VoiceStand works with:

```text
NO GPU
NO NPU
NO network
```

GPU/NPU acceleration may improve model size, throughput, or latency, but it must never be required for normal operation.

### 2.2 Optimize perceived latency, not artificial sub-10 ms inference claims

Speech recognition latency should be measured as user-visible behavior:

- activation-to-capture latency;
- speech-to-partial-text latency;
- release-to-final-text latency;
- real-time factor (RTF);
- p50/p95 latency under load;
- bounded memory use;
- accuracy/WER on a fixed corpus.

A model invocation taking a few milliseconds is not meaningful if the complete dictation path takes substantially longer.

### 2.3 VoiceStand owns orchestration, not transformer internals

VoiceStand should own:

- microphone capture;
- buffering;
- VAD / endpointing;
- activation / PTT;
- backend selection;
- transcription lifecycle;
- text normalization;
- Linux text insertion;
- configuration;
- model management;
- observability and recovery.

VoiceStand should **not** own a bespoke Whisper encoder, decoder, tokenizer, attention implementation, weight loader, quantization engine, or sampling framework unless there is a demonstrated requirement that existing inference engines cannot satisfy.

### 2.4 Local-first and offline by design

Core dictation must require no network access after model acquisition.

Model downloads and update checks must be separate, explicit operations rather than hidden dependencies of transcription.

### 2.5 Hardware acceleration is a backend capability

The core application should not contain Intel-specific assumptions.

Preferred abstraction:

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

Possible implementations:

```text
WhisperCppBackend      # initial/default candidate
SherpaOnnxBackend      # streaming/alternative candidate
Future accelerator backend(s)
```

CPU operation is normal operation for every backend.

---

## 3. Current Repository Assessment

The current Rust tree contains useful infrastructure but is not yet an end-to-end voice-input product.

### 3.1 Useful components to retain

The following concepts are worth preserving and tightening:

- Rust implementation;
- CPAL-based audio capture;
- audio buffering;
- VAD infrastructure;
- configuration/types;
- event/state coordination;
- model management concept;
- structured tracing/error handling;
- separation between audio, state, speech, and application orchestration.

The audio subsystem in `rust/voicestand-audio/` already contains real CPAL device discovery, stream setup, live sample processing, buffering, and VAD-oriented event generation.

### 3.2 Critical paths that are not production implementations yet

#### ASR

The active integration path in `rust/voicestand-core/src/integration.rs` currently returns a placeholder CPU transcription string rather than performing CPU speech recognition.

The disabled `voicestand-speech` crate contains a custom Candle-based Whisper-like implementation, but `model.rs` does not load real Whisper weights and creates simplified/randomly initialized layers. That code should not be completed into a home-grown inference stack for v1.

#### Global hotkey / PTT

`rust/voicestand-state/src/hotkey.rs` and `ptt.rs` contain useful interfaces and event types, but their monitoring/start paths are placeholders rather than working Linux-wide activation backends.

#### Text insertion

There is no complete first-class subsystem that commits transcription into whichever application currently owns keyboard focus.

This is a defining product capability and must be treated as a core subsystem rather than a UI afterthought.

### 3.3 Health semantics are based on the old accelerator architecture

The current integration layer conceptually treats the hardware manager as fundamental. The redesigned health model must instead require:

```text
required:
    audio capture
    ASR backend
    activation backend
    text output backend

optional:
    GPU
    NPU
    wake word
    GUI
    waveform display
    advanced adaptation
```

---

## 4. Target Architecture

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
                    │ Vulkan/CUDA/etc. optional │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │ Text normalization     │
                     │ punctuation            │
                     │ substitutions          │
                     └────────────┬───────────┘
                                  │
                     ┌────────────▼────────────┐
                     │ Linux text-input backend│
                     │ IBus/Fcitx/X11/Wayland  │
                     └─────────────────────────┘
```

The application should remain usable without a full GUI. A tray/status UI can be added around the core service rather than becoming part of the critical transcription path.

---

## 5. ASR Backend Direction

## 5.1 Initial production candidate: whisper.cpp

Use `whisper.cpp` as the first backend to benchmark and integrate rather than maintaining a custom Candle implementation.

Reasons:

- mature CPU-only execution path;
- x86 SIMD optimization;
- quantized model support;
- C API suitable for Rust FFI;
- Linux support;
- optional GPU/accelerator backends;
- no Python runtime requirement;
- widely used Whisper implementation;
- compatible with the low-spec/offline product goal.

Repository:

- <https://github.com/ggml-org/whisper.cpp>

This is a candidate, not a permanent architectural dependency. The backend trait must keep VoiceStand capable of replacing it.

## 5.2 Second benchmark candidate: sherpa-onnx

Benchmark a streaming ASR backend through `sherpa-onnx`, particularly small/INT8 streaming models such as Zipformer-family models where appropriate.

Reasons:

- actual streaming ASR architectures;
- ONNX Runtime CPU execution;
- INT8 model availability;
- native APIs;
- incremental partial recognition;
- potential advantage over Whisper for continuously responsive CPU dictation.

Repository:

- <https://github.com/k2-fsa/sherpa-onnx>

The decision between Whisper and a streaming transducer should be empirical.

## 5.3 FUTO as a feasibility reference

FUTO Voice Input demonstrates that Whisper-class local dictation is practical on phone-class hardware. Its model repository limits supported custom models to Whisper tiny/base/small scale because larger models are generally unsuitable for phones, and discusses optimization specifically for short dictation.

References:

- <https://github.com/futo-org/voice-input>
- <https://github.com/futo-org/voice-input-models>

This is useful evidence for the hardware target, not an instruction to duplicate FUTO's Android architecture.

---

## 6. Model Policy

Do not expose every possible model as a product decision.

Ship three user-facing performance profiles and map them to validated model/backend configurations.

### Fast

Goal: acceptable dictation on older or low-end x86-64 machines.

Initial candidate:

```text
Whisper tiny.en-class model
quantized
CPU
```

Intended baseline:

- 2–4 CPU cores;
- 8 GB system RAM should be more than sufficient;
- no GPU;
- low startup and memory overhead.

### Balanced — default

Goal: best overall interactive dictation profile.

Initial candidate:

```text
Whisper base.en-class model
quantized after benchmarking
CPU baseline
```

This should be the primary optimization target.

### Accurate

Goal: higher accuracy when the hardware budget permits it.

Initial candidate:

```text
Whisper small.en-class model
CPU on sufficiently fast systems
GPU optional
```

Do not promote medium/large models into the normal desktop path until measurements demonstrate that they improve the product enough to justify their cost.

### Automatic profile selection

Hardware detection should recommend a profile, not hard-code the backend.

Example inputs:

- physical cores / logical threads;
- SIMD capabilities;
- available RAM;
- optional supported GPU backend;
- measured warm-up benchmark;
- user preference for speed vs accuracy.

User selection always overrides automatic recommendations.

---

## 7. Streaming and Endpointing

The redesign must stop thinking of every utterance as a fixed 30-second Whisper window.

Short commands and dictation should be optimized as short commands and dictation.

Desired flow:

```text
capture frames
    ↓
ring buffer
    ↓
VAD / PTT state
    ↓
incremental ASR where supported
    ↓
partial hypothesis
    ↓
release / endpoint
    ↓
final decode
    ↓
commit
```

Avoid unconditional padding of short utterances to 30 seconds.

Where supported by the backend, reuse decoder/cache state across chunks rather than reprocessing the entire utterance on every partial update.

PTT release should force prompt finalization rather than waiting for a long silence detector timeout.

---

## 8. Latency and Performance Metrics

Remove the old global `<10 ms transcription` target.

Use metrics that correspond to user experience.

### Activation latency

```text
button event → microphone capture active
Target: < 30 ms
```

### Partial hypothesis latency

```text
spoken audio → useful partial text available
Target: ~100–300 ms on normal hardware
```

### Finalization latency

```text
PTT release / endpoint → final committed text
Ideal: < 300 ms
Low-spec acceptable: < 600 ms typical
Production p95 target: < 700 ms on baseline test hardware
```

### Sustained inference

```text
RTF < 1.0    mandatory
RTF < 0.5    good baseline
RTF < 0.25   excellent
```

### Required benchmark outputs

Every candidate backend/model/profile should record:

- WER;
- p50 first-partial latency;
- p95 first-partial latency;
- p50 release-to-final latency;
- p95 release-to-final latency;
- real-time factor;
- RSS after model load;
- peak RSS;
- model load time;
- CPU utilization;
- throughput scaling at 1/2/4/8 threads;
- package energy per utterance where practical;
- thermal throttling behavior during sustained dictation.

Benchmarks must use a fixed, versioned speech corpus so regressions can be detected.

---

## 9. Linux Activation / PTT

Activation is a first-class subsystem.

Recommended abstraction:

```rust
trait ActivationBackend {
    fn register(&mut self, binding: &Binding) -> Result<()>;
    fn events(&mut self) -> Result<ActivationEventStream>;
}
```

Required events:

```text
Pressed
Released
ToggleOn
ToggleOff
BackendUnavailable
BindingConflict
```

Target environments:

- Wayland;
- X11;
- KDE Plasma;
- GNOME;
- wlroots-based compositors where practical.

Avoid requiring root or broad `/dev/input` access simply to implement a global hotkey.

Prefer desktop-supported APIs/portals where available, then use compositor/X11-specific backends as required.

The application must handle a lost/restarted desktop session without leaving recording active or modifiers stuck.

---

## 10. Linux Text Output

Text insertion is as important as speech recognition.

Define it explicitly:

```rust
trait TextSink {
    fn begin(&mut self) -> Result<()>;
    fn partial(&mut self, text: &str) -> Result<()>;
    fn commit(&mut self, text: &str) -> Result<()>;
    fn cancel(&mut self) -> Result<()>;
}
```

Candidate backends:

```text
IBus
Fcitx5
Wayland/compositor integration
X11
clipboard + synthetic paste fallback
```

### Preferred long-term direction: input-method integration

Investigate IBus/Fcitx5 as the primary Linux architecture.

That allows VoiceStand to behave like a genuine input method instead of a process that merely simulates keyboard typing.

Conceptually:

```text
VoiceStand = Linux IME whose input source is speech
```

This maps cleanly to the desired behavior across:

- browsers;
- terminals;
- editors;
- IDEs;
- office applications;
- chat clients;
- arbitrary native text fields.

### Clipboard fallback

Clipboard + paste injection may be retained as a compatibility fallback, but it must:

- restore the user's previous clipboard content when possible;
- avoid leaking transcription into persistent clipboard history without explicit policy;
- handle password/secure-input fields conservatively;
- never synthesize uncontrolled shell commands merely because a terminal is focused.

---

## 11. Text Post-Processing

Keep post-processing deterministic and lightweight in the default path.

Core functions:

- whitespace normalization;
- punctuation cleanup;
- capitalization;
- optional spoken punctuation commands;
- configurable phrase substitutions;
- optional command vocabulary;
- duplicate/repetition suppression;
- final/partial reconciliation.

Do not require an LLM for ordinary dictation.

If an optional language-model correction stage is introduced later, it must be explicitly separable from raw ASR and must not delay baseline text commitment.

---

## 12. Audio Architecture

Retain the existing CPAL work where practical, but simplify around the product flow.

Required properties:

- 16 kHz mono canonical ASR format internally;
- accept real hardware sample rates and resample as required;
- bounded ring buffers;
- no unbounded audio retention;
- recover from device removal/reconnection;
- explicit microphone selection;
- PipeWire-friendly behavior on modern Linux;
- avoid blocking work inside real-time audio callbacks;
- separate capture from inference threads/tasks.

The audio callback should perform the minimum work necessary to copy/enqueue frames safely.

VAD and resampling should be benchmarked independently from ASR so their overhead is visible.

---

## 13. Threading / Resource Strategy

CPU-first does not mean consume every available thread.

VoiceStand is an interactive desktop utility and should remain polite to foreground applications.

Default policy:

- reserve system responsiveness;
- avoid automatically occupying all logical CPUs;
- benchmark 1/2/4/8-thread inference scaling;
- select the smallest thread count that maintains the target RTF;
- make thread count configurable;
- keep the model loaded between utterances;
- avoid repeated allocator churn on the hot path;
- avoid model reload between activations.

For heterogeneous CPUs, affinity/topology tuning can be added only after normal scheduler behavior is measured.

P-core/E-core/NPU-specific scheduling is not a v1 requirement.

---

## 14. GPU / Accelerator Policy

GPU support is opportunistic.

Desired behavior:

```text
GPU available and validated
    → optionally use it

GPU missing / broken / unsupported
    → continue normally on CPU
```

Never describe CPU mode as degraded/fallback mode in the architecture or UI.

Potential future accelerators:

- Vulkan;
- CUDA;
- ROCm/HIP-supported backend;
- OpenVINO;
- vendor NPUs.

Each accelerator must pass the same correctness corpus as CPU and must be disableable at runtime/configuration level.

---

## 15. Scope Reduction for v1

Remove or defer the following from the critical path:

- Intel GNA wake-word operation;
- Intel NPU requirement;
- custom Whisper implementation;
- hybrid NPU/CPU scheduling;
- elaborate thermal management;
- advanced deployment-manager logic;
- waveform rendering;
- always-on wake word;
- adaptive voice learning unless measurements show a clear benefit;
- complex GUI features;
- model-size proliferation;
- unsupported sub-10 ms ASR claims.

These can remain experimental modules or be reintroduced after the core product is stable.

---

## 16. Proposed Crate Direction

The exact names can change, but the responsibilities should converge toward:

```text
voicestand-types
    shared data structures and errors

voicestand-audio
    capture, ring buffer, resampling, VAD

voicestand-asr
    SpeechBackend trait + backend implementations

voicestand-activation
    PTT/toggle/hotkey backends

voicestand-input
    IBus/Fcitx/X11/Wayland text sinks

voicestand-core
    transcription session state machine and orchestration

voicestand
    daemon/application entry point

voicestand-ui          optional
    tray/settings/status UI only
```

Intel-specific code should not sit in the central dependency graph. If retained:

```text
voicestand-intel       optional/experimental
```

---

## 17. Session State Machine

Define a small explicit state machine rather than distributing activation state across loosely coupled components.

Suggested states:

```text
Idle
Activating
Listening
Recognizing
Finalizing
Committing
ErrorRecoverable
ShuttingDown
```

Typical PTT transition:

```text
Idle
  ↓ press
Activating
  ↓ capture active
Listening
  ↓ partial inference
Recognizing
  ↓ release
Finalizing
  ↓ final result
Committing
  ↓ text sink complete
Idle
```

Required invariants:

- one active capture session at a time;
- one final commit per session;
- stale partial results cannot overwrite a newer session;
- release always terminates the intended session;
- backend errors return the system to a recoverable state;
- microphone capture is stopped on shutdown/session loss.

Use a monotonically increasing session ID for cross-task/event correlation.

---

## 18. Production Acceptance Environment

The minimum production validation machine should intentionally lack accelerator hardware.

Baseline example:

```text
Linux
Wayland
4 CPU cores
8 GB RAM
integrated/basic graphics only
no usable NPU
no network during transcription
```

Validation sequence:

1. start VoiceStand;
2. model loads once;
3. focus a browser text field;
4. press PTT;
5. dictate at least 20 words;
6. release PTT;
7. verify text commits correctly;
8. focus a terminal and repeat;
9. focus an editor and repeat;
10. focus LibreOffice or equivalent and repeat;
11. change microphone and repeat;
12. disconnect/reconnect microphone and recover;
13. suspend/resume and recover;
14. restart compositor/session components where practical and recover;
15. execute 500 consecutive activation cycles;
16. verify RSS remains bounded;
17. verify no stuck microphone state;
18. verify no stuck modifiers/keys;
19. verify no unintended network traffic;
20. verify all transcriptions belong to the correct session/focused target.

---

## 19. Production Performance Gates

Initial CPU-only targets:

```text
RTF                         < 0.5 on Balanced baseline
release-to-commit p50       < 300 ms
release-to-commit p95       < 700 ms
idle CPU                    approximately 0%
idle/model-loaded RSS       bounded and measured
500-session soak            no unbounded RSS growth
network during dictation    none
```

Accuracy gates must be based on a versioned corpus rather than undocumented percentages.

Required quality metrics:

- overall WER;
- command/short-utterance WER;
- punctuation quality;
- proper noun failure rate on a defined test set;
- repetition/hallucination rate;
- blank/silence false-transcription rate;
- noisy-room subset;
- different microphone subset.

---

## 20. Security and Privacy Requirements

VoiceStand handles microphone data and arbitrary user text, so privacy is part of the architecture.

Requirements:

- no network dependency for transcription;
- no telemetry by default;
- no persistent raw audio by default;
- logs must not contain full transcription text unless debug mode explicitly enables it;
- model downloads should be integrity-checked;
- secure temporary-file behavior;
- no root requirement for normal use;
- least-privilege desktop integration;
- clipboard fallback must minimize exposure to clipboard managers;
- redact/avoid sensitive text in crash reports;
- bound all audio/session buffers;
- reject malformed/untrusted model metadata rather than using unsafe deserialization patterns.

---

## 21. Benchmark Matrix Before Locking the Backend

Do not select the permanent ASR backend by preference.

Benchmark at minimum:

```text
whisper.cpp tiny.en quantized
whisper.cpp base.en quantized
whisper.cpp small.en quantized
sherpa-onnx streaming INT8 candidate
```

Test on at least:

```text
low-spec 4-core CPU
mid-range modern laptop CPU
high-core desktop/server CPU
optional integrated/discrete GPU system
```

For each combination record:

```text
WER
short-utterance WER
RTF
first-partial p50/p95
finalization p50/p95
RSS
peak RSS
load time
1/2/4/8-thread scaling
CPU utilization
power/energy if available
```

The default backend/profile is the one that delivers the best actual dictation experience under the baseline constraints.

---

## 22. Migration Plan

### Phase 0 — freeze old assumptions

- stop adding Intel NPU/GNA-specific features to the core path;
- mark old latency claims as historical/experimental;
- keep old code available only where it provides reusable components.

### Phase 1 — real CPU ASR

- define `SpeechBackend`;
- integrate `whisper.cpp` through a small FFI boundary;
- load a real tiny/base model;
- feed captured PCM into the backend;
- return real `Transcript` objects;
- remove the placeholder CPU transcription result;
- add fixed WAV-file integration tests.

**Exit gate:** deterministic command-line transcription of a corpus using CPU only.

### Phase 2 — real PTT

- define `ActivationBackend`;
- implement at least one reliable Linux global activation path;
- support press/release and toggle semantics;
- bind activation to audio session lifecycle.

**Exit gate:** PTT reliably starts/stops a real CPU transcription session.

### Phase 3 — real text insertion

- define `TextSink`;
- implement an initial Linux text sink;
- investigate IBus/Fcitx5 as preferred long-term integration;
- retain a carefully implemented compatibility fallback.

**Exit gate:** browser/editor/terminal dictation works end-to-end.

### Phase 4 — streaming UX

- add partial hypotheses where supported;
- optimize endpointing;
- force fast finalization on PTT release;
- eliminate unnecessary full-window padding/reprocessing;
- benchmark partial/final latency.

**Exit gate:** interactive latency meets production targets on baseline CPU.

### Phase 5 — backend benchmark

- add sherpa-onnx experimental backend;
- run the fixed corpus and latency matrix;
- choose default backend/profile from measurements.

### Phase 6 — robustness

- audio-device recovery;
- suspend/resume;
- compositor/session recovery;
- cancellation/stale-session handling;
- 500+ session soak tests;
- fault injection around backend/device failure.

### Phase 7 — packaging / UX

- tray/status UI;
- model setup UI;
- profile selection;
- packaging for target distributions;
- autostart/service integration;
- reproducible release builds.

### Phase 8 — optional acceleration

Only after CPU production gates pass:

- Vulkan;
- CUDA;
- ROCm/other backend;
- OpenVINO/NPU experiments;
- wake word if still useful.

---

## 23. Immediate Engineering Worklist

Highest-value order:

1. **Delete the custom Candle Whisper implementation from the production plan.**
2. **Introduce `SpeechBackend`.**
3. **Integrate real CPU inference via whisper.cpp.**
4. **Create a fixed WAV/corpus benchmark harness.**
5. **Replace placeholder CPU transcription in `integration.rs`.**
6. **Introduce explicit transcription session IDs/state machine.**
7. **Implement real global PTT.**
8. **Implement the first real text sink.**
9. **Benchmark Wayland/X11/IBus/Fcitx integration paths.**
10. **Optimize release-to-final latency.**
11. **Benchmark tiny/base/small and thread counts.**
12. **Benchmark a streaming sherpa-onnx candidate.**
13. **Run soak/recovery tests.**
14. **Only then add optional GPU acceleration and UI polish.**

---

## 24. Definition of Production-Ready

VoiceStand is production-ready when the following statement is true:

> On an ordinary accelerator-free Linux machine, a user can start VoiceStand, focus essentially any normal text-capable application, press or toggle a configured activation control, dictate locally, and receive accurate text with bounded latency and memory use for hundreds of consecutive sessions without network access, stuck capture state, or manual recovery.

Production readiness is **not** defined by:

- crate count;
- GUI completeness;
- accelerator support;
- theoretical inference latency;
- presence of deployment scripts;
- successful unit tests that do not exercise real ASR and desktop input.

It is defined by the complete user path working reliably.

---

## 25. Final Direction

VoiceStand should become a small, reliable Linux input utility rather than a hardware-acceleration showcase.

The architecture priority is:

```text
1. CPU correctness
2. low latency
3. Linux-wide input integration
4. robustness
5. accuracy
6. resource efficiency
7. optional acceleration
8. optional GUI/advanced features
```

The existing Rust/audio/state work provides a useful base, but the redesign should aggressively replace unfinished custom ASR and accelerator-centric assumptions with proven inference backends and first-class Linux input-method integration.
