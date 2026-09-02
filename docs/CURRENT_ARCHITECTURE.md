# Current VoiceStand architecture

**Updated:** 2026-09-02

**Status:** midpoint implementation; not yet production-ready

## Product boundary

VoiceStand is a fully local Linux push-to-talk dictation utility. Normal operation must require only CPU, RAM, microphone access, and a supported Linux text-input path. Transcription must not require a cloud API or network access.

## Implemented path

```text
X11/XWayland hold Ctrl+Alt+V or toggle Ctrl+Alt+Space
  -> CPAL microphone callback
  -> mono 16 kHz normalization
  -> bounded buffer, 300 ms pre-roll, VAD and PTT utterance assembly
  -> resident whisper.cpp backend
  -> stabilized partial events
  -> exact-cache finalization, or bounded overlapping-tail reconciliation
  -> correctness-preserving full decode fallback
  -> sanitized TextSink commit through xdotool
  -> focused application
```

The application has deterministic session cancellation, a 500-session PTT soak, bounded utterance duration, and generation guards that prevent partials from an older session being committed into a newer one.

## Backend decision

### whisper.cpp

Whisper.cpp is implemented and remains useful as an offline compatibility and correctness fallback. It is not suitable as the normal interactive backend on the measured host: tiny.en required 20.8 seconds to warm-decode the 11-second JFK fixture (RTF 1.89). Exact partial reuse and bounded-tail decoding reduce duplicate work but cannot turn a slower-than-real-time engine into responsive live dictation.

### sherpa-onnx Zipformer

Sherpa-onnx 1.13.7 with the English 20M streaming Zipformer is the selected live candidate. Independent native tests measured:

- 0.11–0.15 compute RTF;
- 63–103 ms final stream flush;
- 1.6–1.7 seconds compute for approximately 11 seconds of JFK audio;
- useful partial updates during incremental 100 ms input.

The candidate passes the latency gate. It is not yet wired into the Rust application and has not passed a broad WER corpus. JFK with production-like 300 ms pre-roll retained the full phrase structure but produced two word-level errors. Promotion therefore requires both integration and accuracy evidence.

## Target live path

```text
100 ms normalized audio frames
  -> one persistent sherpa-onnx OnlineRecognizer stream per PTT session
  -> incremental partial result
  -> endpoint or PTT release
  -> final stream flush (<700 ms p95 gate)
  -> text commit

whisper.cpp
  -> explicit offline/fallback or optional second pass
```

The live path must never launch a new full-utterance decode for every partial. Model state stays resident and recognition state advances incrementally.

## Interfaces and ownership

- `voicestand-audio` owns capture, normalization, VAD, pre-roll, and bounded utterance assembly.
- `voicestand-asr` owns backend contracts, model validation, inference, and backend-neutral transcripts.
- `voicestand-core` owns session generation, cancellation, partial/final publication, metrics, and recovery orchestration.
- `voicestand-state` owns activation state and X11/XWayland hotkeys.
- `voicestand-text` owns transcript sanitization and focused-window insertion.
- The application crate owns startup, configuration, event routing, and clean shutdown.

## Safety and privacy properties

- Audio and transcripts remain local.
- Raw audio is not persisted during normal operation.
- Transcript content is excluded from normal diagnostic logging.
- Audio/session buffers are bounded.
- Text insertion removes newline and control characters so dictation cannot implicitly submit a terminal command.
- External commands receive arguments directly; transcript text is not passed through a shell.
- Stale ASR work cannot publish into a newer PTT generation.

## Current platform scope

X11 and XWayland are the implemented activation and text-insertion path. Native Wayland, IBus, and Fcitx5 remain planned. VoiceStand reports unsupported desktop conditions rather than silently claiming global activation.

## Required next gates

1. Integrate the sherpa-onnx Rust online recognizer.
2. Pass fixed-corpus latency and accuracy gates through the Rust backend.
3. Pass browser, editor, and terminal insertion acceptance.
4. Pass device disconnect/reconnect and suspend/resume recovery.
5. Complete a live microphone/output soak in addition to deterministic tests.
6. Add native Wayland/input-method support or clearly constrain the release to X11/XWayland.
