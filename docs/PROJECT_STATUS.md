# VoiceStand Project Status

**Architecture:** CPU-first local Linux dictation

**Updated:** 2026-09-02

**Production status:** end-to-end X11/XWayland path implemented; desktop acceptance and robustness work remain

## Phase ledger

| Phase | Status | Evidence / remaining gate |
|---|---|---|
| 0. Freeze accelerator assumptions | Complete | CPU is the required baseline; Intel hardware is optional/legacy. |
| 1. Real CPU ASR | Complete | `whisper.cpp` backend, real model loading, WAV smoke test, cancellation, metrics. |
| 2. Real PTT | Implemented, acceptance pending | X11/XWayland hold and toggle bindings drive capture; manual desktop validation remains. |
| 3. Real text insertion | Implemented, acceptance pending | `TextSink` plus focused-window `xdotool` backend; browser/editor/terminal matrix remains. |
| 4. Streaming UX | In progress, target failing | Percentile harness implemented. Current 11 s JFK/tiny.en baseline on this host: warm decode p50 20.8 s, RTF 1.89 with four threads. Exact partials finalize immediately; releases up to two seconds beyond a partial decode only a one-second-overlap tail and merge on verified word overlap. Ambiguous or larger gaps retain the full-decode fallback. |
| 5. Backend benchmark | Rust streaming backend and PTT core wiring implemented | sherpa-onnx 1.13.7 with the 20M English Zipformer now provides live partials and immediate release flush, with model auto-discovery and Whisper fallback. A post-integration debug run produced 22 updates, 0.46 RTF, and 356 ms final flush. Broader WER and release-mode percentiles remain. |
| 6. Robustness | In progress | Deterministic 500-session PTT/ASR soak and stale-decode cancellation pass; device/session recovery, suspend/resume, and fault injection remain. |
| 7. Packaging / UX | In progress | Verified archive, user-local installer, desktop file, optional autostart, and model installer implemented; tray/setup UI remains. |
| 8. Optional acceleration | Deferred | Begins only after CPU production gates pass. |

## Current supported path

```text
Ctrl+Alt+V hold or Ctrl+Alt+Space toggle
  -> CPAL microphone capture (normalized to 16 kHz mono)
  -> bounded utterance assembly with pre-roll
  -> partial/final whisper.cpp CPU decoding (temporary live backend)
  -> Ctrl+Alt+V release forces finalization
  -> sanitized transcript typed into the focused X11/XWayland application
```

The compatibility text sink does not insert unstable partials. Final text is passed directly as a process argument without a shell, and newline/control characters are removed so dictation cannot implicitly submit a terminal command.

Native Wayland, IBus, and Fcitx5 are not yet implemented. On a session without X11/XWayland, VoiceStand reports the activation/text backend as unavailable instead of silently falling back to a placeholder.

## Local release gate

`./scripts/ci-local.sh` is the authoritative pre-push gate. It performs:

- shell validation and patch whitespace checks;
- locked workspace check and all workspace tests;
- strict ASR, activation, and text-output linting;
- release build;
- deterministic release archive assembly and checksum verification;
- isolated user-local install, desktop entry, and autostart acceptance.

The GitHub Actions job invokes this same script. A local green run does not claim that a hosted runner has executed successfully; it minimizes hosted iteration by testing the shared logic first.

## Immediate next work

1. Run the manual X11 acceptance matrix in browser, editor, and terminal fields.
2. Exercise the persistent Zipformer backend through complete desktop PTT tests; retain whisper.cpp as explicit fallback.
3. Add audio-device loss/recovery and suspend/resume fault tests.
4. Implement native Wayland/input-method backend selection.
5. Run a live microphone and desktop-output soak in addition to the deterministic 500-session test.

VoiceStand is production-ready only after the complete desktop path survives the acceptance matrix and robustness gates for hundreds of sessions without stuck capture state or manual recovery.
