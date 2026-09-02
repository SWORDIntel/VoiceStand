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
| 4. Streaming UX | In progress, p95 release gate narrowly failing | Complete release-mode orchestration on JFK: first partial p50/p95 648/932 ms; release-to-final p50/p95 227/761 ms. Median passes, but p95 exceeds the 700 ms target by 61 ms. |
| 5. Backend benchmark | Corpus harness implemented; broader accuracy pending | The release harness drives the real audio pipeline and PTT lifecycle, reports percentiles and WER, and caught/fixed a release-only missing-pre-roll bug. Fixed references cover all three current WAV fixtures. |
| 6. Robustness | In progress | Deterministic 500-session PTT/ASR soak and stale-decode cancellation pass; device/session recovery, suspend/resume, and fault injection remain. |
| 7. Packaging / UX | In progress | Verified archive, user-local installer, desktop file, optional autostart, and model installer implemented; tray/setup UI remains. |
| 8. Optional acceleration | Deferred | Begins only after CPU production gates pass. |

## Current supported path

```text
Ctrl+Alt+V hold or Ctrl+Alt+Space toggle
  -> CPAL microphone capture (normalized to 16 kHz mono)
  -> bounded utterance assembly with pre-roll
  -> persistent sherpa-onnx Zipformer streaming with live partials
  -> whisper.cpp CPU final decode when streaming is unavailable
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
2. Reduce release-to-final p95 below 700 ms and expand the reference corpus beyond three fixtures.
3. Add audio-device loss/recovery and suspend/resume fault tests.
4. Implement native Wayland/input-method backend selection.
5. Run a live microphone and desktop-output soak in addition to the deterministic 500-session test.

VoiceStand is production-ready only after the complete desktop path survives the acceptance matrix and robustness gates for hundreds of sessions without stuck capture state or manual recovery.
