# Speech benchmark corpus

These upstream fixtures are 16-bit, 16 kHz, mono PCM WAV files. Checksums make
the local latency corpus reproducible and detect silent upstream replacements.

| File | Duration | Upstream source | SHA-256 |
|---|---:|---|---|
| `jfk.wav` | 11.000 s | [whisper.cpp canonical sample](https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav) | `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` |
| `sherpa-en-16k.wav` | 3.845 s | [sherpa-onnx ASR fixture](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en-16k.wav) | `7dc81c7a113dc64f0a7201f1d2cb78157d1635c415ab779a35ec07690bc8241c` |
| `sherpa-whisper-0.wav` | 6.625 s | [sherpa-onnx Whisper fixture](https://huggingface.co/csukuangfj/sherpa-onnx-whisper-tiny.en/resolve/main/test_wavs/0.wav) | `6bc58a4efdf20daac252b6b1502632601a71efe0308f6757dc1eda34891a7e4f` |

Verify the corpus from the repository root:

```bash
sha256sum -c benchmarks/corpus/SHA256SUMS
```
