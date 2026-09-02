#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPOSITORY_ROOT
MODEL_DIR="${1:-${XDG_CONFIG_HOME:-${HOME}/.config}/voicestand/models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17}"
ITERATIONS="${2:-5}"
FINAL_PADDING_MS="${3:-400}"
readonly MODEL_DIR ITERATIONS FINAL_PADDING_MS

cd "${REPOSITORY_ROOT}/rust"
cargo build --release --locked -p voicestand-core --example streaming_ptt_benchmark

while IFS=$'\t' read -r wav reference; do
    target/release/examples/streaming_ptt_benchmark \
        "${MODEL_DIR}" \
        "${REPOSITORY_ROOT}/benchmarks/corpus/${wav}" \
        "${ITERATIONS}" \
        100 \
        "${reference}" \
        "${FINAL_PADDING_MS}"
done < "${REPOSITORY_ROOT}/benchmarks/corpus/references.tsv"
