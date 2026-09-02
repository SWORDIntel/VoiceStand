#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPOSITORY_ROOT
cd "${REPOSITORY_ROOT}"

echo "[ci] validating shell scripts"
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck scripts/ci-local.sh scripts/install-model.sh scripts/benchmark-streaming.sh
else
    bash -n scripts/ci-local.sh scripts/install-model.sh scripts/benchmark-streaming.sh
fi

echo "[ci] validating patch whitespace"
git diff --check

cd rust

echo "[ci] checking locked workspace"
cargo check --workspace --locked

echo "[ci] testing locked workspace"
cargo test --workspace --locked

echo "[ci] applying strict lint gates to production boundaries"
cargo clippy --locked -p voicestand-asr -p voicestand-state -p voicestand-text --all-targets -- -D warnings

echo "[ci] building the release executable"
cargo build --locked --release -p voicestand

echo "[ci] packaging and verifying the release bundle"
cd "${REPOSITORY_ROOT}"
scripts/package-release.sh

echo "[ci] all local CI gates passed"
