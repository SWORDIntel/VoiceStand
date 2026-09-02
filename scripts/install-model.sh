#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_NAME="ggml-tiny.en.bin"
readonly MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${MODEL_NAME}"
readonly MODEL_SHA256="921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f"
readonly CONFIG_ROOT="${XDG_CONFIG_HOME:-${HOME}/.config}/voicestand"
readonly MODEL_DIR="${VOICESTAND_MODEL_DIR:-${CONFIG_ROOT}/models}"
readonly DESTINATION="${MODEL_DIR}/${MODEL_NAME}"

mkdir -p "${MODEL_DIR}"

if [[ -f "${DESTINATION}" ]] && echo "${MODEL_SHA256}  ${DESTINATION}" | sha256sum --check --status; then
    echo "VoiceStand model is already installed: ${DESTINATION}"
    exit 0
fi

command -v curl >/dev/null 2>&1 || {
    echo "curl is required to download the VoiceStand model" >&2
    exit 1
}

temporary="$(mktemp "${MODEL_DIR}/.${MODEL_NAME}.XXXXXX")"
trap 'rm -f "${temporary}"' EXIT

echo "Downloading ${MODEL_NAME} to ${MODEL_DIR}"
curl --fail --location --retry 3 --progress-bar --output "${temporary}" "${MODEL_URL}"
echo "${MODEL_SHA256}  ${temporary}" | sha256sum --check --status || {
    echo "Model checksum verification failed" >&2
    exit 1
}
chmod 0644 "${temporary}"
mv "${temporary}" "${DESTINATION}"
trap - EXIT

echo "Installed verified VoiceStand model: ${DESTINATION}"
