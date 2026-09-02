#!/usr/bin/env bash
set -euo pipefail

if ! command -v xdotool >/dev/null 2>&1; then
    echo "VoiceStand requires xdotool for the current X11/XWayland desktop backend." >&2
    echo "Install it with: sudo apt-get install xdotool" >&2
    exit 1
fi

PREFIX="${XDG_BIN_HOME:-${HOME}/.local/bin}"
APPLICATIONS_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
AUTOSTART_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/autostart"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

install -d "${PREFIX}" "${APPLICATIONS_DIR}"
install -m 0755 "${SOURCE_DIR}/voicestand" "${PREFIX}/voicestand"
sed "s|^Exec=.*|Exec=${PREFIX}/voicestand|" "${SOURCE_DIR}/voicestand.desktop" \
    > "${APPLICATIONS_DIR}/voicestand.desktop"
chmod 0644 "${APPLICATIONS_DIR}/voicestand.desktop"

if [[ "${1:-}" == "--autostart" ]]; then
    install -d "${AUTOSTART_DIR}"
    install -m 0644 "${APPLICATIONS_DIR}/voicestand.desktop" \
        "${AUTOSTART_DIR}/voicestand.desktop"
fi

echo "Installed VoiceStand to ${PREFIX}/voicestand"
echo "Install the speech model with: ${SOURCE_DIR}/install-model.sh"
