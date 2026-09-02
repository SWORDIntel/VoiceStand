#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPOSITORY_ROOT
VERSION="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "${REPOSITORY_ROOT}/rust/Cargo.toml" | head -n 1)"
readonly VERSION
ARCHITECTURE="$(uname -m)"
readonly ARCHITECTURE
readonly PACKAGE_NAME="voicestand-${VERSION}-linux-${ARCHITECTURE}"
readonly DIST_DIR="${REPOSITORY_ROOT}/dist"
readonly ARCHIVE="${DIST_DIR}/${PACKAGE_NAME}.tar.gz"

[[ -n "${VERSION}" ]] || { echo "Unable to determine workspace version" >&2; exit 1; }
[[ -x "${REPOSITORY_ROOT}/rust/target/release/voicestand" ]] || {
    echo "Release binary missing; run scripts/ci-local.sh first" >&2
    exit 1
}

staging_root="$(mktemp -d)"
verification_root="$(mktemp -d)"
trap 'rm -rf "${staging_root}" "${verification_root}"' EXIT
package_root="${staging_root}/${PACKAGE_NAME}"
mkdir -p "${package_root}"

install -m 0755 "${REPOSITORY_ROOT}/rust/target/release/voicestand" "${package_root}/voicestand"
install -m 0755 "${REPOSITORY_ROOT}/scripts/install-model.sh" "${package_root}/install-model.sh"
install -m 0755 "${REPOSITORY_ROOT}/packaging/install.sh" "${package_root}/install.sh"
install -m 0644 "${REPOSITORY_ROOT}/packaging/voicestand.desktop" "${package_root}/voicestand.desktop"
install -m 0644 "${REPOSITORY_ROOT}/README.md" "${package_root}/README.md"
install -m 0644 "${REPOSITORY_ROOT}/LICENSE" "${package_root}/LICENSE"

(cd "${package_root}" && sha256sum voicestand install-model.sh install.sh voicestand.desktop README.md LICENSE > SHA256SUMS)
mkdir -p "${DIST_DIR}"
tar --sort=name --owner=0 --group=0 --numeric-owner -czf "${ARCHIVE}" -C "${staging_root}" "${PACKAGE_NAME}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"

tar -xzf "${ARCHIVE}" -C "${verification_root}"
(cd "${verification_root}/${PACKAGE_NAME}" && sha256sum --check --status SHA256SUMS)
"${verification_root}/${PACKAGE_NAME}/voicestand" --version >/dev/null
test_home="${verification_root}/test-home"
XDG_BIN_HOME="${test_home}/bin" \
XDG_DATA_HOME="${test_home}/share" \
XDG_CONFIG_HOME="${test_home}/config" \
HOME="${test_home}" \
    "${verification_root}/${PACKAGE_NAME}/install.sh" --autostart >/dev/null
"${test_home}/bin/voicestand" --version >/dev/null
grep -Fq "Exec=${test_home}/bin/voicestand" "${test_home}/share/applications/voicestand.desktop"
test -f "${test_home}/config/autostart/voicestand.desktop"

echo "Created and verified ${ARCHIVE}"
