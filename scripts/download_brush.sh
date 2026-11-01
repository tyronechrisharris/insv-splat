#!/usr/bin/env bash
set -euo pipefail

# Download Brush v0.3.0 standalone Vulkan 3DGS trainer/viewer into ./bin/brush_app
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${ROOT_DIR}/bin"
mkdir -p "${BIN_DIR}"

ARCH="$(uname -m)"
OS="$(uname -s)"

if [[ "${OS}" != "Linux" || "${ARCH}" != "x86_64" ]]; then
  echo "Automatic download script targets Linux x86_64. Please grab your platform build from:"
  echo "  https://github.com/ArthurBrussee/brush/releases"
  exit 1
fi

URL="https://github.com/ArthurBrussee/brush/releases/download/v0.3.0/brush-app-x86_64-unknown-linux-gnu.tar.xz"
TMP_ARCHIVE="$(mktemp --suffix=.tar.xz)"
trap 'rm -f "${TMP_ARCHIVE}"' EXIT

echo "Downloading Brush v0.3.0 from ${URL}..."
curl -fL "${URL}" -o "${TMP_ARCHIVE}"
tar -xJf "${TMP_ARCHIVE}" -C "${BIN_DIR}" --strip-components=1
chmod +x "${BIN_DIR}/brush_app"
echo "Installed Brush binary to ${BIN_DIR}/brush_app"
