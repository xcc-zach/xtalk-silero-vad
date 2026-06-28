#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$ROOT_DIR/$VENV_DIR/bin/python"

pip_index_args=()
if [ -n "${PIP_INDEX_URL:-}" ]; then
  pip_index_args+=(--index-url "$PIP_INDEX_URL")
fi
if [ -n "${PIP_EXTRA_INDEX_URL:-}" ]; then
  pip_index_args+=(--extra-index-url "$PIP_EXTRA_INDEX_URL")
fi
if [ -n "${PIP_TRUSTED_HOST:-}" ]; then
  for host in $PIP_TRUSTED_HOST; do
    pip_index_args+=(--trusted-host "$host")
  done
fi

pip_no_proxy() {
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u FTP_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u ftp_proxy -u all_proxy -u no_proxy \
    PIP_CONFIG_FILE=/dev/null \
    "$VENV_PYTHON" -m pip --isolated "$@" "${pip_index_args[@]}"
}

echo "Installing Python packages with proxy environment disabled for pip..."
pip_no_proxy install --upgrade pip setuptools wheel

if [ "${INSTALL_GPU_TORCH:-0}" = "1" ]; then
  pip_no_proxy install --upgrade --force-reinstall "torch>=1.12.0" "torchaudio>=0.12.0,<2.10"
else
  pip_no_proxy install --index-url https://download.pytorch.org/whl/cpu "torch>=1.12.0" "torchaudio>=0.12.0,<2.10"
fi

pip_no_proxy install -e ".[onnx-cpu]" \
  "fastapi>=0.110,<1" \
  "uvicorn[standard]>=0.27,<1" \
  "numpy>=1.23"

echo "Install finished. Start the service with: bash start.sh --port 8000"
