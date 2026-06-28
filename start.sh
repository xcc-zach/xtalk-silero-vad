#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_WORKERS="${VAD_MODEL_WORKERS:-}"
USE_ONNX="${VAD_USE_ONNX:-1}"
DEVICE="${VAD_DEVICE:-cpu}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --model-workers)
      MODEL_WORKERS="$2"
      shift 2
      ;;
    --torch)
      USE_ONNX="0"
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --cpu)
      DEVICE="cpu"
      shift
      ;;
    --cuda|--gpu)
      DEVICE="cuda"
      USE_ONNX="0"
      shift
      ;;
    --onnx)
      USE_ONNX="1"
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash start.sh [--host 0.0.0.0] [--port 8000] [--model-workers 4] [--device cpu|cuda|auto] [--onnx|--torch]
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ ! -f "$ROOT_DIR/.venv/bin/activate" ]; then
  echo ".venv not found. Run: bash install.sh" >&2
  exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"

export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONUSERBASE PYTHONPATH
if [ "${PRESERVE_LD_LIBRARY_PATH:-0}" != "1" ]; then
  unset LD_LIBRARY_PATH
fi

case "$DEVICE" in
  cuda|cuda:*)
    USE_ONNX="0"
    ;;
esac

if [ -n "$MODEL_WORKERS" ]; then
  export VAD_MODEL_WORKERS="$MODEL_WORKERS"
fi
export VAD_USE_ONNX="$USE_ONNX"
export VAD_DEVICE="$DEVICE"
export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR"

exec python -m uvicorn serving.server:app --host "$HOST" --port "$PORT"
