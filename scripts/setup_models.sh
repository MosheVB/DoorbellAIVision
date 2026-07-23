#!/usr/bin/env bash
# DoorbellAIVision — download YOLO11 weights into ./models/yolo11/ (host-visible).
#
# Usage:
#   ./scripts/setup_models.sh
#   ./scripts/setup_models.sh --force
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${MODELS_DIR:-$PROJECT_DIR/models}"
YOLO11_DIR="$MODELS_DIR/yolo11"
YOLO11_PT_DEFAULT="$YOLO11_DIR/yolo11s.pt"
FORCE_EXPORT_BUILD="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_EXPORT_BUILD="true"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--force]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 2
      ;;
  esac
done

have_nonempty_file() {
  local p="$1"
  [[ -f "$p" && -s "$p" ]]
}

maybe_chown_dir() {
  local dir="$1"
  if [[ -d "$dir" && -w "$dir" ]]; then
    return 0
  fi
  if [[ "$(id -u)" -eq 0 ]]; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    echo "[models] Fixing write perms: chown -R \$(id -u):\$(id -g) '$dir'"
    sudo chown -R "$(id -u):$(id -g)" "$dir"
  else
    echo "[models] ERROR: '$dir' not writable and sudo not available."
    exit 1
  fi
}

if [[ "$FORCE_EXPORT_BUILD" == "true" ]]; then
  rm -f "$YOLO11_PT_DEFAULT" || true
fi

if have_nonempty_file "$YOLO11_PT_DEFAULT"; then
  echo "[models] YOLO11 weights already present → $YOLO11_PT_DEFAULT"
  exit 0
fi

mkdir -p "$YOLO11_DIR"
maybe_chown_dir "$YOLO11_DIR"

echo "[models] Downloading YOLO11 weights → $YOLO11_DIR"
(
  cd "$PROJECT_DIR"
  docker compose run --rm --entrypoint python3 doorbell-ai \
    /app/scripts/ensure_yolo11_weights.py --dest-dir /models/yolo11 --name yolo11s.pt
)

if ! have_nonempty_file "$YOLO11_PT_DEFAULT"; then
  echo "[models] ERROR: missing $YOLO11_PT_DEFAULT"
  exit 1
fi

echo "[models] Done: $YOLO11_PT_DEFAULT"
