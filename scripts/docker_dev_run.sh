#!/usr/bin/env bash
# Bounded live RTSP run: annotated video + auto-stop (does not rely on camera.env toggles).
# Usage: ./scripts/docker_dev_run.sh
#        FILE_MAX_FRAMES=5000 ./scripts/docker_dev_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${FILE_MAX_FRAMES:=3600}"
exec docker compose run --rm \
  -e "FILE_MAX_FRAMES=${FILE_MAX_FRAMES}" \
  -e SAVE_DETECTIONS=true \
  -e ENABLE_WEB=false \
  -e DISABLE_PKG_CAM=true \
  --entrypoint python3 doorbell-ai /app/scripts/analyze.py
