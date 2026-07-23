#!/usr/bin/env bash
# Replay a file from the mounted output dir (container path /output/...).
# Example:  ./scripts/replay_output.sh /output/test.mp4
set -euo pipefail
cd "$(dirname "$0")/.."
VID="${1:-/output/test.mp4}"
exec docker compose run --rm \
  -e "INPUT_VIDEO=${VID}" \
  -e SAVE_DETECTIONS=true \
  -e RECORD_RTSP=false \
  -e DISABLE_PKG_CAM=true \
  -e ENABLE_WEB=false \
  --entrypoint python3 doorbell-ai /app/scripts/analyze.py
