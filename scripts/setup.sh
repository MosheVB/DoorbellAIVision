#!/bin/bash
# DoorbellAIVision - Setup Script
# Configures the Jetson for running AI inference on an RTSP camera feed
# using jetson-containers (JetPack 6 / L4T R36.x)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " DoorbellAIVision - Jetson Setup"
echo " JetPack 6 / L4T R36 / CUDA 12.6"
echo "========================================"

# ── Step 1: Set NVIDIA as the default Docker runtime ──────────────────────────
echo ""
echo "[1/4] Setting nvidia as default Docker runtime..."

DAEMON_JSON=/etc/docker/daemon.json
sudo python3 -c "
import json
with open('$DAEMON_JSON', 'r') as f:
    cfg = json.load(f)
cfg['default-runtime'] = 'nvidia'
with open('$DAEMON_JSON', 'w') as f:
    json.dump(cfg, f, indent=4)
print('  daemon.json updated.')
"

echo "  Restarting Docker..."
sudo systemctl restart docker
echo "  Docker restarted."

# ── Step 1b: Ensure pip3 is installed ─────────────────────────────────────────
echo ""
echo "[1b/4] Ensuring pip3 is installed..."
if ! command -v pip3 &>/dev/null; then
    sudo apt-get install -y python3-pip
fi
pip3 --version

# ── Step 2: Clone jetson-containers ───────────────────────────────────────────
echo ""
echo "[2/4] Cloning jetson-containers..."

JETSON_CONTAINERS_DIR="$HOME/jetson-containers"
if [ -d "$JETSON_CONTAINERS_DIR" ]; then
    echo "  Already cloned at $JETSON_CONTAINERS_DIR, pulling latest..."
    git -C "$JETSON_CONTAINERS_DIR" pull
else
    git clone --depth=1 https://github.com/dusty-nv/jetson-containers "$JETSON_CONTAINERS_DIR"
fi

# ── Step 3: Install jetson-containers Python dependencies ─────────────────────
echo ""
echo "[3/4] Installing jetson-containers host tools..."

pip3 install -q --upgrade pip
pip3 install -q -r "$JETSON_CONTAINERS_DIR/requirements.txt"

# ── Step 4: Pull the inference container image ────────────────────────────────
echo ""
echo "[4/4] Pulling the jetson-inference container for JetPack 6..."
echo "  (This may take a while — ~6GB image)"

docker pull dustynv/jetson-inference:r36.3.0

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Edit config/camera.env with your RTSP URL"
echo "   2. Run:  docker compose up"
echo "========================================"
