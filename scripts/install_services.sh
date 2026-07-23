#!/bin/bash
# DoorbellAIVision — Install systemd unit for the Docker pipeline only (no local GUI / kiosk).
#
# Creates: /etc/systemd/system/doorbell-ai.service
# Usage (as your normal login user, not sudo -u root for the whole script):
#   bash scripts/install_services.sh
#
# After git pull or Dockerfile changes:
#   docker compose build && sudo systemctl restart doorbell-ai

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INSTALL_USER="$(whoami)"

echo "DoorbellAIVision — doorbell-ai.service (Docker only)"
echo "  Project: $PROJECT_DIR"
echo "  User:    $INSTALL_USER"
echo ""

if ! groups "$INSTALL_USER" | grep -qw docker; then
    echo "Adding $INSTALL_USER to group docker..."
    sudo usermod -aG docker "$INSTALL_USER"
    echo "  Re-login (or reboot) so group membership applies."
fi

echo "Installing /etc/systemd/system/doorbell-ai.service ..."
sudo tee /etc/systemd/system/doorbell-ai.service > /dev/null << EOF
[Unit]
Description=DoorbellAIVision AI Pipeline (Docker)
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${PROJECT_DIR}
# After updates: docker compose build && sudo systemctl restart doorbell-ai
ExecStart=/usr/bin/docker compose up -d --remove-orphans
ExecStop=/usr/bin/docker compose down
ExecReload=/usr/bin/docker compose build && /usr/bin/docker compose up -d --remove-orphans
TimeoutStartSec=300
User=${INSTALL_USER}
Group=docker
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable doorbell-ai.service
echo "  Enabled doorbell-ai.service (not started yet unless you run systemctl start)."

cd "$PROJECT_DIR"
if ! docker images | grep -q doorbellaivision; then
    echo "Building image (first time)..."
    docker compose build
fi

echo ""
echo "Start now:  sudo systemctl start doorbell-ai"
echo "Status:     sudo systemctl status doorbell-ai"
echo "Logs:       docker compose logs -f"
echo ""
