#!/bin/bash
# Install SpeechScore API as a systemd service
# Run with: sudo bash deploy/install-service.sh

set -e

SERVICE_FILE="deploy/speechscore.service"
DEST="/etc/systemd/system/speechscore.service"

if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Run this from the speech3 directory"
    exit 1
fi

echo "Installing SpeechScore API service..."
cp "$SERVICE_FILE" "$DEST"
systemctl daemon-reload
systemctl enable speechscore
echo ""
echo "Service installed. Commands:"
echo "  sudo systemctl start speechscore    # Start"
echo "  sudo systemctl stop speechscore     # Stop"
echo "  sudo systemctl restart speechscore  # Restart"
echo "  sudo systemctl status speechscore   # Status"
echo "  journalctl -u speechscore -f        # Logs"
echo ""
echo "Start now? (y/N)"
read -r answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    systemctl start speechscore
    sleep 3
    systemctl status speechscore --no-pager
fi
