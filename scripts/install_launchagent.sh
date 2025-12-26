#!/bin/bash
# Install NubHQ Video Processor LaunchAgent
# This makes the processor start automatically when the volume is mounted

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.nubhq.video-processor.plist"
PLIST_SRC="$API_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"

echo "Installing NubHQ Video Processor LaunchAgent..."

# Check if plist exists
if [ ! -f "$PLIST_SRC" ]; then
    echo "Error: $PLIST_SRC not found"
    exit 1
fi

# Create LaunchAgents directory if needed
mkdir -p "$HOME/Library/LaunchAgents"

# Unload existing if present
if launchctl list | grep -q "com.nubhq.video-processor"; then
    echo "Unloading existing LaunchAgent..."
    launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

# Copy plist
echo "Copying plist to $PLIST_DST..."
cp "$PLIST_SRC" "$PLIST_DST"

# Load the agent
echo "Loading LaunchAgent..."
launchctl load "$PLIST_DST"

echo ""
echo "LaunchAgent installed successfully!"
echo ""
echo "The video processor will now:"
echo "  - Start automatically when /Volumes/NUB_Workspace is mounted"
echo "  - Restart if it crashes"
echo "  - Log to /Volumes/NUB_Workspace/.nubhq/processor.log"
echo ""
echo "Commands:"
echo "  Start:   launchctl start com.nubhq.video-processor"
echo "  Stop:    launchctl stop com.nubhq.video-processor"
echo "  Status:  launchctl list | grep nubhq"
echo "  Logs:    tail -f /Volumes/NUB_Workspace/.nubhq/processor.log"
echo ""
echo "To uninstall:"
echo "  launchctl unload ~/Library/LaunchAgents/$PLIST_NAME"
echo "  rm ~/Library/LaunchAgents/$PLIST_NAME"
