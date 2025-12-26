#!/bin/bash
# NubHQ Video Workspace Setup
# Creates the folder structure for video processing

set -e

BASE="${NUBHQ_WORKSPACE:-/Volumes/NUB_Workspace}"

echo "Setting up NubHQ workspace at: $BASE"

# Check if path exists or can be created
if [ ! -d "$BASE" ]; then
    # Try to create it if it's a local path (not a volume)
    if [[ "$BASE" == /Volumes/* ]]; then
        echo "Error: Volume $BASE is not mounted"
        echo "Please mount the volume or set NUBHQ_WORKSPACE to a local path"
        exit 1
    else
        echo "Creating base directory: $BASE"
        mkdir -p "$BASE" || {
            echo "Error: Cannot create $BASE"
            exit 1
        }
    fi
fi

# Create folder structure
echo "Creating folders..."

mkdir -p "$BASE/input"
mkdir -p "$BASE/processing"
mkdir -p "$BASE/output/auto-queued"
mkdir -p "$BASE/output/review"
mkdir -p "$BASE/output/library"
mkdir -p "$BASE/output/failed"
mkdir -p "$BASE/output/thumbnails"
mkdir -p "$BASE/output/captions"
mkdir -p "$BASE/output/watermarked"
mkdir -p "$BASE/output/templates"
mkdir -p "$BASE/archive"
mkdir -p "$BASE/.nubhq"

# Set permissions
chmod 755 "$BASE/input"
chmod 755 "$BASE/output"

echo ""
echo "Workspace ready!"
echo ""
echo "Folder structure:"
echo "  $BASE/"
echo "  ├── input/           <- Drop videos here"
echo "  ├── processing/      <- Videos being processed"
echo "  ├── output/"
echo "  │   ├── auto-queued/ <- High-confidence outputs"
echo "  │   ├── review/      <- Needs manual review"
echo "  │   ├── library/     <- Final approved videos"
echo "  │   └── failed/      <- Failed processing"
echo "  ├── archive/         <- Original files"
echo "  └── .nubhq/          <- System data"
echo ""
echo "To start processing, drop .mp4/.mov files into:"
echo "  $BASE/input"
