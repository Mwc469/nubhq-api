#!/bin/bash
# NubHQ Video Workspace Setup
# Creates the folder structure for video processing

set -e

BASE="${NUBHQ_WORKSPACE:-/Volumes/Lil Hoss/NubHQ}"

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

# Create folder structure matching existing NubHQ layout
echo "Creating folders..."

# Working drive (Lil Hoss)
mkdir -p "$BASE/01_Inbox_ToSort"
mkdir -p "$BASE/01_Inbox_ToSort/processing"
mkdir -p "$BASE/02_Library"
mkdir -p "$BASE/03_Exports"
mkdir -p "$BASE/03_Exports/auto-queued"
mkdir -p "$BASE/03_Exports/review"
mkdir -p "$BASE/03_Exports/thumbnails"
mkdir -p "$BASE/03_Exports/captions"
mkdir -p "$BASE/04_Archive"
mkdir -p "$BASE/05_Metadata"
mkdir -p "$BASE/database"
mkdir -p "$BASE/logs"
mkdir -p "$BASE/temp"
mkdir -p "$BASE/cache"

# Archive drive (Big Hoss) - if available
ARCHIVE_BASE="/Volumes/Big Hoss/NubHQ"
if [ -d "/Volumes/Big Hoss" ]; then
    mkdir -p "$ARCHIVE_BASE/04_Archive"
    echo "Archive drive (Big Hoss) configured"
fi

echo ""
echo "Workspace ready!"
echo ""
echo "Two-drive setup:"
echo ""
echo "  Lil Hoss (Working Drive):"
echo "  └── NubHQ/"
echo "      ├── 01_Inbox_ToSort/  <- Drop videos here"
echo "      ├── 02_Library/       <- Approved videos"
echo "      ├── 03_Exports/       <- Processed outputs"
echo "      ├── 04_Archive/       <- Local archive"
echo "      ├── 05_Metadata/      <- Video metadata"
echo "      ├── database/         <- Learning DB"
echo "      └── logs/             <- Processing logs"
echo ""
echo "  Big Hoss (Archive Drive):"
echo "  └── NubHQ/"
echo "      └── 04_Archive/       <- Long-term archive"
echo ""
echo "To start processing, drop .mp4/.mov files into:"
echo "  $BASE/01_Inbox_ToSort"
