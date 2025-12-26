#!/bin/bash
# NubHQ Video Workspace Setup
# Creates the folder structure for video processing

set -e

WORK_DRIVE="${NUBHQ_WORK:-/Volumes/Lil Hoss/NubHQ}"
STORAGE_DRIVE="${NUBHQ_STORAGE:-/Volumes/Big Hoss/NubHQ}"

echo "Setting up NubHQ workspace..."
echo "  Work drive: $WORK_DRIVE"
echo "  Storage drive: $STORAGE_DRIVE"

# Check work drive
if [ ! -d "/Volumes/Lil Hoss" ] && [[ "$WORK_DRIVE" == /Volumes/* ]]; then
    echo "Error: Lil Hoss not mounted"
    exit 1
fi

# Check storage drive
if [ ! -d "/Volumes/Big Hoss" ] && [[ "$STORAGE_DRIVE" == /Volumes/* ]]; then
    echo "Error: Big Hoss not mounted"
    exit 1
fi

echo "Creating folders..."

# Lil Hoss (Working Drive) - Active processing only
mkdir -p "$WORK_DRIVE/01_Inbox_ToSort"
mkdir -p "$WORK_DRIVE/01_Inbox_ToSort/processing"
mkdir -p "$WORK_DRIVE/temp"
mkdir -p "$WORK_DRIVE/cache"
mkdir -p "$WORK_DRIVE/logs"

# Big Hoss (Storage Drive) - Everything else
mkdir -p "$STORAGE_DRIVE/02_Library"
mkdir -p "$STORAGE_DRIVE/03_Exports"
mkdir -p "$STORAGE_DRIVE/03_Exports/auto-queued"
mkdir -p "$STORAGE_DRIVE/03_Exports/review"
mkdir -p "$STORAGE_DRIVE/03_Exports/thumbnails"
mkdir -p "$STORAGE_DRIVE/03_Exports/captions"
mkdir -p "$STORAGE_DRIVE/04_Archive"
mkdir -p "$STORAGE_DRIVE/05_Metadata"
mkdir -p "$STORAGE_DRIVE/database"

echo ""
echo "Workspace ready!"
echo ""
echo "Two-drive setup:"
echo ""
echo "  Lil Hoss (Working Drive):"
echo "  └── NubHQ/"
echo "      ├── 01_Inbox_ToSort/  <- Drop videos here"
echo "      ├── temp/             <- Processing temp files"
echo "      ├── cache/            <- FFmpeg cache"
echo "      └── logs/             <- Processing logs"
echo ""
echo "  Big Hoss (Storage Drive):"
echo "  └── NubHQ/"
echo "      ├── 02_Library/       <- Approved videos"
echo "      ├── 03_Exports/       <- Processed outputs"
echo "      ├── 04_Archive/       <- Long-term archive"
echo "      ├── 05_Metadata/      <- Video metadata"
echo "      └── database/         <- Learning DB"
echo ""
echo "To start processing, drop .mp4/.mov files into:"
echo "  $WORK_DRIVE/01_Inbox_ToSort"
