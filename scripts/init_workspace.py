#!/usr/bin/env python3
"""
Initialize NubHQ Video Workspace Folder Structure
=================================================
Run this script on the Mac mini to create all required directories
for the video processing pipeline.

Usage:
    python scripts/init_workspace.py [--base-path /Volumes/NUB_Workspace]
"""

import argparse
import os
from pathlib import Path


FOLDERS = [
    "input",              # Drop zone for raw footage
    "processing",         # Active FFmpeg work (symlink to NVME in production)
    "output/auto-queued", # High confidence → approval queue
    "output/review",      # Low confidence → manual review
    "output/library",     # Processed & approved clips
    "archive",            # Original files post-processing
    "templates",          # Compilation templates (intro/outro clips)
    ".nubhq",             # Config and databases
    ".nubhq/sync_cache",  # Audio fingerprint cache
]


def init_workspace(base_path: str, dry_run: bool = False):
    """Create all workspace directories."""
    base = Path(base_path)

    print(f"Initializing NubHQ workspace at: {base}")
    print("-" * 50)

    if not base.exists():
        if dry_run:
            print(f"[DRY RUN] Would create base: {base}")
        else:
            base.mkdir(parents=True)
            print(f"Created base: {base}")

    for folder in FOLDERS:
        folder_path = base / folder
        if folder_path.exists():
            print(f"✓ Exists: {folder}")
        else:
            if dry_run:
                print(f"[DRY RUN] Would create: {folder}")
            else:
                folder_path.mkdir(parents=True)
                print(f"✓ Created: {folder}")

    # Create .gitignore in .nubhq to exclude databases from version control
    gitignore_path = base / ".nubhq" / ".gitignore"
    if not gitignore_path.exists() and not dry_run:
        gitignore_path.write_text("*.db\n*.db-journal\nsync_cache/\n")
        print("✓ Created .nubhq/.gitignore")

    print("-" * 50)
    print("Workspace initialization complete!")
    print()
    print("Next steps:")
    print("1. Set environment variable: export NUBHQ_WORKSPACE=/Volumes/NUB_Workspace")
    print("2. For NVME acceleration, symlink processing folder:")
    print("   rm -rf /Volumes/NUB_Workspace/processing")
    print("   ln -s /Volumes/NVME_Drive/nubhq_processing /Volumes/NUB_Workspace/processing")
    print("3. Start the video processor:")
    print("   python -m app.worker.intelligent_processor")


def main():
    parser = argparse.ArgumentParser(description="Initialize NubHQ video workspace")
    parser.add_argument(
        "--base-path",
        default="/Volumes/NUB_Workspace",
        help="Base path for workspace (default: /Volumes/NUB_Workspace)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without making changes"
    )
    args = parser.parse_args()

    init_workspace(args.base_path, args.dry_run)


if __name__ == "__main__":
    main()
