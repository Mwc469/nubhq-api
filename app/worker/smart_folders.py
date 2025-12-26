"""
Smart Watch Folders

Subfolder-based profile routing:
- Drop in 01_Inbox_ToSort/youtube/ -> uses youtube profile
- Drop in 01_Inbox_ToSort/tiktok/ -> uses tiktok profile
- Drop in 01_Inbox_ToSort/archive/ -> uses archive profile
- Drop in 01_Inbox_ToSort/ (root) -> auto-detect best profile

Also supports:
- Priority folders (urgent/, high/)
- Batch grouping (project_name/)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .quick_profiles import ProcessingProfile, PROFILES, get_profile
from .content_analyzer import ContentAnalyzer


@dataclass
class FolderRouting:
    """Result of folder-based routing"""
    profile: ProcessingProfile
    priority: int                  # 0=normal, 1=high, 2=urgent
    batch_id: Optional[str]        # Grouping for related videos
    auto_detected: bool            # True if profile was auto-detected
    source_folder: str             # The subfolder name
    reason: str                    # Why this profile was chosen


class SmartFolderRouter:
    """Route videos based on folder structure"""

    # Folder name -> profile mapping
    FOLDER_PROFILES = {
        # Platform folders
        'youtube': 'youtube',
        'yt': 'youtube',
        'youtube_4k': 'youtube_4k',
        'yt4k': 'youtube_4k',

        'instagram': 'instagram',
        'ig': 'instagram',
        'insta': 'instagram',

        'reels': 'instagram_reels',
        'ig_reels': 'instagram_reels',

        'tiktok': 'tiktok',
        'tt': 'tiktok',
        'shorts': 'tiktok',  # YouTube Shorts uses same format

        'twitter': 'twitter',
        'x': 'twitter',

        # Style folders
        'archive': 'archive',
        'raw': 'archive',

        'fast': 'fast',
        'quick': 'fast',

        'podcast': 'podcast',
        'pod': 'podcast',

        'cinematic': 'cinematic',
        'film': 'cinematic',
    }

    # Priority folders
    PRIORITY_FOLDERS = {
        'urgent': 2,
        'asap': 2,
        'priority': 2,
        'high': 1,
        'important': 1,
    }

    @classmethod
    def route(cls, video_path: Path, base_input_dir: Path) -> FolderRouting:
        """
        Determine processing based on folder location.

        Args:
            video_path: Full path to the video file
            base_input_dir: The root input directory (01_Inbox_ToSort)

        Returns:
            FolderRouting with profile, priority, and batch info
        """
        # Get relative path from input dir
        try:
            rel_path = video_path.relative_to(base_input_dir)
        except ValueError:
            rel_path = video_path

        # Get folder components
        parts = list(rel_path.parts[:-1])  # Exclude filename

        # Initialize defaults
        profile_name = None
        priority = 0
        batch_id = None
        reason = "default"

        # Check each folder level
        for part in parts:
            folder_lower = part.lower()

            # Check for profile folder
            if folder_lower in cls.FOLDER_PROFILES and profile_name is None:
                profile_name = cls.FOLDER_PROFILES[folder_lower]
                reason = f"folder: {part}/"

            # Check for priority folder
            if folder_lower in cls.PRIORITY_FOLDERS:
                priority = max(priority, cls.PRIORITY_FOLDERS[folder_lower])

            # Any other folder becomes batch_id
            if folder_lower not in cls.FOLDER_PROFILES and folder_lower not in cls.PRIORITY_FOLDERS:
                batch_id = part

        # If no profile from folder, auto-detect
        auto_detected = False
        if profile_name is None:
            try:
                detected_profile, analysis = ContentAnalyzer.get_profile_for_video(video_path)
                profile_name = detected_profile.name
                auto_detected = True
                reason = f"auto-detected: {analysis.content_type.value}"
            except Exception as e:
                logging.warning(f"Auto-detection failed: {e}")
                profile_name = "youtube"
                auto_detected = True
                reason = "fallback to youtube"

        # Get actual profile object
        profile = get_profile(profile_name)
        if not profile:
            profile = get_profile("youtube")
            profile_name = "youtube"
            reason = "fallback to youtube"

        source_folder = "/".join(parts) if parts else "(root)"

        return FolderRouting(
            profile=profile,
            priority=priority,
            batch_id=batch_id,
            auto_detected=auto_detected,
            source_folder=source_folder,
            reason=reason
        )

    @classmethod
    def create_profile_folders(cls, base_dir: Path) -> Dict[str, Path]:
        """Create all profile subfolders in the input directory"""
        created = {}

        # Create main profile folders
        main_profiles = ['youtube', 'instagram', 'tiktok', 'archive', 'fast', 'podcast']
        for profile in main_profiles:
            folder = base_dir / profile
            folder.mkdir(exist_ok=True)
            created[profile] = folder

        # Create priority folders
        for priority in ['urgent', 'high']:
            folder = base_dir / priority
            folder.mkdir(exist_ok=True)
            created[priority] = folder

        return created

    @classmethod
    def get_folder_help(cls) -> str:
        """Get help text explaining folder routing"""
        lines = [
            "Smart Folder Routing",
            "=" * 40,
            "",
            "Drop videos into subfolders for automatic profile selection:",
            "",
            "Platform Folders:",
        ]

        # Group by profile
        profile_folders = {}
        for folder, profile in cls.FOLDER_PROFILES.items():
            if profile not in profile_folders:
                profile_folders[profile] = []
            profile_folders[profile].append(folder)

        for profile, folders in sorted(profile_folders.items()):
            folder_list = ", ".join(f"{f}/" for f in folders[:3])
            lines.append(f"  {folder_list} -> {profile} profile")

        lines.extend([
            "",
            "Priority Folders:",
            "  urgent/, asap/ -> Process first (priority 2)",
            "  high/, important/ -> Process soon (priority 1)",
            "",
            "Batch Grouping:",
            "  project_name/ -> Videos grouped with same batch_id",
            "",
            "Combining:",
            "  youtube/urgent/ -> YouTube profile with urgent priority",
            "  my_project/tiktok/ -> TikTok profile, batch='my_project'",
            "",
            "Root folder -> Auto-detect best profile",
        ])

        return "\n".join(lines)


def setup_smart_folders(input_dir: Path) -> None:
    """Set up smart folder structure in input directory"""
    logging.info(f"Setting up smart folders in {input_dir}")

    created = SmartFolderRouter.create_profile_folders(input_dir)

    print("\nSmart folders created:")
    for name, path in created.items():
        print(f"  {path}/")

    print("\n" + SmartFolderRouter.get_folder_help())
