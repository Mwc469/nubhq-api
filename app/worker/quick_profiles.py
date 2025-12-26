"""
Quick Mode Profiles for NubHQ Video Processor

Provides preset configurations for fast, no-prompt video processing.
Set NUBHQ_QUICK_MODE environment variable to use a profile.

Example:
    export NUBHQ_QUICK_MODE=youtube
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ProcessingProfile:
    """A complete processing configuration"""
    name: str
    description: str
    resolution: str  # source, 4k, 1080p, 720p, square_1080, vertical_1080
    quality: str     # max, high, medium, low, web
    audio: str       # none, light, standard, loud, youtube
    color: str       # none, auto_correct, warm, cool, vintage, high_contrast
    noise_reduction: str = "none"  # none, light, medium, aggressive
    add_intro: str = "none"  # none, short, standard
    add_outro: str = "none"  # none, short, standard, subscribe


# Built-in profiles
PROFILES: Dict[str, ProcessingProfile] = {
    "youtube": ProcessingProfile(
        name="youtube",
        description="Optimized for YouTube - 1080p, good quality, proper loudness",
        resolution="1080p",
        quality="high",
        audio="youtube",  # -13 LUFS (YouTube standard)
        color="auto_correct",
        noise_reduction="light",
    ),
    "youtube_4k": ProcessingProfile(
        name="youtube_4k",
        description="YouTube 4K - Maximum quality for 4K uploads",
        resolution="4k",
        quality="max",
        audio="youtube",
        color="auto_correct",
        noise_reduction="light",
    ),
    "instagram": ProcessingProfile(
        name="instagram",
        description="Square format for Instagram feed",
        resolution="square_1080",
        quality="medium",
        audio="loud",
        color="warm",
        noise_reduction="none",
    ),
    "instagram_reels": ProcessingProfile(
        name="instagram_reels",
        description="Vertical 9:16 for Instagram Reels",
        resolution="vertical_1080",
        quality="medium",
        audio="loud",
        color="high_contrast",
        noise_reduction="none",
    ),
    "tiktok": ProcessingProfile(
        name="tiktok",
        description="Optimized for TikTok - punchy and loud",
        resolution="vertical_1080",
        quality="medium",
        audio="loud",
        color="high_contrast",
        noise_reduction="none",
    ),
    "twitter": ProcessingProfile(
        name="twitter",
        description="Compressed for Twitter/X uploads",
        resolution="1080p",
        quality="web",
        audio="standard",
        color="auto_correct",
        noise_reduction="none",
    ),
    "archive": ProcessingProfile(
        name="archive",
        description="Maximum quality preservation for archival",
        resolution="source",
        quality="max",
        audio="light",
        color="none",
        noise_reduction="none",
    ),
    "fast": ProcessingProfile(
        name="fast",
        description="Quick processing with minimal changes",
        resolution="source",
        quality="medium",
        audio="light",
        color="none",
        noise_reduction="none",
    ),
    "podcast": ProcessingProfile(
        name="podcast",
        description="Optimized for talking head / podcast content",
        resolution="1080p",
        quality="high",
        audio="standard",  # -14 LUFS
        color="auto_correct",
        noise_reduction="medium",
    ),
    "cinematic": ProcessingProfile(
        name="cinematic",
        description="Film-like look with color grading",
        resolution="1080p",
        quality="high",
        audio="standard",
        color="cool",
        noise_reduction="light",
    ),
}


def get_profile(name: str) -> Optional[ProcessingProfile]:
    """Get a profile by name"""
    return PROFILES.get(name.lower())


def get_quick_mode_profile() -> Optional[ProcessingProfile]:
    """Get the profile from NUBHQ_QUICK_MODE environment variable"""
    mode = os.environ.get("NUBHQ_QUICK_MODE", "").lower()
    if mode:
        return get_profile(mode)
    return None


def is_quick_mode_enabled() -> bool:
    """Check if quick mode is enabled"""
    return bool(os.environ.get("NUBHQ_QUICK_MODE"))


def list_profiles() -> Dict[str, str]:
    """List all available profiles with descriptions"""
    return {name: profile.description for name, profile in PROFILES.items()}


def profile_to_decisions(profile: ProcessingProfile) -> Dict[str, str]:
    """Convert a profile to decision dictionary for the processor"""
    return {
        "OUTPUT_RESOLUTION": profile.resolution,
        "OUTPUT_QUALITY": profile.quality,
        "AUDIO_NORMALIZE": profile.audio,
        "COLOR_GRADE": profile.color,
        "NOISE_REDUCTION": profile.noise_reduction,
        "ADD_INTRO": profile.add_intro,
        "ADD_OUTRO": profile.add_outro,
    }
