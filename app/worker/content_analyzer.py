"""
Content-Aware Profile Selection

Analyzes video content and automatically selects the best processing profile.
Detects: talking head, action/sports, cinematic, tutorial, music video, etc.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from .quick_profiles import ProcessingProfile, get_profile


class ContentType(Enum):
    """Detected content types"""
    TALKING_HEAD = "talking_head"      # Podcast, interview, vlog
    ACTION = "action"                   # Sports, fast motion
    CINEMATIC = "cinematic"             # Film-like, slow, artistic
    TUTORIAL = "tutorial"               # Screen recording, how-to
    MUSIC_VIDEO = "music_video"         # Music-focused content
    GAMING = "gaming"                   # Game footage
    NATURE = "nature"                   # Landscape, wildlife
    UNKNOWN = "unknown"


@dataclass
class ContentAnalysis:
    """Results of content analysis"""
    content_type: ContentType
    confidence: float
    aspect_ratio: str           # landscape, portrait, square
    has_face: bool
    face_time_ratio: float      # Portion of video with faces
    motion_level: str           # low, medium, high
    speech_ratio: float         # Portion with speech
    music_ratio: float          # Portion with music
    scene_change_rate: float    # Scene changes per minute
    avg_shot_duration: float    # Average shot length in seconds
    is_screen_recording: bool
    suggested_profile: str
    reasons: list


class ContentAnalyzer:
    """Analyze video content to determine optimal processing profile"""

    # Profile mapping based on content type and aspect ratio
    PROFILE_MAP = {
        # (content_type, aspect_ratio) -> profile_name
        (ContentType.TALKING_HEAD, "landscape"): "podcast",
        (ContentType.TALKING_HEAD, "portrait"): "tiktok",
        (ContentType.TALKING_HEAD, "square"): "instagram",

        (ContentType.ACTION, "landscape"): "youtube",
        (ContentType.ACTION, "portrait"): "instagram_reels",
        (ContentType.ACTION, "square"): "instagram",

        (ContentType.CINEMATIC, "landscape"): "cinematic",
        (ContentType.CINEMATIC, "portrait"): "tiktok",
        (ContentType.CINEMATIC, "square"): "instagram",

        (ContentType.TUTORIAL, "landscape"): "youtube",
        (ContentType.TUTORIAL, "portrait"): "tiktok",
        (ContentType.TUTORIAL, "square"): "instagram",

        (ContentType.MUSIC_VIDEO, "landscape"): "youtube",
        (ContentType.MUSIC_VIDEO, "portrait"): "tiktok",
        (ContentType.MUSIC_VIDEO, "square"): "instagram",

        (ContentType.GAMING, "landscape"): "youtube",
        (ContentType.GAMING, "portrait"): "tiktok",
        (ContentType.GAMING, "square"): "instagram",

        (ContentType.NATURE, "landscape"): "cinematic",
        (ContentType.NATURE, "portrait"): "instagram_reels",
        (ContentType.NATURE, "square"): "instagram",

        (ContentType.UNKNOWN, "landscape"): "youtube",
        (ContentType.UNKNOWN, "portrait"): "tiktok",
        (ContentType.UNKNOWN, "square"): "instagram",
    }

    @classmethod
    def analyze(cls, video_path: Path) -> ContentAnalysis:
        """Full content analysis of a video"""
        path_str = str(video_path)

        # Get basic video info
        probe_data = cls._probe_video(path_str)

        # Extract dimensions
        video_stream = next(
            (s for s in probe_data.get('streams', []) if s['codec_type'] == 'video'),
            None
        )

        if not video_stream:
            raise ValueError("No video stream found")

        width = int(video_stream.get('width', 1920))
        height = int(video_stream.get('height', 1080))
        duration = float(probe_data.get('format', {}).get('duration', 0))

        # Determine aspect ratio
        aspect_ratio = cls._get_aspect_ratio(width, height)

        # Analyze motion
        motion_info = cls._analyze_motion(path_str, duration)

        # Analyze audio content
        audio_info = cls._analyze_audio_content(path_str)

        # Detect faces (simplified - checks for face-like regions)
        face_info = cls._detect_faces(path_str, duration)

        # Check if screen recording
        is_screen = cls._detect_screen_recording(path_str, width, height)

        # Determine content type
        content_type, confidence, reasons = cls._classify_content(
            motion_info, audio_info, face_info, is_screen, duration
        )

        # Get suggested profile
        profile_key = (content_type, aspect_ratio)
        suggested_profile = cls.PROFILE_MAP.get(profile_key, "youtube")

        return ContentAnalysis(
            content_type=content_type,
            confidence=confidence,
            aspect_ratio=aspect_ratio,
            has_face=face_info['has_face'],
            face_time_ratio=face_info['face_ratio'],
            motion_level=motion_info['level'],
            speech_ratio=audio_info['speech_ratio'],
            music_ratio=audio_info['music_ratio'],
            scene_change_rate=motion_info['scene_rate'],
            avg_shot_duration=motion_info['avg_shot'],
            is_screen_recording=is_screen,
            suggested_profile=suggested_profile,
            reasons=reasons
        )

    @classmethod
    def get_profile_for_video(cls, video_path: Path) -> Tuple[ProcessingProfile, ContentAnalysis]:
        """Analyze video and return the best processing profile"""
        analysis = cls.analyze(video_path)
        profile = get_profile(analysis.suggested_profile)

        if not profile:
            profile = get_profile("youtube")  # fallback

        return profile, analysis

    @staticmethod
    def _probe_video(path: str) -> dict:
        """Get video metadata with ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return json.loads(result.stdout)
        except Exception as e:
            logging.error(f"Probe failed: {e}")
            return {}

    @staticmethod
    def _get_aspect_ratio(width: int, height: int) -> str:
        """Classify aspect ratio"""
        ratio = width / height if height > 0 else 1.0

        if ratio > 1.2:
            return "landscape"
        elif ratio < 0.8:
            return "portrait"
        else:
            return "square"

    @staticmethod
    def _analyze_motion(path: str, duration: float) -> Dict:
        """Analyze motion intensity and scene changes"""
        cmd = [
            'ffmpeg', '-i', path, '-vf',
            "select='gt(scene,0.3)',showinfo",
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            scene_changes = result.stderr.count('pts_time:')

            # Calculate metrics
            scene_rate = (scene_changes / duration) * 60 if duration > 0 else 0
            avg_shot = duration / max(scene_changes, 1)

            # Classify motion level
            if scene_rate > 20:
                level = "high"
            elif scene_rate > 8:
                level = "medium"
            else:
                level = "low"

            return {
                'scene_changes': scene_changes,
                'scene_rate': scene_rate,
                'avg_shot': avg_shot,
                'level': level,
            }
        except Exception as e:
            logging.warning(f"Motion analysis failed: {e}")
            return {'scene_changes': 0, 'scene_rate': 0, 'avg_shot': 10, 'level': 'medium'}

    @staticmethod
    def _analyze_audio_content(path: str) -> Dict:
        """Analyze audio for speech vs music"""
        # Use silence detection as a proxy for content type
        cmd = [
            'ffmpeg', '-i', path, '-af',
            'silencedetect=noise=-30dB:d=0.5',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Count silence periods
            silence_starts = result.stderr.count('silence_start')
            _silence_ends = result.stderr.count('silence_end')  # kept for future use

            # More silence periods typically = speech (pauses between sentences)
            # Continuous audio = music

            if silence_starts > 10:
                speech_ratio = 0.8
                music_ratio = 0.2
            elif silence_starts > 3:
                speech_ratio = 0.5
                music_ratio = 0.5
            else:
                speech_ratio = 0.2
                music_ratio = 0.8

            return {
                'speech_ratio': speech_ratio,
                'music_ratio': music_ratio,
                'silence_periods': silence_starts,
            }
        except Exception as e:
            logging.warning(f"Audio analysis failed: {e}")
            return {'speech_ratio': 0.5, 'music_ratio': 0.5, 'silence_periods': 0}

    @staticmethod
    def _detect_faces(path: str, duration: float) -> Dict:
        """Simplified face detection using frame sampling"""
        # Sample frames and check for face-like regions
        # This is a simplified heuristic - could be enhanced with ML

        sample_count = min(5, int(duration / 10) + 1)
        face_frames = 0

        for i in range(sample_count):
            t = duration * (i + 1) / (sample_count + 1)

            # Extract frame and analyze histogram for skin tones
            cmd = [
                'ffmpeg', '-ss', str(t), '-i', path,
                '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-'
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.stdout:
                    # Simple skin tone detection in RGB
                    pixels = list(result.stdout)
                    if len(pixels) >= 3:
                        # Check for skin-tone-like colors (rough heuristic)
                        skin_pixels = 0
                        for j in range(0, len(pixels) - 2, 3):
                            r, g, b = pixels[j], pixels[j+1], pixels[j+2]
                            # Skin tone range (very simplified)
                            if 80 < r < 250 and 50 < g < 200 and 30 < b < 180:
                                if r > g > b:
                                    skin_pixels += 1

                        skin_ratio = skin_pixels / (len(pixels) / 3)
                        if skin_ratio > 0.05:  # At least 5% skin tones
                            face_frames += 1
            except Exception:
                pass

        has_face = face_frames > sample_count * 0.4
        face_ratio = face_frames / sample_count if sample_count > 0 else 0

        return {
            'has_face': has_face,
            'face_ratio': face_ratio,
            'face_frames': face_frames,
        }

    @staticmethod
    def _detect_screen_recording(path: str, width: int, height: int) -> bool:
        """Detect if video is likely a screen recording"""
        # Screen recordings often have:
        # - Standard screen resolutions
        # - Very low motion in most areas
        # - Sharp edges and text

        screen_resolutions = [
            (1920, 1080), (2560, 1440), (3840, 2160),
            (1280, 720), (1366, 768), (1440, 900),
            (1680, 1050), (2560, 1600),
        ]

        # Check if resolution matches common screen sizes
        is_screen_res = (width, height) in screen_resolutions

        # Could add more sophisticated detection here
        return is_screen_res and width >= 1280

    @classmethod
    def _classify_content(
        cls,
        motion: Dict,
        audio: Dict,
        face: Dict,
        is_screen: bool,
        duration: float
    ) -> Tuple[ContentType, float, list]:
        """Classify content type based on analysis"""
        scores = {ct: 0.0 for ct in ContentType}
        reasons = []

        # Screen recording detection
        if is_screen:
            scores[ContentType.TUTORIAL] += 0.5
            reasons.append("Detected screen recording resolution")

        # Face-heavy content
        if face['has_face'] and face['face_ratio'] > 0.6:
            scores[ContentType.TALKING_HEAD] += 0.4
            reasons.append("Face present in most frames")

        # Speech-heavy content
        if audio['speech_ratio'] > 0.6:
            scores[ContentType.TALKING_HEAD] += 0.3
            scores[ContentType.TUTORIAL] += 0.2
            reasons.append("Speech-dominant audio")

        # Music-heavy content
        if audio['music_ratio'] > 0.7:
            scores[ContentType.MUSIC_VIDEO] += 0.4
            scores[ContentType.CINEMATIC] += 0.2
            reasons.append("Music-dominant audio")

        # High motion content
        if motion['level'] == 'high':
            scores[ContentType.ACTION] += 0.4
            scores[ContentType.GAMING] += 0.2
            reasons.append("High motion detected")

        # Low motion, long shots = cinematic
        if motion['level'] == 'low' and motion['avg_shot'] > 8:
            scores[ContentType.CINEMATIC] += 0.3
            scores[ContentType.NATURE] += 0.2
            reasons.append("Long takes, low motion")

        # Medium motion with faces = vlog/talking head
        if motion['level'] == 'medium' and face['has_face']:
            scores[ContentType.TALKING_HEAD] += 0.2
            reasons.append("Medium motion with faces")

        # Short video likely social media
        if duration < 60:
            # Short videos bias toward action/music
            scores[ContentType.ACTION] += 0.1
            scores[ContentType.MUSIC_VIDEO] += 0.1

        # Find best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Calculate confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5

        # If no clear winner, default to unknown
        if best_score < 0.2:
            best_type = ContentType.UNKNOWN
            confidence = 0.3
            reasons.append("No strong content indicators")

        return best_type, confidence, reasons
