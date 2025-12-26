#!/usr/bin/env python3
"""
NubHQ Content Combiner
======================
Tools for combining and creating content from processed videos:
1. Highlight Reels - Extract and compile best moments
2. Multi-Angle Sync - Synchronize multiple camera angles by audio
3. Template Compilation - Fill templates with clips

🦭 The walrus assembles the chaos!
"""

import os
import json
import logging
import subprocess
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Try to import audio analysis libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning("numpy not available - some features limited")


# ============================================================
# CONFIGURATION
# ============================================================

class CombinerConfig:
    """Content combiner configuration"""

    # Output directories
    OUTPUT_DIR = Path(os.environ.get('NUBHQ_OUTPUT', '/Volumes/NUB_Workspace/output'))
    HIGHLIGHTS_DIR = OUTPUT_DIR / 'highlights'
    MULTICAM_DIR = OUTPUT_DIR / 'multicam'
    COMPILED_DIR = OUTPUT_DIR / 'compiled'

    # Highlight extraction
    DEFAULT_HIGHLIGHT_DURATION = 60  # seconds
    MIN_CLIP_DURATION = 2.0  # Minimum clip length
    MAX_CLIP_DURATION = 15.0  # Maximum single clip length
    CROSSFADE_DURATION = 0.5  # Transition duration

    # Audio sync
    AUDIO_SAMPLE_RATE = 16000  # Hz for fingerprinting
    SYNC_SEARCH_WINDOW = 30  # Seconds to search for sync point

    # Templates
    TEMPLATES_DIR = Path(os.environ.get('NUBHQ_TEMPLATES', '/Volumes/NUB_Workspace/templates'))

    @classmethod
    def ensure_dirs(cls):
        """Create output directories"""
        for d in [cls.HIGHLIGHTS_DIR, cls.MULTICAM_DIR, cls.COMPILED_DIR, cls.TEMPLATES_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Clip:
    """A video clip segment"""
    source_path: str
    start_time: float
    end_time: float
    score: float = 1.0
    label: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SyncResult:
    """Result of multi-angle synchronization"""
    reference_video: str
    synced_videos: Dict[str, float]  # path -> offset in seconds
    confidence: float
    method: str  # "audio" or "manual"


@dataclass
class CompilationResult:
    """Result of content compilation"""
    success: bool
    output_path: Optional[str]
    source_clips: List[Clip]
    duration: float
    template_used: Optional[str]
    error: Optional[str] = None


# ============================================================
# HIGHLIGHT EXTRACTOR
# ============================================================

class HighlightExtractor:
    """Extract and compile highlight reels from videos"""

    def __init__(self):
        CombinerConfig.ensure_dirs()

    def extract_highlights(
        self,
        video_path: Path,
        target_duration: int = None,
        output_name: str = None
    ) -> CompilationResult:
        """
        Extract best moments and compile into highlight reel.

        video_path: Source video
        target_duration: Target output duration in seconds (default: 60)
        output_name: Output filename (default: auto-generated)
        """
        target_duration = target_duration or CombinerConfig.DEFAULT_HIGHLIGHT_DURATION

        try:
            # Get video duration
            source_duration = self._get_duration(str(video_path))

            if source_duration <= target_duration:
                # Video shorter than target, just copy it
                output_path = self._copy_video(video_path, output_name)
                return CompilationResult(
                    success=True,
                    output_path=str(output_path),
                    source_clips=[Clip(str(video_path), 0, source_duration)],
                    duration=source_duration,
                    template_used=None
                )

            # Find best moments
            clips = self._find_highlight_clips(str(video_path), source_duration, target_duration)

            if not clips:
                return CompilationResult(
                    success=False,
                    output_path=None,
                    source_clips=[],
                    duration=0,
                    template_used=None,
                    error="No highlight clips found"
                )

            # Compile clips into highlight reel
            output_path = self._compile_clips(video_path, clips, output_name)

            total_duration = sum(c.duration for c in clips)

            return CompilationResult(
                success=True,
                output_path=str(output_path),
                source_clips=clips,
                duration=total_duration,
                template_used=None
            )

        except Exception as e:
            logging.exception(f"Highlight extraction failed: {e}")
            return CompilationResult(
                success=False,
                output_path=None,
                source_clips=[],
                duration=0,
                template_used=None,
                error=str(e)
            )

    def _get_duration(self, path: str) -> float:
        """Get video duration"""
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return float(data.get('format', {}).get('duration', 0))

    def _find_highlight_clips(
        self,
        path: str,
        source_duration: float,
        target_duration: float
    ) -> List[Clip]:
        """Find the best clips based on audio and motion"""
        clips = []

        # Analyze audio levels throughout video
        audio_peaks = self._find_audio_peaks(path, source_duration)

        # Analyze motion/scene changes
        motion_peaks = self._find_motion_peaks(path, source_duration)

        # Combine and score time ranges
        all_peaks = []
        for t, score in audio_peaks:
            all_peaks.append((t, score, 'audio'))
        for t, score in motion_peaks:
            all_peaks.append((t, score * 0.8, 'motion'))  # Weight motion slightly less

        # Sort by score
        all_peaks.sort(key=lambda x: x[1], reverse=True)

        # Select non-overlapping clips
        used_ranges = []
        total_duration = 0

        for peak_time, score, peak_type in all_peaks:
            if total_duration >= target_duration:
                break

            # Determine clip boundaries
            clip_duration = min(
                CombinerConfig.MAX_CLIP_DURATION,
                target_duration - total_duration
            )

            start = max(0, peak_time - clip_duration / 3)  # Peak at 1/3 into clip
            end = min(source_duration, start + clip_duration)

            # Ensure minimum duration
            if end - start < CombinerConfig.MIN_CLIP_DURATION:
                continue

            # Check for overlap with existing clips
            overlaps = False
            for used_start, used_end in used_ranges:
                if not (end < used_start or start > used_end):
                    overlaps = True
                    break

            if not overlaps:
                clips.append(Clip(
                    source_path=path,
                    start_time=start,
                    end_time=end,
                    score=score,
                    label=peak_type
                ))
                used_ranges.append((start, end))
                total_duration += end - start

        # Sort clips by time for natural flow
        clips.sort(key=lambda c: c.start_time)

        return clips

    def _find_audio_peaks(self, path: str, duration: float) -> List[Tuple[float, float]]:
        """Find timestamps of audio peaks"""
        peaks = []
        segment_duration = 3.0  # Analyze in 3-second chunks

        for start in range(0, int(duration), int(segment_duration)):
            cmd = [
                'ffmpeg', '-ss', str(start), '-t', str(segment_duration),
                '-i', path, '-af', 'volumedetect', '-f', 'null', '-'
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                for line in result.stderr.split('\n'):
                    if 'max_volume:' in line:
                        max_vol = float(line.split('max_volume:')[1].split('dB')[0].strip())

                        # Normalize to 0-1 score
                        score = min(1.0, (max_vol + 20) / 15)
                        if score > 0.5:  # Only keep significant peaks
                            peaks.append((start + segment_duration / 2, score))
            except:
                pass

        return peaks

    def _find_motion_peaks(self, path: str, duration: float) -> List[Tuple[float, float]]:
        """Find timestamps of high motion"""
        peaks = []

        cmd = [
            'ffmpeg', '-i', path, '-vf',
            'select=\'gt(scene,0.3)\',showinfo',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            import re
            for line in result.stderr.split('\n'):
                if 'pts_time:' in line:
                    match = re.search(r'pts_time:(\d+\.?\d*)', line)
                    if match:
                        t = float(match.group(1))
                        peaks.append((t, 0.7))  # Fixed score for scene changes
        except:
            pass

        return peaks

    def _compile_clips(self, source: Path, clips: List[Clip], output_name: str = None) -> Path:
        """Compile clips into a single video with crossfades"""
        output_name = output_name or f"{source.stem}_highlights_{int(datetime.now().timestamp())}.mp4"
        output_path = CombinerConfig.HIGHLIGHTS_DIR / output_name

        # Create concat file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_file = f.name

            for clip in clips:
                # Extract each clip to a temp file
                clip_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(clip.start_time),
                    '-t', str(clip.duration),
                    '-i', clip.source_path,
                    '-c:v', 'libx264', '-crf', '20',
                    '-c:a', 'aac', '-b:a', '192k',
                    clip_file
                ]
                subprocess.run(cmd, capture_output=True, timeout=120)

                f.write(f"file '{clip_file}'\n")

        # Concatenate clips
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264', '-crf', '20',
            '-c:a', 'aac', '-b:a', '192k',
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, timeout=300)

        # Cleanup
        os.unlink(concat_file)

        return output_path

    def _copy_video(self, source: Path, output_name: str = None) -> Path:
        """Copy video to highlights directory"""
        output_name = output_name or f"{source.stem}_highlight.mp4"
        output_path = CombinerConfig.HIGHLIGHTS_DIR / output_name

        cmd = [
            'ffmpeg', '-y', '-i', str(source),
            '-c', 'copy', str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

        return output_path


# ============================================================
# MULTI-ANGLE SYNC
# ============================================================

class MultiAngleSync:
    """Synchronize multiple camera angles by audio"""

    def __init__(self):
        CombinerConfig.ensure_dirs()

    def sync_by_audio(self, videos: List[Path]) -> SyncResult:
        """
        Synchronize multiple videos by their audio tracks.
        Uses cross-correlation of audio waveforms.

        videos: List of video paths to sync
        Returns: SyncResult with offsets for each video
        """
        if len(videos) < 2:
            return SyncResult(
                reference_video=str(videos[0]) if videos else "",
                synced_videos={},
                confidence=0,
                method="audio"
            )

        # Use first video as reference
        reference = videos[0]
        offsets = {str(reference): 0.0}
        confidences = []

        # Extract reference audio fingerprint
        ref_audio = self._extract_audio_data(reference)

        for video in videos[1:]:
            video_audio = self._extract_audio_data(video)

            if ref_audio is not None and video_audio is not None:
                offset, confidence = self._find_offset(ref_audio, video_audio)
                offsets[str(video)] = offset
                confidences.append(confidence)
            else:
                # Fallback to 0 offset
                offsets[str(video)] = 0.0
                confidences.append(0.0)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return SyncResult(
            reference_video=str(reference),
            synced_videos=offsets,
            confidence=avg_confidence,
            method="audio"
        )

    def _extract_audio_data(self, video_path: Path) -> Optional[Any]:
        """Extract audio waveform data from video"""
        if not HAS_NUMPY:
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
                # Extract audio as raw PCM
                cmd = [
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-t', str(CombinerConfig.SYNC_SEARCH_WINDOW),
                    '-ar', str(CombinerConfig.AUDIO_SAMPLE_RATE),
                    '-ac', '1', '-f', 's16le',
                    tmp.name
                ]
                subprocess.run(cmd, capture_output=True, timeout=60)

                # Read as numpy array
                with open(tmp.name, 'rb') as f:
                    audio_data = np.frombuffer(f.read(), dtype=np.int16)

                os.unlink(tmp.name)
                return audio_data.astype(np.float32) / 32768.0  # Normalize

        except Exception as e:
            logging.warning(f"Audio extraction failed for {video_path}: {e}")
            return None

    def _find_offset(self, ref_audio: Any, target_audio: Any) -> Tuple[float, float]:
        """Find time offset between two audio tracks using cross-correlation"""
        if not HAS_NUMPY:
            return 0.0, 0.0

        try:
            # Cross-correlation
            correlation = np.correlate(ref_audio, target_audio, mode='full')

            # Find peak
            peak_idx = np.argmax(np.abs(correlation))

            # Convert to time offset
            offset_samples = peak_idx - len(target_audio) + 1
            offset_seconds = offset_samples / CombinerConfig.AUDIO_SAMPLE_RATE

            # Calculate confidence from peak prominence
            peak_value = np.abs(correlation[peak_idx])
            avg_value = np.mean(np.abs(correlation))
            confidence = min(1.0, peak_value / (avg_value * 3)) if avg_value > 0 else 0

            return offset_seconds, confidence

        except Exception as e:
            logging.warning(f"Offset calculation failed: {e}")
            return 0.0, 0.0

    def create_multicam_timeline(
        self,
        sync_result: SyncResult,
        output_name: str = None
    ) -> Optional[Path]:
        """
        Create a synchronized multi-camera timeline.
        Outputs an XML file compatible with video editors.
        """
        if not sync_result.synced_videos:
            return None

        output_name = output_name or f"multicam_{int(datetime.now().timestamp())}.xml"
        output_path = CombinerConfig.MULTICAM_DIR / output_name

        # Generate FCPXML (Final Cut Pro compatible)
        xml_content = self._generate_fcpxml(sync_result)

        with open(output_path, 'w') as f:
            f.write(xml_content)

        return output_path

    def _generate_fcpxml(self, sync_result: SyncResult) -> str:
        """Generate FCPXML for multi-angle editing"""
        videos = list(sync_result.synced_videos.keys())

        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
    <resources>
'''

        # Add asset references
        for i, video in enumerate(videos):
            xml += f'''        <asset id="r{i+1}" src="file://{video}" hasVideo="1" hasAudio="1"/>
'''

        xml += '''    </resources>
    <library>
        <event name="MultiCam Sync">
            <mc-clip name="Synced MultiCam">
                <mc-source angleID="1">
'''

        # Add clips with offsets
        for i, video in enumerate(videos):
            offset = sync_result.synced_videos[video]
            offset_frames = int(offset * 24)  # Assuming 24fps
            xml += f'''                    <clip name="Angle {i+1}" offset="{offset_frames}/24s" ref="r{i+1}"/>
'''

        xml += '''                </mc-source>
            </mc-clip>
        </event>
    </library>
</fcpxml>'''

        return xml


# ============================================================
# TEMPLATE COMPILER
# ============================================================

class TemplateCompiler:
    """Compile videos using predefined templates"""

    # Built-in templates
    TEMPLATES = {
        "instagram_reel": {
            "name": "Instagram Reel",
            "duration": 30,
            "aspect": "9:16",
            "resolution": (1080, 1920),
            "segments": [
                {"type": "hook", "duration": 3, "label": "Hook"},
                {"type": "highlight", "duration": 22, "label": "Content"},
                {"type": "cta", "duration": 5, "label": "Call to Action"},
            ]
        },
        "youtube_short": {
            "name": "YouTube Short",
            "duration": 60,
            "aspect": "9:16",
            "resolution": (1080, 1920),
            "segments": [
                {"type": "intro", "duration": 3, "label": "Intro"},
                {"type": "highlight", "duration": 52, "label": "Main Content"},
                {"type": "outro", "duration": 5, "label": "Outro"},
            ]
        },
        "tiktok": {
            "name": "TikTok",
            "duration": 15,
            "aspect": "9:16",
            "resolution": (1080, 1920),
            "segments": [
                {"type": "hook", "duration": 2, "label": "Hook"},
                {"type": "highlight", "duration": 13, "label": "Content"},
            ]
        },
        "youtube_recap": {
            "name": "YouTube Recap",
            "duration": 180,
            "aspect": "16:9",
            "resolution": (1920, 1080),
            "segments": [
                {"type": "intro", "duration": 10, "label": "Intro"},
                {"type": "highlight", "duration": 150, "label": "Highlights"},
                {"type": "outro", "duration": 20, "label": "Outro"},
            ]
        },
        "story": {
            "name": "Story Format",
            "duration": 15,
            "aspect": "9:16",
            "resolution": (1080, 1920),
            "segments": [
                {"type": "highlight", "duration": 15, "label": "Story"},
            ]
        },
    }

    def __init__(self):
        CombinerConfig.ensure_dirs()
        self.highlight_extractor = HighlightExtractor()

    def list_templates(self) -> List[Dict]:
        """List available templates"""
        return [
            {
                "id": tid,
                "name": t["name"],
                "duration": t["duration"],
                "aspect": t["aspect"],
                "segments": len(t["segments"])
            }
            for tid, t in self.TEMPLATES.items()
        ]

    def compile(
        self,
        template_id: str,
        source_videos: List[Path],
        intro_video: Path = None,
        outro_video: Path = None,
        output_name: str = None
    ) -> CompilationResult:
        """
        Compile videos using a template.

        template_id: Template to use
        source_videos: Source videos to pull highlights from
        intro_video: Custom intro (optional)
        outro_video: Custom outro (optional)
        output_name: Output filename
        """
        if template_id not in self.TEMPLATES:
            return CompilationResult(
                success=False,
                output_path=None,
                source_clips=[],
                duration=0,
                template_used=template_id,
                error=f"Unknown template: {template_id}"
            )

        template = self.TEMPLATES[template_id]

        try:
            clips = []

            for segment in template["segments"]:
                seg_type = segment["type"]
                seg_duration = segment["duration"]

                if seg_type == "intro":
                    if intro_video and intro_video.exists():
                        clips.append(Clip(
                            source_path=str(intro_video),
                            start_time=0,
                            end_time=seg_duration,
                            label="intro"
                        ))
                    # else: skip intro

                elif seg_type == "outro":
                    if outro_video and outro_video.exists():
                        clips.append(Clip(
                            source_path=str(outro_video),
                            start_time=0,
                            end_time=seg_duration,
                            label="outro"
                        ))
                    # else: skip outro

                elif seg_type in ["highlight", "hook", "cta"]:
                    # Extract highlights from source videos
                    highlight_clips = self._extract_segment_clips(
                        source_videos, seg_duration, seg_type
                    )
                    clips.extend(highlight_clips)

            if not clips:
                return CompilationResult(
                    success=False,
                    output_path=None,
                    source_clips=[],
                    duration=0,
                    template_used=template_id,
                    error="No clips extracted for template"
                )

            # Compile into final video
            output_path = self._compile_template(
                clips, template, output_name
            )

            total_duration = sum(c.duration for c in clips)

            return CompilationResult(
                success=True,
                output_path=str(output_path),
                source_clips=clips,
                duration=total_duration,
                template_used=template_id
            )

        except Exception as e:
            logging.exception(f"Template compilation failed: {e}")
            return CompilationResult(
                success=False,
                output_path=None,
                source_clips=[],
                duration=0,
                template_used=template_id,
                error=str(e)
            )

    def _extract_segment_clips(
        self,
        source_videos: List[Path],
        target_duration: float,
        segment_type: str
    ) -> List[Clip]:
        """Extract clips for a template segment"""
        all_clips = []
        duration_per_source = target_duration / len(source_videos) if source_videos else 0

        for video in source_videos:
            # Use highlight extractor to find best moments
            result = self.highlight_extractor.extract_highlights(
                video,
                target_duration=int(duration_per_source)
            )

            if result.success and result.source_clips:
                all_clips.extend(result.source_clips)

        # Trim to target duration
        total = 0
        final_clips = []

        for clip in sorted(all_clips, key=lambda c: c.score, reverse=True):
            if total >= target_duration:
                break

            remaining = target_duration - total
            if clip.duration > remaining:
                # Trim clip
                clip = Clip(
                    source_path=clip.source_path,
                    start_time=clip.start_time,
                    end_time=clip.start_time + remaining,
                    score=clip.score,
                    label=segment_type
                )

            final_clips.append(clip)
            total += clip.duration

        return final_clips

    def _compile_template(
        self,
        clips: List[Clip],
        template: Dict,
        output_name: str = None
    ) -> Path:
        """Compile clips into template format"""
        output_name = output_name or f"{template['name'].lower().replace(' ', '_')}_{int(datetime.now().timestamp())}.mp4"
        output_path = CombinerConfig.COMPILED_DIR / output_name

        width, height = template["resolution"]

        # Create concat file with scaling
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_file = f.name

            for clip in clips:
                # Extract and scale each clip
                clip_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name

                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(clip.start_time),
                    '-t', str(clip.duration),
                    '-i', clip.source_path,
                    '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
                    '-c:v', 'libx264', '-crf', '20',
                    '-c:a', 'aac', '-b:a', '192k',
                    clip_file
                ]
                subprocess.run(cmd, capture_output=True, timeout=120)

                f.write(f"file '{clip_file}'\n")

        # Concatenate
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264', '-crf', '20',
            '-c:a', 'aac', '-b:a', '192k',
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, timeout=600)

        os.unlink(concat_file)

        return output_path


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_highlight_reel(
    video_path: str,
    duration: int = 60,
    output_name: str = None
) -> CompilationResult:
    """Quick function to create a highlight reel"""
    extractor = HighlightExtractor()
    return extractor.extract_highlights(Path(video_path), duration, output_name)


def sync_videos(videos: List[str]) -> SyncResult:
    """Quick function to sync multiple videos"""
    syncer = MultiAngleSync()
    return syncer.sync_by_audio([Path(v) for v in videos])


def compile_for_platform(
    platform: str,
    source_videos: List[str],
    output_name: str = None
) -> CompilationResult:
    """Quick function to compile for a specific platform"""
    compiler = TemplateCompiler()

    platform_map = {
        "instagram": "instagram_reel",
        "youtube_short": "youtube_short",
        "tiktok": "tiktok",
        "youtube": "youtube_recap",
        "story": "story",
    }

    template_id = platform_map.get(platform.lower(), "instagram_reel")
    return compiler.compile(template_id, [Path(v) for v in source_videos], output_name=output_name)


# ============================================================
# MAIN (for testing)
# ============================================================

def main():
    """Test content combiner"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python content_combiner.py <command> <video_path> [options]")
        print()
        print("Commands:")
        print("  highlight <video> [duration]  - Create highlight reel")
        print("  sync <video1> <video2> ...    - Sync multiple videos")
        print("  compile <template> <video>    - Compile using template")
        print("  templates                     - List available templates")
        sys.exit(1)

    command = sys.argv[1]

    if command == "templates":
        compiler = TemplateCompiler()
        print("\n📋 Available Templates:")
        for t in compiler.list_templates():
            print(f"  • {t['id']}: {t['name']} ({t['duration']}s, {t['aspect']})")
        sys.exit(0)

    if command == "highlight":
        video_path = Path(sys.argv[2])
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60

        print(f"\n🎬 Creating {duration}s highlight reel from: {video_path.name}")
        result = create_highlight_reel(str(video_path), duration)

        if result.success:
            print(f"✅ Created: {result.output_path}")
            print(f"   Duration: {result.duration:.1f}s")
            print(f"   Clips: {len(result.source_clips)}")
        else:
            print(f"❌ Failed: {result.error}")

    elif command == "sync":
        videos = [Path(v) for v in sys.argv[2:]]

        print(f"\n🎥 Syncing {len(videos)} videos...")
        result = sync_videos([str(v) for v in videos])

        print(f"Reference: {Path(result.reference_video).name}")
        print(f"Confidence: {result.confidence:.2f}")
        print("\nOffsets:")
        for video, offset in result.synced_videos.items():
            print(f"  {Path(video).name}: {offset:+.3f}s")

    elif command == "compile":
        template = sys.argv[2]
        videos = [Path(v) for v in sys.argv[3:]]

        print(f"\n📦 Compiling with template '{template}'...")
        result = compile_for_platform(template, [str(v) for v in videos])

        if result.success:
            print(f"✅ Created: {result.output_path}")
            print(f"   Duration: {result.duration:.1f}s")
        else:
            print(f"❌ Failed: {result.error}")


if __name__ == '__main__':
    main()
