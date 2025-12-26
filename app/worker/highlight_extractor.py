"""
Auto Highlight Extractor

Finds and extracts the best clips from long videos:
- Energy peaks (audio + motion)
- Face/performance moments
- Scene transitions
- Auto-generates multiple clip lengths (15s, 30s, 60s)

Uses engagement_scorer for moment detection, then extracts actual clips.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .engagement_scorer import EngagementScorer, Moment


@dataclass
class ExtractedHighlight:
    """An extracted highlight clip"""
    path: Path
    start_time: float
    end_time: float
    duration: float
    score: float
    reason: str
    format: str  # 'landscape', 'portrait', 'square'


class HighlightExtractor:
    """Extract highlight clips from long videos"""

    # Target clip durations
    CLIP_DURATIONS = [15, 30, 60]  # seconds

    # Minimum source video duration for highlight extraction
    MIN_SOURCE_DURATION = 120  # 2 minutes

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.scorer = EngagementScorer()

    def extract_highlights(
        self,
        video_path: Path,
        max_clips: int = 5,
        target_durations: List[int] = None,
        formats: List[str] = None
    ) -> List[ExtractedHighlight]:
        """
        Extract highlight clips from a video.

        Args:
            video_path: Source video
            max_clips: Maximum clips to extract per duration
            target_durations: Clip lengths in seconds (default: 15, 30, 60)
            formats: Output formats (default: ['landscape'])

        Returns:
            List of extracted highlight clips
        """
        durations = target_durations or self.CLIP_DURATIONS
        formats = formats or ['landscape']

        # Get video duration
        source_duration = self._get_duration(video_path)

        if source_duration < self.MIN_SOURCE_DURATION:
            logging.info(f"Video too short for highlight extraction ({source_duration:.0f}s)")
            return []

        # Score the video to find best moments
        logging.info(f"Analyzing {video_path.name} for highlights...")
        score = self.scorer.score(video_path)

        if not score.best_moments:
            logging.info("No highlight moments found")
            return []

        logging.info(f"Found {len(score.best_moments)} potential highlights")

        extracted = []

        for target_duration in durations:
            # Filter moments that can fit the target duration
            suitable_moments = [
                m for m in score.best_moments
                if m.end_time - m.start_time >= target_duration * 0.8  # Allow 80% minimum
                or self._can_extend_moment(m, source_duration, target_duration)
            ]

            # Take top N moments
            for i, moment in enumerate(suitable_moments[:max_clips]):
                for fmt in formats:
                    try:
                        clip = self._extract_clip(
                            video_path,
                            moment,
                            target_duration,
                            fmt,
                            clip_index=i
                        )
                        if clip:
                            extracted.append(clip)
                    except Exception as e:
                        logging.error(f"Failed to extract clip: {e}")

        return extracted

    def extract_best_clip(
        self,
        video_path: Path,
        duration: int = 30,
        format: str = 'landscape'
    ) -> Optional[ExtractedHighlight]:
        """Extract just the single best highlight clip"""
        clips = self.extract_highlights(
            video_path,
            max_clips=1,
            target_durations=[duration],
            formats=[format]
        )
        return clips[0] if clips else None

    def auto_generate_shorts(
        self,
        video_path: Path,
        count: int = 3
    ) -> List[ExtractedHighlight]:
        """
        Auto-generate vertical shorts from a landscape video.
        Perfect for TikTok/Reels/Shorts repurposing.
        """
        return self.extract_highlights(
            video_path,
            max_clips=count,
            target_durations=[15, 30, 60],
            formats=['portrait']
        )

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', str(video_path)
        ]
        try:
            import json
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return float(data.get('format', {}).get('duration', 0))
        except Exception as e:
            logging.error(f"Failed to get duration: {e}")
            return 0

    def _can_extend_moment(
        self,
        moment: Moment,
        source_duration: float,
        target_duration: int
    ) -> bool:
        """Check if a moment can be extended to target duration"""
        moment_duration = moment.end_time - moment.start_time
        needed = target_duration - moment_duration

        # Can we extend before?
        extend_before = moment.start_time
        # Can we extend after?
        extend_after = source_duration - moment.end_time

        return extend_before + extend_after >= needed

    def _extract_clip(
        self,
        video_path: Path,
        moment: Moment,
        target_duration: int,
        format: str,
        clip_index: int
    ) -> Optional[ExtractedHighlight]:
        """Extract a single clip from the video"""

        # Calculate actual clip boundaries
        moment_duration = moment.end_time - moment.start_time

        if moment_duration >= target_duration:
            # Moment is long enough, use the best part (center)
            center = (moment.start_time + moment.end_time) / 2
            start_time = max(0, center - target_duration / 2)
            end_time = start_time + target_duration
        else:
            # Need to extend the moment
            needed = target_duration - moment_duration
            extend_before = min(needed / 2, moment.start_time)
            extend_after = needed - extend_before

            start_time = moment.start_time - extend_before
            end_time = moment.end_time + extend_after

        # Build output filename
        format_suffix = {'landscape': 'h', 'portrait': 'v', 'square': 's'}
        suffix = format_suffix.get(format, 'h')

        output_name = f"{video_path.stem}_highlight_{clip_index+1}_{target_duration}s_{suffix}.mp4"
        output_path = self.output_dir / output_name

        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-i', str(video_path),
            '-t', str(target_duration),
        ]

        # Add format-specific filters
        if format == 'portrait':
            # Crop to 9:16 vertical
            cmd.extend(['-vf', 'crop=ih*9/16:ih,scale=1080:1920'])
        elif format == 'square':
            # Crop to 1:1
            cmd.extend(['-vf', 'crop=min(iw\\,ih):min(iw\\,ih),scale=1080:1080'])
        else:
            # Keep landscape, ensure 1080p
            cmd.extend(['-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2'])

        # Encoding settings
        cmd.extend([
            '-c:v', 'libx264', '-crf', '20', '-preset', 'medium',
            '-c:a', 'aac', '-b:a', '192k',
            str(output_path)
        ])

        logging.info(f"Extracting {target_duration}s {format} clip at {start_time:.1f}s...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and output_path.exists():
                return ExtractedHighlight(
                    path=output_path,
                    start_time=start_time,
                    end_time=end_time,
                    duration=target_duration,
                    score=moment.score,
                    reason=moment.reason,
                    format=format
                )
            else:
                logging.error(f"FFmpeg failed: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            logging.error("Clip extraction timed out")
        except Exception as e:
            logging.error(f"Extraction failed: {e}")

        return None


def extract_highlights_cli():
    """CLI entry point for highlight extraction"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python highlight_extractor.py <video_path> [output_dir]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else video_path.parent / "highlights"

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting highlights from: {video_path.name}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    extractor = HighlightExtractor(output_dir)
    highlights = extractor.extract_highlights(video_path)

    if highlights:
        print(f"\nExtracted {len(highlights)} highlight clips:")
        for h in highlights:
            print(f"  - {h.path.name}")
            print(f"    {h.start_time:.1f}s - {h.end_time:.1f}s ({h.duration}s, {h.format})")
            print(f"    Score: {h.score:.2f} - {h.reason}")
    else:
        print("\nNo highlights extracted.")


if __name__ == '__main__':
    extract_highlights_cli()
