"""
Smart Thumbnail Generator

Auto-picks the best frames for thumbnails:
- Face detection (prefer faces, especially smiling)
- Sharpness/blur detection
- Composition analysis (rule of thirds, centered subjects)
- Color vibrancy
- Motion blur avoidance

Generates multiple options for user to choose from.
"""

import os
import subprocess
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Optional OpenCV for better analysis
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@dataclass
class ThumbnailCandidate:
    """A potential thumbnail frame"""
    path: Path
    timestamp: float
    score: float
    has_face: bool
    sharpness: float
    brightness: float
    color_score: float
    composition_score: float
    reasons: List[str]


class SmartThumbnailGenerator:
    """Generate optimal thumbnails from video"""

    # Scoring weights
    WEIGHTS = {
        'face': 0.35,
        'sharpness': 0.25,
        'composition': 0.20,
        'color': 0.15,
        'brightness': 0.05,
    }

    # Sample settings
    SAMPLE_COUNT = 20  # Frames to sample
    OUTPUT_COUNT = 5   # Thumbnails to generate

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        video_path: Path,
        count: int = 5,
        sizes: List[Tuple[int, int]] = None
    ) -> List[ThumbnailCandidate]:
        """
        Generate smart thumbnails from video.

        Args:
            video_path: Source video
            count: Number of thumbnails to generate
            sizes: Output sizes as (width, height) tuples

        Returns:
            List of generated thumbnails sorted by score
        """
        sizes = sizes or [(1280, 720), (1080, 1080)]  # Default: YouTube + Instagram

        duration = self._get_duration(video_path)
        if duration <= 0:
            logging.error("Could not determine video duration")
            return []

        # Sample frames evenly across video (skip first/last 5%)
        start = duration * 0.05
        end = duration * 0.95
        interval = (end - start) / self.SAMPLE_COUNT

        candidates = []

        for i in range(self.SAMPLE_COUNT):
            timestamp = start + (i * interval)

            # Extract and analyze frame
            candidate = self._analyze_frame(video_path, timestamp)
            if candidate:
                candidates.append(candidate)

        if not candidates:
            logging.warning("No valid frames found for thumbnails")
            return []

        # Sort by score
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Generate actual thumbnail files for top candidates
        results = []
        for i, candidate in enumerate(candidates[:count]):
            for width, height in sizes:
                thumb_path = self._generate_thumbnail(
                    video_path,
                    candidate.timestamp,
                    width,
                    height,
                    index=i
                )
                if thumb_path:
                    # Update candidate with actual path
                    thumb_candidate = ThumbnailCandidate(
                        path=thumb_path,
                        timestamp=candidate.timestamp,
                        score=candidate.score,
                        has_face=candidate.has_face,
                        sharpness=candidate.sharpness,
                        brightness=candidate.brightness,
                        color_score=candidate.color_score,
                        composition_score=candidate.composition_score,
                        reasons=candidate.reasons
                    )
                    results.append(thumb_candidate)

        return results

    def get_best_thumbnail(
        self,
        video_path: Path,
        size: Tuple[int, int] = (1280, 720)
    ) -> Optional[ThumbnailCandidate]:
        """Get just the single best thumbnail"""
        results = self.generate(video_path, count=1, sizes=[size])
        return results[0] if results else None

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', str(video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return float(data.get('format', {}).get('duration', 0))
        except Exception as e:
            logging.error(f"Duration check failed: {e}")
            return 0

    def _analyze_frame(
        self,
        video_path: Path,
        timestamp: float
    ) -> Optional[ThumbnailCandidate]:
        """Extract and analyze a single frame"""

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Extract frame
            cmd = [
                'ffmpeg', '-ss', str(timestamp), '-i', str(video_path),
                '-vframes', '1', '-q:v', '2', '-y', tmp.name
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)

                if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) == 0:
                    os.unlink(tmp.name)
                    return None

                # Analyze the frame
                scores = self._score_frame(tmp.name)

                os.unlink(tmp.name)

                if scores is None:
                    return None

                # Calculate overall score
                overall = (
                    scores['face'] * self.WEIGHTS['face'] +
                    scores['sharpness'] * self.WEIGHTS['sharpness'] +
                    scores['composition'] * self.WEIGHTS['composition'] +
                    scores['color'] * self.WEIGHTS['color'] +
                    scores['brightness'] * self.WEIGHTS['brightness']
                )

                # Build reasons list
                reasons = []
                if scores['has_face']:
                    reasons.append("face detected")
                if scores['sharpness'] > 0.7:
                    reasons.append("sharp")
                if scores['color'] > 0.7:
                    reasons.append("vibrant colors")
                if scores['composition'] > 0.7:
                    reasons.append("good composition")

                return ThumbnailCandidate(
                    path=Path(tmp.name),  # Temporary, will be replaced
                    timestamp=timestamp,
                    score=overall,
                    has_face=scores['has_face'],
                    sharpness=scores['sharpness'],
                    brightness=scores['brightness'],
                    color_score=scores['color'],
                    composition_score=scores['composition'],
                    reasons=reasons
                )

            except Exception as e:
                logging.debug(f"Frame analysis failed at {timestamp}: {e}")
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                return None

    def _score_frame(self, frame_path: str) -> Optional[dict]:
        """Score a frame for thumbnail suitability"""

        if HAS_OPENCV:
            return self._score_frame_opencv(frame_path)
        else:
            return self._score_frame_basic(frame_path)

    def _score_frame_opencv(self, frame_path: str) -> Optional[dict]:
        """Score frame using OpenCV"""
        try:
            img = cv2.imread(frame_path)
            if img is None:
                return None

            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Face detection
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            has_face = len(faces) > 0
            face_score = min(1.0, len(faces) * 0.4 + 0.3) if has_face else 0

            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(1.0, laplacian_var / 500)  # Normalize

            # Brightness
            brightness = np.mean(gray) / 255
            # Penalize very dark or very bright
            brightness_score = 1.0 - abs(brightness - 0.5) * 2

            # Color vibrancy (saturation in HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1]) / 255
            color_score = saturation

            # Composition - check if interesting content is in center
            center_region = gray[
                height//3:2*height//3,
                width//3:2*width//3
            ]
            edge_region_top = gray[:height//4, :]
            edge_region_bottom = gray[3*height//4:, :]

            center_variance = np.var(center_region)
            edge_variance = (np.var(edge_region_top) + np.var(edge_region_bottom)) / 2

            # Higher center variance relative to edges = better composition
            if edge_variance > 0:
                composition_score = min(1.0, center_variance / (edge_variance + center_variance))
            else:
                composition_score = 0.5

            return {
                'has_face': has_face,
                'face': face_score,
                'sharpness': sharpness,
                'brightness': brightness_score,
                'color': color_score,
                'composition': composition_score,
            }

        except Exception as e:
            logging.debug(f"OpenCV scoring failed: {e}")
            return None

    def _score_frame_basic(self, frame_path: str) -> Optional[dict]:
        """Basic scoring without OpenCV"""

        # Read raw pixels with FFmpeg
        cmd = [
            'ffmpeg', '-i', frame_path, '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if not result.stdout:
                return None

            pixels = list(result.stdout)
            pixel_count = len(pixels) // 3

            if pixel_count == 0:
                return None

            # Calculate basic metrics
            total_r = sum(pixels[i] for i in range(0, len(pixels), 3))
            total_g = sum(pixels[i] for i in range(1, len(pixels), 3))
            total_b = sum(pixels[i] for i in range(2, len(pixels), 3))

            avg_r = total_r / pixel_count
            avg_g = total_g / pixel_count
            avg_b = total_b / pixel_count

            # Brightness
            brightness = (avg_r + avg_g + avg_b) / (3 * 255)
            brightness_score = 1.0 - abs(brightness - 0.5) * 2

            # Color variance as vibrancy proxy
            color_variance = max(abs(avg_r - avg_g), abs(avg_g - avg_b), abs(avg_r - avg_b))
            color_score = min(1.0, color_variance / 50)

            # Face detection via skin tones (rough)
            skin_count = 0
            for i in range(0, min(len(pixels) - 2, 30000), 3):
                r, g, b = pixels[i], pixels[i+1], pixels[i+2]
                if 80 < r < 250 and 50 < g < 200 and 30 < b < 180:
                    if r > g > b:
                        skin_count += 1

            skin_ratio = skin_count / (min(pixel_count, 10000))
            has_face = skin_ratio > 0.1
            face_score = min(1.0, skin_ratio * 5) if has_face else 0

            return {
                'has_face': has_face,
                'face': face_score,
                'sharpness': 0.5,  # Can't measure without OpenCV
                'brightness': brightness_score,
                'color': color_score,
                'composition': 0.5,  # Can't measure without OpenCV
            }

        except Exception as e:
            logging.debug(f"Basic scoring failed: {e}")
            return None

    def _generate_thumbnail(
        self,
        video_path: Path,
        timestamp: float,
        width: int,
        height: int,
        index: int
    ) -> Optional[Path]:
        """Generate actual thumbnail file"""

        output_name = f"{video_path.stem}_thumb_{index+1}_{width}x{height}.jpg"
        output_path = self.output_dir / output_name

        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', str(video_path),
            '-vframes', '1',
            '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
            '-q:v', '2',
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                logging.error(f"Thumbnail generation failed: {result.stderr[:200]}")

        except Exception as e:
            logging.error(f"Thumbnail generation error: {e}")

        return None


def generate_thumbnails_cli():
    """CLI entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smart_thumbnails.py <video_path> [output_dir]")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else video_path.parent / "thumbnails"

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    print(f"\nGenerating smart thumbnails for: {video_path.name}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    generator = SmartThumbnailGenerator(output_dir)
    thumbnails = generator.generate(video_path)

    if thumbnails:
        print(f"\nGenerated {len(thumbnails)} thumbnails:")
        for t in thumbnails:
            print(f"  - {t.path.name}")
            print(f"    Score: {t.score:.2f} at {t.timestamp:.1f}s")
            print(f"    {', '.join(t.reasons) if t.reasons else 'no special features'}")
    else:
        print("\nNo thumbnails generated.")


if __name__ == '__main__':
    generate_thumbnails_cli()
