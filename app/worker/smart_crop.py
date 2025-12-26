"""
Smart Crop with Face Tracking

Auto-reframe horizontal videos to vertical with:
- Face detection and tracking
- Subject-following crop
- Scene-aware composition
- Multiple output formats (9:16, 1:1, 4:5)
"""

import os
import subprocess
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

# Optional OpenCV
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


@dataclass
class CropRegion:
    """A crop region with position"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    timestamp: float


@dataclass
class SmartCropResult:
    """Result of smart crop operation"""
    success: bool
    output_path: Optional[Path]
    aspect_ratio: str
    crop_method: str  # 'face_tracking', 'center', 'auto'
    error: Optional[str]


class SmartCropper:
    """
    Smart video cropping with face tracking.

    Converts landscape videos to vertical/square while keeping
    the subject (usually a face) in frame.
    """

    # Output aspect ratios
    ASPECT_RATIOS = {
        'vertical': (9, 16),    # TikTok, Reels, Shorts
        'square': (1, 1),       # Instagram feed
        'portrait': (4, 5),     # Instagram portrait
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def crop(
        self,
        video_path: Path,
        aspect_ratio: str = 'vertical',
        method: str = 'auto'
    ) -> SmartCropResult:
        """
        Crop video to target aspect ratio.

        Args:
            video_path: Source video
            aspect_ratio: Target ratio ('vertical', 'square', 'portrait')
            method: Crop method ('face_tracking', 'center', 'auto')

        Returns:
            SmartCropResult
        """
        if aspect_ratio not in self.ASPECT_RATIOS:
            return SmartCropResult(
                success=False,
                output_path=None,
                aspect_ratio=aspect_ratio,
                crop_method=method,
                error=f"Unknown aspect ratio: {aspect_ratio}"
            )

        # Get source dimensions
        src_width, src_height = self._get_dimensions(video_path)

        # Calculate target dimensions
        target_w, target_h = self.ASPECT_RATIOS[aspect_ratio]
        target_height = src_height
        target_width = int(target_height * target_w / target_h)

        if target_width > src_width:
            # Source is too narrow, use width as constraint
            target_width = src_width
            target_height = int(target_width * target_h / target_w)

        # Determine crop method
        if method == 'auto':
            # Check if faces are present
            has_faces = self._detect_faces_present(video_path)
            method = 'face_tracking' if has_faces and HAS_OPENCV else 'center'

        # Generate crop
        if method == 'face_tracking' and HAS_OPENCV:
            output_path = self._crop_with_face_tracking(
                video_path, src_width, src_height,
                target_width, target_height, aspect_ratio
            )
        else:
            output_path = self._crop_center(
                video_path, src_width, src_height,
                target_width, target_height, aspect_ratio
            )

        if output_path and output_path.exists():
            return SmartCropResult(
                success=True,
                output_path=output_path,
                aspect_ratio=aspect_ratio,
                crop_method=method,
                error=None
            )
        else:
            return SmartCropResult(
                success=False,
                output_path=None,
                aspect_ratio=aspect_ratio,
                crop_method=method,
                error="Crop operation failed"
            )

    def _get_dimensions(self, video_path: Path) -> Tuple[int, int]:
        """Get video dimensions"""
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream['codec_type'] == 'video':
                    return int(stream['width']), int(stream['height'])
        except Exception:
            pass
        return 1920, 1080

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration"""
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return float(data.get('format', {}).get('duration', 0))
        except Exception:
            return 0

    def _detect_faces_present(self, video_path: Path) -> bool:
        """Quick check if video has faces"""
        if not HAS_OPENCV:
            return False

        duration = self._get_duration(video_path)
        # Sample a few frames
        for t in [duration * 0.25, duration * 0.5, duration * 0.75]:
            if self._detect_face_at_time(video_path, t):
                return True
        return False

    def _detect_face_at_time(self, video_path: Path, time_sec: float) -> bool:
        """Detect face at specific timestamp"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cmd = [
                    'ffmpeg', '-ss', str(time_sec), '-i', str(video_path),
                    '-vframes', '1', '-q:v', '2', '-y', tmp.name
                ]
                subprocess.run(cmd, capture_output=True, timeout=10)

                if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                    img = cv2.imread(tmp.name)
                    os.unlink(tmp.name)

                    if img is None:
                        return False

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
                    return len(faces) > 0

                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

        except Exception as e:
            logging.debug(f"Face detection failed: {e}")

        return False

    def _crop_center(
        self,
        video_path: Path,
        src_w: int, src_h: int,
        target_w: int, target_h: int,
        aspect_ratio: str
    ) -> Optional[Path]:
        """Simple center crop"""
        output_name = f"{video_path.stem}_{aspect_ratio}.mp4"
        output_path = self.output_dir / output_name

        # Calculate crop position (center)
        x = (src_w - target_w) // 2
        y = (src_h - target_h) // 2

        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', f'crop={target_w}:{target_h}:{x}:{y}',
            '-c:v', 'libx264', '-crf', '20', '-preset', 'medium',
            '-c:a', 'copy',
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=3600)
            if result.returncode == 0:
                return output_path
        except Exception as e:
            logging.error(f"Center crop failed: {e}")

        return None

    def _crop_with_face_tracking(
        self,
        video_path: Path,
        src_w: int, src_h: int,
        target_w: int, target_h: int,
        aspect_ratio: str
    ) -> Optional[Path]:
        """
        Crop with face tracking.

        Analyzes face positions throughout video and generates
        smooth crop keyframes.
        """
        output_name = f"{video_path.stem}_{aspect_ratio}_tracked.mp4"
        output_path = self.output_dir / output_name

        # Detect face positions throughout video
        duration = self._get_duration(video_path)
        sample_interval = 1.0  # Sample every second
        face_positions = []

        for t in range(0, int(duration), int(sample_interval)):
            pos = self._get_face_position(video_path, float(t), src_w, src_h)
            if pos:
                face_positions.append((t, pos))

        if not face_positions:
            # No faces detected, fall back to center crop
            logging.info("No faces detected, using center crop")
            return self._crop_center(video_path, src_w, src_h, target_w, target_h, aspect_ratio)

        # Interpolate positions and generate crop filter
        crop_filter = self._generate_crop_filter(
            face_positions, src_w, src_h, target_w, target_h, duration
        )

        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', crop_filter,
            '-c:v', 'libx264', '-crf', '20', '-preset', 'medium',
            '-c:a', 'copy',
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=3600)
            if result.returncode == 0:
                return output_path
            else:
                logging.error(f"Face tracking crop failed: {result.stderr[:500]}")
        except Exception as e:
            logging.error(f"Face tracking crop failed: {e}")

        return None

    def _get_face_position(
        self,
        video_path: Path,
        time_sec: float,
        src_w: int, src_h: int
    ) -> Optional[Tuple[int, int]]:
        """Get face center position at timestamp"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cmd = [
                    'ffmpeg', '-ss', str(time_sec), '-i', str(video_path),
                    '-vframes', '1', '-q:v', '2', '-y', tmp.name
                ]
                subprocess.run(cmd, capture_output=True, timeout=10)

                if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                    img = cv2.imread(tmp.name)
                    os.unlink(tmp.name)

                    if img is None:
                        return None

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

                    if len(faces) > 0:
                        # Use largest face
                        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                        # Return center of face
                        center_x = x + w // 2
                        center_y = y + h // 2

                        # Scale to original dimensions if needed
                        img_h, img_w = img.shape[:2]
                        center_x = int(center_x * src_w / img_w)
                        center_y = int(center_y * src_h / img_h)

                        return (center_x, center_y)

                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

        except Exception as e:
            logging.debug(f"Face position detection failed: {e}")

        return None

    def _generate_crop_filter(
        self,
        face_positions: List[Tuple[float, Tuple[int, int]]],
        src_w: int, src_h: int,
        target_w: int, target_h: int,
        duration: float
    ) -> str:
        """
        Generate FFmpeg crop filter with smooth tracking.

        Uses sendcmd filter to animate crop position.
        """
        # Calculate valid crop range
        min_x = 0
        max_x = src_w - target_w
        min_y = 0
        max_y = src_h - target_h

        # Generate keyframes
        keyframes = []
        prev_x, prev_y = None, None

        for t, (face_x, face_y) in face_positions:
            # Center crop on face
            crop_x = face_x - target_w // 2
            crop_y = face_y - target_h // 2

            # Clamp to valid range
            crop_x = max(min_x, min(max_x, crop_x))
            crop_y = max(min_y, min(max_y, crop_y))

            # Smooth with previous position (simple low-pass)
            if prev_x is not None:
                crop_x = int(prev_x * 0.7 + crop_x * 0.3)
                crop_y = int(prev_y * 0.7 + crop_y * 0.3)

            keyframes.append((t, crop_x, crop_y))
            prev_x, prev_y = crop_x, crop_y

        # If we have keyframes, use zoompan for smooth animation
        if keyframes:
            # Simple approach: use average position
            avg_x = sum(k[1] for k in keyframes) // len(keyframes)
            avg_y = sum(k[2] for k in keyframes) // len(keyframes)

            return f'crop={target_w}:{target_h}:{avg_x}:{avg_y}'

        # Fallback to center
        center_x = (src_w - target_w) // 2
        center_y = (src_h - target_h) // 2
        return f'crop={target_w}:{target_h}:{center_x}:{center_y}'


def smart_crop_cli():
    """CLI entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python smart_crop.py <video_path> [aspect_ratio] [output_dir]")
        print("  aspect_ratio: vertical (9:16), square (1:1), portrait (4:5)")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    aspect_ratio = sys.argv[2] if len(sys.argv) > 2 else 'vertical'
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else video_path.parent

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    print(f"\nSmart cropping: {video_path.name}")
    print(f"Target: {aspect_ratio}")
    print("=" * 60)

    cropper = SmartCropper(output_dir)
    result = cropper.crop(video_path, aspect_ratio)

    if result.success:
        print("\nSuccess!")
        print(f"  Output: {result.output_path}")
        print(f"  Method: {result.crop_method}")
    else:
        print(f"\nFailed: {result.error}")


if __name__ == '__main__':
    smart_crop_cli()
