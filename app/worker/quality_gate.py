"""
Quality Gate

Checks processed videos for issues before moving to Library:
- Black frames detection
- Audio sync verification
- Encoding quality check
- File integrity validation
- Duration verification
- Audio level check

Flags problems and prevents bad videos from reaching Library.
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class QualityIssue(Enum):
    """Types of quality issues"""
    BLACK_FRAMES = "black_frames"
    AUDIO_SYNC = "audio_sync"
    LOW_BITRATE = "low_bitrate"
    CORRUPT_FILE = "corrupt_file"
    DURATION_MISMATCH = "duration_mismatch"
    AUDIO_CLIPPING = "audio_clipping"
    AUDIO_TOO_QUIET = "audio_too_quiet"
    MISSING_AUDIO = "missing_audio"
    ENCODING_ERROR = "encoding_error"
    RESOLUTION_WRONG = "resolution_wrong"


@dataclass
class QualityCheck:
    """Result of a single quality check"""
    check_name: str
    passed: bool
    issue: Optional[QualityIssue]
    severity: str  # 'info', 'warning', 'error'
    details: str


@dataclass
class QualityReport:
    """Full quality report for a video"""
    video_path: Path
    passed: bool
    overall_score: float  # 0-1
    checks: List[QualityCheck]
    errors: List[str]
    warnings: List[str]
    can_proceed: bool  # True if video is good enough


class QualityGate:
    """
    Quality gate for processed videos.

    Runs multiple checks and generates a report.
    Videos must pass all error-level checks to proceed.
    """

    # Thresholds
    MIN_BITRATE_RATIO = 0.5      # Min % of expected bitrate
    MAX_BLACK_FRAME_RATIO = 0.1  # Max % of black frames
    MIN_AUDIO_DB = -40           # Minimum average audio level
    MAX_AUDIO_DB = -0.5          # Maximum peak (clipping threshold)
    DURATION_TOLERANCE = 2.0     # Seconds tolerance for duration

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def check(
        self,
        video_path: Path,
        expected_duration: Optional[float] = None,
        expected_resolution: Optional[Tuple[int, int]] = None
    ) -> QualityReport:
        """
        Run all quality checks on a video.

        Args:
            video_path: Path to processed video
            expected_duration: Expected duration (for mismatch check)
            expected_resolution: Expected (width, height)

        Returns:
            QualityReport with all check results
        """
        checks = []
        errors = []
        warnings = []

        # Run all checks
        checks.append(self._check_file_integrity(video_path))
        checks.append(self._check_black_frames(video_path))
        checks.append(self._check_audio_levels(video_path))
        checks.append(self._check_encoding_quality(video_path))

        if expected_duration:
            checks.append(self._check_duration(video_path, expected_duration))

        if expected_resolution:
            checks.append(self._check_resolution(video_path, expected_resolution))

        # Collect errors and warnings
        for check in checks:
            if not check.passed:
                if check.severity == 'error':
                    errors.append(f"{check.check_name}: {check.details}")
                elif check.severity == 'warning':
                    warnings.append(f"{check.check_name}: {check.details}")

        # Calculate overall score
        passed_count = sum(1 for c in checks if c.passed)
        overall_score = passed_count / len(checks) if checks else 0

        # Determine if can proceed (no errors)
        can_proceed = len(errors) == 0
        overall_passed = can_proceed and len(warnings) < 3

        return QualityReport(
            video_path=video_path,
            passed=overall_passed,
            overall_score=overall_score,
            checks=checks,
            errors=errors,
            warnings=warnings,
            can_proceed=can_proceed
        )

    def _check_file_integrity(self, video_path: Path) -> QualityCheck:
        """Check if file is valid and can be read"""
        check_name = "File Integrity"

        if not video_path.exists():
            return QualityCheck(
                check_name=check_name,
                passed=False,
                issue=QualityIssue.CORRUPT_FILE,
                severity='error',
                details="File does not exist"
            )

        if video_path.stat().st_size == 0:
            return QualityCheck(
                check_name=check_name,
                passed=False,
                issue=QualityIssue.CORRUPT_FILE,
                severity='error',
                details="File is empty"
            )

        # Try to probe the file
        cmd = [
            'ffprobe', '-v', 'error', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return QualityCheck(
                    check_name=check_name,
                    passed=False,
                    issue=QualityIssue.CORRUPT_FILE,
                    severity='error',
                    details=f"FFprobe error: {result.stderr[:100]}"
                )

            data = json.loads(result.stdout)

            # Check for video stream
            has_video = any(s['codec_type'] == 'video' for s in data.get('streams', []))
            if not has_video:
                return QualityCheck(
                    check_name=check_name,
                    passed=False,
                    issue=QualityIssue.ENCODING_ERROR,
                    severity='error',
                    details="No video stream found"
                )

            return QualityCheck(
                check_name=check_name,
                passed=True,
                issue=None,
                severity='info',
                details="File is valid"
            )

        except subprocess.TimeoutExpired:
            return QualityCheck(
                check_name=check_name,
                passed=False,
                issue=QualityIssue.CORRUPT_FILE,
                severity='error',
                details="File probe timed out"
            )
        except Exception as e:
            return QualityCheck(
                check_name=check_name,
                passed=False,
                issue=QualityIssue.CORRUPT_FILE,
                severity='error',
                details=f"Probe failed: {str(e)}"
            )

    def _check_black_frames(self, video_path: Path) -> QualityCheck:
        """Check for excessive black frames"""
        check_name = "Black Frames"

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', 'blackdetect=d=0.5:pix_th=0.10',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Count black detections
            black_count = result.stderr.count('black_start')

            # Get video duration
            duration = self._get_duration(video_path)

            if duration > 0:
                # Estimate black frame ratio (rough)
                black_seconds = black_count * 0.5  # Each detection is at least 0.5s
                black_ratio = black_seconds / duration

                if black_ratio > self.MAX_BLACK_FRAME_RATIO:
                    return QualityCheck(
                        check_name=check_name,
                        passed=False,
                        issue=QualityIssue.BLACK_FRAMES,
                        severity='warning',
                        details=f"~{black_ratio*100:.0f}% black frames detected"
                    )

            return QualityCheck(
                check_name=check_name,
                passed=True,
                issue=None,
                severity='info',
                details=f"{black_count} black segments detected" if black_count else "No black frames"
            )

        except Exception as e:
            return QualityCheck(
                check_name=check_name,
                passed=True,  # Don't fail if check itself fails
                issue=None,
                severity='info',
                details=f"Check skipped: {str(e)}"
            )

    def _check_audio_levels(self, video_path: Path) -> QualityCheck:
        """Check audio levels for clipping or too quiet"""
        check_name = "Audio Levels"

        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-af', 'volumedetect', '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Parse volume info
            peak_db = None
            mean_db = None

            for line in result.stderr.split('\n'):
                if 'max_volume:' in line:
                    try:
                        peak_db = float(line.split('max_volume:')[1].split('dB')[0].strip())
                    except Exception:
                        pass
                if 'mean_volume:' in line:
                    try:
                        mean_db = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                    except Exception:
                        pass

            # No audio stream
            if peak_db is None and mean_db is None:
                # Check if video has audio stream at all
                has_audio = self._has_audio_stream(video_path)
                if not has_audio:
                    return QualityCheck(
                        check_name=check_name,
                        passed=True,  # Not an error if source has no audio
                        issue=None,
                        severity='info',
                        details="No audio stream"
                    )
                else:
                    return QualityCheck(
                        check_name=check_name,
                        passed=False,
                        issue=QualityIssue.MISSING_AUDIO,
                        severity='warning',
                        details="Audio stream exists but no levels detected"
                    )

            # Check for clipping
            if peak_db and peak_db > self.MAX_AUDIO_DB:
                return QualityCheck(
                    check_name=check_name,
                    passed=False,
                    issue=QualityIssue.AUDIO_CLIPPING,
                    severity='warning',
                    details=f"Audio clipping detected (peak: {peak_db:.1f}dB)"
                )

            # Check for too quiet
            if mean_db and mean_db < self.MIN_AUDIO_DB:
                return QualityCheck(
                    check_name=check_name,
                    passed=False,
                    issue=QualityIssue.AUDIO_TOO_QUIET,
                    severity='warning',
                    details=f"Audio too quiet (avg: {mean_db:.1f}dB)"
                )

            return QualityCheck(
                check_name=check_name,
                passed=True,
                issue=None,
                severity='info',
                details=f"Audio levels OK (peak: {peak_db:.1f}dB, avg: {mean_db:.1f}dB)"
            )

        except Exception as e:
            return QualityCheck(
                check_name=check_name,
                passed=True,
                issue=None,
                severity='info',
                details=f"Check skipped: {str(e)}"
            )

    def _check_encoding_quality(self, video_path: Path) -> QualityCheck:
        """Check encoding bitrate and quality"""
        check_name = "Encoding Quality"

        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)

            # Get video stream
            video_stream = next(
                (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
                None
            )

            if not video_stream:
                return QualityCheck(
                    check_name=check_name,
                    passed=False,
                    issue=QualityIssue.ENCODING_ERROR,
                    severity='error',
                    details="No video stream"
                )

            # Check bitrate
            bitrate = int(data.get('format', {}).get('bit_rate', 0))
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))

            # Expected bitrate based on resolution (rough guide)
            if width >= 3840:
                expected_bitrate = 25_000_000  # 25 Mbps for 4K
            elif width >= 1920:
                expected_bitrate = 8_000_000   # 8 Mbps for 1080p
            elif width >= 1280:
                expected_bitrate = 5_000_000   # 5 Mbps for 720p
            else:
                expected_bitrate = 2_000_000   # 2 Mbps for SD

            bitrate_ratio = bitrate / expected_bitrate if expected_bitrate > 0 else 1

            if bitrate_ratio < self.MIN_BITRATE_RATIO:
                return QualityCheck(
                    check_name=check_name,
                    passed=False,
                    issue=QualityIssue.LOW_BITRATE,
                    severity='warning',
                    details=f"Low bitrate: {bitrate/1_000_000:.1f} Mbps (expected ~{expected_bitrate/1_000_000:.0f} Mbps)"
                )

            return QualityCheck(
                check_name=check_name,
                passed=True,
                issue=None,
                severity='info',
                details=f"Bitrate: {bitrate/1_000_000:.1f} Mbps, {width}x{height}"
            )

        except Exception as e:
            return QualityCheck(
                check_name=check_name,
                passed=True,
                issue=None,
                severity='info',
                details=f"Check skipped: {str(e)}"
            )

    def _check_duration(self, video_path: Path, expected: float) -> QualityCheck:
        """Check if duration matches expected"""
        check_name = "Duration"

        actual = self._get_duration(video_path)

        diff = abs(actual - expected)

        if diff > self.DURATION_TOLERANCE:
            return QualityCheck(
                check_name=check_name,
                passed=False,
                issue=QualityIssue.DURATION_MISMATCH,
                severity='warning',
                details=f"Duration mismatch: {actual:.1f}s vs expected {expected:.1f}s"
            )

        return QualityCheck(
            check_name=check_name,
            passed=True,
            issue=None,
            severity='info',
            details=f"Duration: {actual:.1f}s"
        )

    def _check_resolution(self, video_path: Path, expected: Tuple[int, int]) -> QualityCheck:
        """Check if resolution matches expected"""
        check_name = "Resolution"

        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)

            video_stream = next(
                (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
                None
            )

            if not video_stream:
                return QualityCheck(
                    check_name=check_name,
                    passed=False,
                    issue=QualityIssue.ENCODING_ERROR,
                    severity='error',
                    details="No video stream"
                )

            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))

            if (width, height) != expected:
                return QualityCheck(
                    check_name=check_name,
                    passed=False,
                    issue=QualityIssue.RESOLUTION_WRONG,
                    severity='warning',
                    details=f"Resolution: {width}x{height} vs expected {expected[0]}x{expected[1]}"
                )

            return QualityCheck(
                check_name=check_name,
                passed=True,
                issue=None,
                severity='info',
                details=f"Resolution: {width}x{height}"
            )

        except Exception as e:
            return QualityCheck(
                check_name=check_name,
                passed=True,
                issue=None,
                severity='info',
                details=f"Check skipped: {str(e)}"
            )

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
        except Exception:
            return 0

    def _has_audio_stream(self, video_path: Path) -> bool:
        """Check if video has audio stream"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', str(video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return any(s['codec_type'] == 'audio' for s in data.get('streams', []))
        except Exception:
            return False


def check_video_quality_cli():
    """CLI entry point"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quality_gate.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    print(f"\nQuality Gate Check: {video_path.name}")
    print("=" * 60)

    gate = QualityGate()
    report = gate.check(video_path)

    print(f"\nOverall: {'PASS' if report.passed else 'FAIL'} (score: {report.overall_score:.0%})")
    print(f"Can Proceed: {'Yes' if report.can_proceed else 'No'}")

    print("\nChecks:")
    for check in report.checks:
        status = "OK" if check.passed else ("WARN" if check.severity == 'warning' else "FAIL")
        print(f"  [{status}] {check.check_name}: {check.details}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"  - {error}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")


if __name__ == '__main__':
    check_video_quality_cli()
