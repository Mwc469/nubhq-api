#!/usr/bin/env python3
"""
NubHQ Smart Video Processor
============================
Enhanced processor that integrates all smart features:
- Smart folder routing
- Content-aware profile selection
- Quality gate validation
- Auto thumbnail generation
- Highlight extraction
- Enhanced learning
- GPU acceleration
- Watermarks
- Auto-captions
- Notifications
"""

import os
import json
import time
import hashlib
import logging
import subprocess
import threading
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass, field
from queue import PriorityQueue, Empty
import signal

# Import all smart modules
from .smart_folders import SmartFolderRouter, FolderRouting
from .quality_gate import QualityGate, QualityReport
from .smart_thumbnails import SmartThumbnailGenerator
from .highlight_extractor import HighlightExtractor
from .preference_learning import EnhancedPreferenceLearner
from .quick_profiles import profile_to_decisions
from .engagement_scorer import EngagementScorer


# ============================================================
# CONFIGURATION
# ============================================================

class SmartConfig:
    """Enhanced configuration"""
    # Drives
    WORK_DRIVE = Path(os.environ.get('NUBHQ_WORK', '/Volumes/Lil Hoss/NubHQ'))
    STORAGE_DRIVE = Path(os.environ.get('NUBHQ_STORAGE', '/Volumes/Big Hoss/NubHQ'))

    # Folders
    INPUT_DIR = STORAGE_DRIVE / '01_Inbox_ToSort'
    LIBRARY_DIR = STORAGE_DRIVE / '02_Library'
    OUTPUT_DIR = STORAGE_DRIVE / '03_Exports'
    ARCHIVE_DIR = STORAGE_DRIVE / '04_Archive'
    METADATA_DIR = STORAGE_DRIVE / '05_Metadata'
    DATABASE_DIR = STORAGE_DRIVE / 'database'

    TEMP_DIR = WORK_DRIVE / 'temp'
    CACHE_DIR = WORK_DRIVE / 'cache'
    LOG_DIR = WORK_DRIVE / 'logs'

    # Output subdirectories
    THUMBNAILS_DIR = OUTPUT_DIR / 'thumbnails'
    CAPTIONS_DIR = OUTPUT_DIR / 'captions'
    HIGHLIGHTS_DIR = OUTPUT_DIR / 'highlights'
    AUTO_QUEUED_DIR = OUTPUT_DIR / 'auto-queued'
    REVIEW_DIR = OUTPUT_DIR / 'review'

    # Feature flags
    ENABLE_THUMBNAILS = os.environ.get('NUBHQ_THUMBNAILS', 'true').lower() == 'true'
    ENABLE_HIGHLIGHTS = os.environ.get('NUBHQ_HIGHLIGHTS', 'true').lower() == 'true'
    ENABLE_CAPTIONS = os.environ.get('NUBHQ_CAPTIONS', 'false').lower() == 'true'
    ENABLE_WATERMARK = os.environ.get('NUBHQ_WATERMARK', 'false').lower() == 'true'
    ENABLE_GPU = os.environ.get('NUBHQ_GPU', 'true').lower() == 'true'
    ENABLE_NOTIFICATIONS = os.environ.get('NUBHQ_NOTIFY', 'false').lower() == 'true'
    ENABLE_QUALITY_GATE = os.environ.get('NUBHQ_QUALITY_GATE', 'true').lower() == 'true'

    # Watermark
    WATERMARK_PATH = Path(os.environ.get('NUBHQ_WATERMARK_PATH', ''))
    WATERMARK_POSITION = os.environ.get('NUBHQ_WATERMARK_POS', 'bottom_right')
    WATERMARK_OPACITY = float(os.environ.get('NUBHQ_WATERMARK_OPACITY', '0.7'))

    # Notifications
    WEBHOOK_URL = os.environ.get('NUBHQ_WEBHOOK_URL', '')
    EMAIL_TO = os.environ.get('NUBHQ_EMAIL_TO', '')

    # Processing
    SUPPORTED_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mts', '.m2ts'}
    POLL_INTERVAL = 2.0
    HIGHLIGHT_MIN_DURATION = 120  # Only extract highlights from videos > 2 min

    # Database
    DB_PATH = DATABASE_DIR / 'learning.db'

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories"""
        dirs = [
            cls.INPUT_DIR, cls.LIBRARY_DIR, cls.OUTPUT_DIR, cls.ARCHIVE_DIR,
            cls.METADATA_DIR, cls.DATABASE_DIR, cls.TEMP_DIR, cls.CACHE_DIR,
            cls.LOG_DIR, cls.THUMBNAILS_DIR, cls.CAPTIONS_DIR, cls.HIGHLIGHTS_DIR,
            cls.AUTO_QUEUED_DIR, cls.REVIEW_DIR
        ]
        for d in dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logging.warning(f"Cannot create {d}")


# ============================================================
# PROCESSED FILES DATABASE
# ============================================================

class ProcessedFilesDB:
    """SQLite-backed storage for processed files (survives restarts)"""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or SmartConfig.DATABASE_DIR / "processed_files.db"
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processed_files (
                    path TEXT PRIMARY KEY,
                    processed_at TEXT NOT NULL,
                    status TEXT DEFAULT 'completed',
                    output_path TEXT,
                    error TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_processed_at ON processed_files(processed_at)')
            conn.commit()

    def is_processed(self, path: Path) -> bool:
        """Check if a file has been processed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT 1 FROM processed_files WHERE path = ?', (str(path),))
            return cursor.fetchone() is not None

    def mark_processed(self, path: Path, status: str = "completed", output_path: Path = None, error: str = None):
        """Mark a file as processed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO processed_files (path, processed_at, status, output_path, error)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(path),
                datetime.now().isoformat(),
                status,
                str(output_path) if output_path else None,
                error
            ))
            conn.commit()

    def get_all_processed(self) -> Set[str]:
        """Get all processed file paths (for backwards compatibility)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT path FROM processed_files')
            return {row[0] for row in cursor.fetchall()}

    def clear_old(self, days: int = 30) -> int:
        """Remove entries older than N days"""
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM processed_files WHERE processed_at < ?', (cutoff,))
            conn.commit()
            return cursor.rowcount


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class ProcessingJob:
    """A video processing job with priority"""
    video_path: Path
    routing: FolderRouting
    priority: int = 0
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        # Higher priority first, then older jobs first
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


@dataclass
class ProcessingResult:
    """Result of processing a video"""
    success: bool
    input_path: Path
    output_path: Optional[Path]
    profile_used: str
    thumbnails: List[Path]
    highlights: List[Path]
    captions_path: Optional[Path]
    quality_report: Optional[QualityReport]
    processing_time: float
    error: Optional[str] = None


# ============================================================
# GPU ENCODER
# ============================================================

class GPUEncoder:
    """Hardware-accelerated encoding using VideoToolbox (Mac)"""

    @staticmethod
    def is_available() -> bool:
        """Check if VideoToolbox is available"""
        cmd = ['ffmpeg', '-encoders']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return 'h264_videotoolbox' in result.stdout
        except Exception:
            return False

    @staticmethod
    def get_encoder() -> str:
        """Get best available encoder"""
        if SmartConfig.ENABLE_GPU and GPUEncoder.is_available():
            return 'h264_videotoolbox'
        return 'libx264'

    @staticmethod
    def get_encoder_options(encoder: str, quality: str) -> List[str]:
        """Get encoder-specific options"""
        if encoder == 'h264_videotoolbox':
            # VideoToolbox quality settings
            quality_map = {
                'max': ['-q:v', '40'],
                'high': ['-q:v', '55'],
                'medium': ['-q:v', '65'],
                'low': ['-q:v', '75'],
                'web': ['-q:v', '70'],
            }
            return quality_map.get(quality, ['-q:v', '55'])
        else:
            # libx264 CRF settings
            crf_map = {
                'max': '16',
                'high': '20',
                'medium': '23',
                'low': '28',
                'web': '26',
            }
            return ['-crf', crf_map.get(quality, '20'), '-preset', 'medium']


# ============================================================
# WATERMARK SYSTEM
# ============================================================

class WatermarkSystem:
    """Add watermarks/branding to videos"""

    @staticmethod
    def get_filter(position: str, opacity: float, watermark_path: Path) -> Optional[str]:
        """Get FFmpeg filter for watermark overlay"""
        if not watermark_path.exists():
            return None

        # Position mapping
        positions = {
            'top_left': '10:10',
            'top_right': 'W-w-10:10',
            'bottom_left': '10:H-h-10',
            'bottom_right': 'W-w-10:H-h-10',
            'center': '(W-w)/2:(H-h)/2',
        }

        pos = positions.get(position, positions['bottom_right'])

        # Scale watermark to 10% of video width, apply opacity
        return f"movie={watermark_path}[wm];[wm]scale=iw*0.1:-1,format=rgba,colorchannelmixer=aa={opacity}[wm2];[0:v][wm2]overlay={pos}"


# ============================================================
# CAPTION GENERATOR
# ============================================================

class CaptionGenerator:
    """Generate captions using Whisper"""

    @staticmethod
    def is_available() -> bool:
        """Check if Whisper is available"""
        try:
            import whisper
            return True
        except ImportError:
            return False

    @staticmethod
    def generate(video_path: Path, output_dir: Path, style: str = 'srt') -> Optional[Path]:
        """Generate captions for a video"""
        if not CaptionGenerator.is_available():
            logging.warning("Whisper not installed. Run: pip install openai-whisper")
            return None

        try:
            import whisper

            # Load model (use base for speed, medium/large for accuracy)
            model = whisper.load_model("base")

            # Transcribe
            result = model.transcribe(str(video_path))

            # Generate output file
            output_path = output_dir / f"{video_path.stem}.{style}"

            if style == 'srt':
                CaptionGenerator._write_srt(result, output_path)
            elif style == 'vtt':
                CaptionGenerator._write_vtt(result, output_path)
            else:
                # Plain text
                with open(output_path, 'w') as f:
                    f.write(result['text'])

            return output_path

        except Exception as e:
            logging.error(f"Caption generation failed: {e}")
            return None

    @staticmethod
    def _write_srt(result: dict, output_path: Path):
        """Write SRT subtitle file"""
        with open(output_path, 'w') as f:
            for i, segment in enumerate(result['segments'], 1):
                start = CaptionGenerator._format_timestamp(segment['start'])
                end = CaptionGenerator._format_timestamp(segment['end'])
                text = segment['text'].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

    @staticmethod
    def _write_vtt(result: dict, output_path: Path):
        """Write VTT subtitle file"""
        with open(output_path, 'w') as f:
            f.write("WEBVTT\n\n")
            for segment in result['segments']:
                start = CaptionGenerator._format_timestamp(segment['start'], vtt=True)
                end = CaptionGenerator._format_timestamp(segment['end'], vtt=True)
                text = segment['text'].strip()
                f.write(f"{start} --> {end}\n{text}\n\n")

    @staticmethod
    def _format_timestamp(seconds: float, vtt: bool = False) -> str:
        """Format seconds as SRT/VTT timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def burn_captions(video_path: Path, srt_path: Path, output_path: Path) -> bool:
        """Burn captions into video"""
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2'",
            '-c:a', 'copy',
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=3600)
            return result.returncode == 0
        except Exception:
            return False


# ============================================================
# NOTIFICATION SYSTEM
# ============================================================

class NotificationSystem:
    """Send notifications on processing events"""

    @staticmethod
    def send_webhook(url: str, data: dict) -> bool:
        """Send webhook notification"""
        if not url:
            return False

        try:
            import requests
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logging.warning(f"Webhook failed: {e}")
            return False

    @staticmethod
    def send_email(to: str, subject: str, body: str) -> bool:
        """Send email notification (requires configured mail)"""
        if not to:
            return False

        try:
            cmd = ['mail', '-s', subject, to]
            result = subprocess.run(cmd, input=body.encode(), capture_output=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            logging.warning(f"Email failed: {e}")
            return False

    @staticmethod
    def notify_complete(result: ProcessingResult):
        """Send notification for completed processing"""
        if not SmartConfig.ENABLE_NOTIFICATIONS:
            return

        data = {
            'event': 'processing_complete',
            'success': result.success,
            'video': result.input_path.name,
            'profile': result.profile_used,
            'time': result.processing_time,
            'output': str(result.output_path) if result.output_path else None,
            'thumbnails': len(result.thumbnails),
            'highlights': len(result.highlights),
            'error': result.error,
        }

        if SmartConfig.WEBHOOK_URL:
            NotificationSystem.send_webhook(SmartConfig.WEBHOOK_URL, data)

        if SmartConfig.EMAIL_TO:
            status = "completed" if result.success else "failed"
            subject = f"NubHQ: {result.input_path.name} {status}"
            body = json.dumps(data, indent=2)
            NotificationSystem.send_email(SmartConfig.EMAIL_TO, subject, body)


# ============================================================
# DEDUPLICATION
# ============================================================

class Deduplicator:
    """Detect and skip duplicate videos"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize hash database"""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS video_hashes (
                    hash TEXT PRIMARY KEY,
                    filename TEXT,
                    path TEXT,
                    size INTEGER,
                    duration REAL,
                    processed_at TEXT
                )
            ''')
            conn.commit()

    def get_hash(self, video_path: Path) -> str:
        """Generate perceptual hash for video"""
        # Use first frame + middle frame + file size for quick hash
        size = video_path.stat().st_size
        duration = self._get_duration(video_path)

        # Sample frames
        frames_data = []
        for t in [1.0, duration / 2]:
            cmd = [
                'ffmpeg', '-ss', str(t), '-i', str(video_path),
                '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'gray',
                '-vf', 'scale=16:16', '-'
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.stdout:
                    frames_data.append(result.stdout[:256])
            except Exception:
                pass

        # Combine into hash
        hash_input = f"{size}:{duration:.1f}:" + "".join(str(f) for f in frames_data)
        return hashlib.md5(hash_input.encode()).hexdigest()

    def is_duplicate(self, video_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if video is a duplicate"""
        video_hash = self.get_hash(video_path)

        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT filename, path FROM video_hashes WHERE hash = ?',
                (video_hash,)
            )
            row = cursor.fetchone()

            if row:
                return True, row[0]

        return False, None

    def record(self, video_path: Path, video_hash: str = None):
        """Record a processed video"""
        if video_hash is None:
            video_hash = self.get_hash(video_path)

        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO video_hashes (hash, filename, path, size, duration, processed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                video_hash,
                video_path.name,
                str(video_path),
                video_path.stat().st_size,
                self._get_duration(video_path),
                datetime.now().isoformat()
            ))
            conn.commit()

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration"""
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return float(data.get('format', {}).get('duration', 0))
        except Exception:
            return 0


# ============================================================
# SMART VIDEO PROCESSOR
# ============================================================

class SmartVideoProcessor:
    """Enhanced processor with all smart features"""

    def __init__(self):
        SmartConfig.ensure_dirs()

        # Initialize components
        self.learner = EnhancedPreferenceLearner(SmartConfig.DB_PATH)
        self.quality_gate = QualityGate()
        self.thumbnail_gen = SmartThumbnailGenerator(SmartConfig.THUMBNAILS_DIR)
        self.highlight_ext = HighlightExtractor(SmartConfig.HIGHLIGHTS_DIR)
        self.engagement_scorer = EngagementScorer()
        self.deduplicator = Deduplicator(SmartConfig.DATABASE_DIR / 'hashes.db')

        # Check GPU availability
        self.encoder = GPUEncoder.get_encoder()
        if self.encoder == 'h264_videotoolbox':
            logging.info("GPU acceleration enabled (VideoToolbox)")
        else:
            logging.info("Using CPU encoding (libx264)")

    def process(self, job: ProcessingJob) -> ProcessingResult:
        """Process a single video with all smart features"""
        start_time = time.time()
        video_path = job.video_path
        routing = job.routing

        logging.info(f"Processing: {video_path.name}")
        logging.info(f"  Profile: {routing.profile.name} ({routing.reason})")
        logging.info(f"  Priority: {routing.priority}")

        try:
            # Check for duplicates
            is_dup, original = self.deduplicator.is_duplicate(video_path)
            if is_dup:
                logging.info(f"  Skipping duplicate of: {original}")
                return ProcessingResult(
                    success=False,
                    input_path=video_path,
                    output_path=None,
                    profile_used=routing.profile.name,
                    thumbnails=[],
                    highlights=[],
                    captions_path=None,
                    quality_report=None,
                    processing_time=time.time() - start_time,
                    error=f"Duplicate of {original}"
                )

            # Get profile settings with learned overrides
            decisions = profile_to_decisions(routing.profile)
            overrides = self.learner.get_setting_overrides(routing.profile.name)
            decisions.update(overrides)

            # Process video
            output_path = self._encode_video(video_path, decisions, routing.profile.name)

            if not output_path or not output_path.exists():
                raise Exception("Encoding failed")

            # Quality gate check
            quality_report = None
            if SmartConfig.ENABLE_QUALITY_GATE:
                quality_report = self.quality_gate.check(output_path)
                if not quality_report.can_proceed:
                    logging.warning(f"  Quality gate failed: {quality_report.errors}")

            # Generate thumbnails
            thumbnails = []
            if SmartConfig.ENABLE_THUMBNAILS:
                thumbs = self.thumbnail_gen.generate(output_path, count=3)
                thumbnails = [t.path for t in thumbs]
                logging.info(f"  Generated {len(thumbnails)} thumbnails")

            # Extract highlights for long videos
            highlights = []
            duration = self._get_duration(video_path)
            if SmartConfig.ENABLE_HIGHLIGHTS and duration > SmartConfig.HIGHLIGHT_MIN_DURATION:
                clips = self.highlight_ext.extract_highlights(output_path, max_clips=3)
                highlights = [c.path for c in clips]
                logging.info(f"  Extracted {len(highlights)} highlights")

            # Generate captions
            captions_path = None
            if SmartConfig.ENABLE_CAPTIONS:
                captions_path = CaptionGenerator.generate(
                    output_path,
                    SmartConfig.CAPTIONS_DIR
                )
                if captions_path:
                    logging.info(f"  Generated captions: {captions_path.name}")

            # Record for deduplication
            self.deduplicator.record(video_path)

            # Move to appropriate destination
            if quality_report and quality_report.passed:
                final_dest = SmartConfig.AUTO_QUEUED_DIR / output_path.name
            else:
                final_dest = SmartConfig.REVIEW_DIR / output_path.name

            output_path.rename(final_dest)

            # Archive original
            archive_dest = SmartConfig.ARCHIVE_DIR / video_path.name
            if archive_dest.exists():
                archive_dest = SmartConfig.ARCHIVE_DIR / f"{video_path.stem}_{int(time.time())}{video_path.suffix}"
            video_path.rename(archive_dest)

            result = ProcessingResult(
                success=True,
                input_path=video_path,
                output_path=final_dest,
                profile_used=routing.profile.name,
                thumbnails=thumbnails,
                highlights=highlights,
                captions_path=captions_path,
                quality_report=quality_report,
                processing_time=time.time() - start_time
            )

            # Send notification
            NotificationSystem.notify_complete(result)

            logging.info(f"  Completed in {result.processing_time:.1f}s")
            return result

        except Exception as e:
            logging.exception(f"Processing failed: {e}")
            return ProcessingResult(
                success=False,
                input_path=video_path,
                output_path=None,
                profile_used=routing.profile.name,
                thumbnails=[],
                highlights=[],
                captions_path=None,
                quality_report=None,
                processing_time=time.time() - start_time,
                error=str(e)
            )

    def _encode_video(self, video_path: Path, decisions: dict, profile_name: str) -> Optional[Path]:
        """Encode video with FFmpeg"""
        output_name = f"{video_path.stem}_{profile_name}.mp4"
        output_path = SmartConfig.OUTPUT_DIR / output_name

        # Build filters
        video_filters = self._build_video_filters(video_path, decisions)
        audio_filters = self._build_audio_filters(decisions)

        # Build command
        cmd = ['ffmpeg', '-y', '-i', str(video_path)]

        # Add video filters
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])

        # Add audio filters
        if audio_filters:
            cmd.extend(['-af', ','.join(audio_filters)])

        # Add encoder settings
        cmd.extend(['-c:v', self.encoder])
        encoder_opts = GPUEncoder.get_encoder_options(self.encoder, decisions.get('OUTPUT_QUALITY', 'high'))
        cmd.extend(encoder_opts)

        # Audio encoding
        cmd.extend(['-c:a', 'aac', '-b:a', '192k'])

        cmd.append(str(output_path))

        logging.info(f"  Encoding with {self.encoder}...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                logging.error(f"FFmpeg error: {result.stderr[:500]}")
                return None

        except subprocess.TimeoutExpired:
            logging.error("Encoding timed out")
            return None

    def _build_video_filters(self, video_path: Path, decisions: dict) -> List[str]:
        """Build video filter chain"""
        filters = []

        # Get source dimensions
        width, height = self._get_dimensions(video_path)

        # Resolution
        res = decisions.get('OUTPUT_RESOLUTION', 'source')
        res_map = {
            'source': (width, height),
            '4k': (3840, 2160),
            '1080p': (1920, 1080),
            '720p': (1280, 720),
            'square_1080': (1080, 1080),
            'vertical_1080': (1080, 1920),
        }
        target_w, target_h = res_map.get(res, (width, height))

        if (target_w, target_h) != (width, height):
            filters.append(f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease")
            filters.append(f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2")

        # Color grading
        color = decisions.get('COLOR_GRADE', 'none')
        color_filters = {
            'auto_correct': "eq=brightness=0.06:contrast=1.1:saturation=1.1",
            'warm': "colorbalance=rs=0.1:gs=0.05:bs=-0.1,eq=saturation=1.2",
            'cool': "colorbalance=rs=-0.1:gs=0:bs=0.1,eq=saturation=0.9",
            'vintage': "curves=vintage,eq=contrast=1.1:saturation=0.8",
            'high_contrast': "eq=contrast=1.3:saturation=1.15:brightness=0.02",
        }
        if color in color_filters:
            filters.append(color_filters[color])

        # Watermark
        if SmartConfig.ENABLE_WATERMARK and SmartConfig.WATERMARK_PATH.exists():
            wm_filter = WatermarkSystem.get_filter(
                SmartConfig.WATERMARK_POSITION,
                SmartConfig.WATERMARK_OPACITY,
                SmartConfig.WATERMARK_PATH
            )
            if wm_filter:
                filters.append(wm_filter)

        return filters

    def _build_audio_filters(self, decisions: dict) -> List[str]:
        """Build audio filter chain"""
        filters = []

        # Normalization
        audio_norm = decisions.get('AUDIO_NORMALIZE', 'none')
        lufs_map = {'light': -16, 'standard': -14, 'loud': -11, 'youtube': -13}
        if audio_norm in lufs_map:
            filters.append(f"loudnorm=I={lufs_map[audio_norm]}:TP=-1.5:LRA=11")

        # Noise reduction (afftdn nf range: -80 to -20)
        noise = decisions.get('NOISE_REDUCTION', 'none')
        noise_map = {'light': 25, 'medium': 35, 'aggressive': 50}
        if noise in noise_map:
            filters.append(f"afftdn=nf=-{noise_map[noise]}")

        return filters

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


# ============================================================
# FOLDER WATCHER
# ============================================================

class SmartFolderWatcher:
    """Watch folders with priority queue"""

    def __init__(self, processor: SmartVideoProcessor):
        self.processor = processor
        self.queue = PriorityQueue()
        self.processed_db = ProcessedFilesDB()  # SQLite-backed persistence
        self.processed = self.processed_db.get_all_processed()  # Load existing
        self.running = False
        self._lock = threading.Lock()

    def start(self):
        """Start watching"""
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("\n" + "=" * 60)
        print("NubHQ Smart Video Processor")
        print("=" * 60)
        print(f"\nInput:   {SmartConfig.INPUT_DIR}")
        print(f"Output:  {SmartConfig.OUTPUT_DIR}")
        print(f"Archive: {SmartConfig.ARCHIVE_DIR}")
        print(f"\nEncoder: {self.processor.encoder}")
        print(f"Features: thumbnails={SmartConfig.ENABLE_THUMBNAILS}, "
              f"highlights={SmartConfig.ENABLE_HIGHLIGHTS}, "
              f"quality_gate={SmartConfig.ENABLE_QUALITY_GATE}")
        print("\nDrop videos into input folder to process!")
        print("Use subfolders (youtube/, tiktok/, etc.) for auto-profiles")
        print("\nPress Ctrl+C to stop\n")

        # Start watcher thread
        watcher = threading.Thread(target=self._watch_loop, daemon=True)
        watcher.start()

        # Process queue in main thread
        self._process_loop()

    def _signal_handler(self, signum, frame):
        print("\n\nShutting down...")
        self.running = False

    def _watch_loop(self):
        """Scan for new videos"""
        while self.running:
            try:
                self._scan_directory(SmartConfig.INPUT_DIR)
            except Exception as e:
                logging.error(f"Watch error: {e}")

            time.sleep(SmartConfig.POLL_INTERVAL)

    def _scan_directory(self, directory: Path):
        """Recursively scan for videos"""
        for item in directory.iterdir():
            if item.is_dir():
                # Skip processing folder
                if item.name != 'processing':
                    self._scan_directory(item)
            elif item.is_file():
                if item.suffix.lower() in SmartConfig.SUPPORTED_FORMATS:
                    with self._lock:
                        # Check both in-memory cache and database
                        if str(item) not in self.processed and not self.processed_db.is_processed(item):
                            self.processed.add(str(item))

                            # Wait for file to finish writing
                            initial_size = item.stat().st_size
                            time.sleep(1)
                            if item.exists() and item.stat().st_size == initial_size:
                                # Route the video
                                routing = SmartFolderRouter.route(item, SmartConfig.INPUT_DIR)
                                job = ProcessingJob(
                                    video_path=item,
                                    routing=routing,
                                    priority=routing.priority
                                )
                                self.queue.put(job)
                                print(f"\nQueued: {item.name} [{routing.profile.name}]")

    def _process_loop(self):
        """Process queued jobs"""
        while self.running:
            try:
                job = self.queue.get(timeout=1.0)
                result = self.processor.process(job)

                if result.success:
                    print(f"\nCompleted: {result.input_path.name}")
                    print(f"  Output: {result.output_path.name if result.output_path else 'N/A'}")
                    print(f"  Time: {result.processing_time:.1f}s")
                    # Persist to database
                    self.processed_db.mark_processed(
                        result.input_path,
                        status="completed",
                        output_path=result.output_path
                    )
                else:
                    print(f"\nFailed: {result.input_path.name}")
                    print(f"  Error: {result.error}")
                    # Persist failure to database
                    self.processed_db.mark_processed(
                        result.input_path,
                        status="failed",
                        error=result.error
                    )

            except Empty:
                continue
            except Exception as e:
                logging.exception(f"Process error: {e}")

        print("\nGoodbye!")

    def start_background(self):
        """Start watching in background threads (non-blocking for FastAPI)"""
        if self.running:
            return False  # Already running

        self.running = True
        self._watcher_thread = threading.Thread(target=self._watch_loop, daemon=True, name="folder-watcher")
        self._processor_thread = threading.Thread(target=self._process_loop, daemon=True, name="video-processor")

        self._watcher_thread.start()
        self._processor_thread.start()

        logging.info(f"Folder watcher started: watching {SmartConfig.INPUT_DIR}")
        return True

    def stop(self):
        """Stop watching gracefully"""
        if not self.running:
            return False

        self.running = False
        logging.info("Folder watcher stopped")
        return True

    def get_status(self) -> dict:
        """Get watcher status"""
        return {
            "running": self.running,
            "input_dir": str(SmartConfig.INPUT_DIR),
            "output_dir": str(SmartConfig.OUTPUT_DIR),
            "queue_size": self.queue.qsize(),
            "processed_count": len(self.processed),
            "encoder": self.processor.encoder if self.processor else "unknown",
        }


# Global watcher instance (initialized lazily)
_folder_watcher: Optional[SmartFolderWatcher] = None


def get_folder_watcher() -> SmartFolderWatcher:
    """Get or create the global folder watcher instance"""
    global _folder_watcher
    if _folder_watcher is None:
        processor = SmartVideoProcessor()
        _folder_watcher = SmartFolderWatcher(processor)
    return _folder_watcher


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(SmartConfig.LOG_DIR / 'smart_processor.log'),
            logging.StreamHandler()
        ]
    )

    processor = SmartVideoProcessor()
    watcher = SmartFolderWatcher(processor)
    watcher.start()


if __name__ == '__main__':
    main()
