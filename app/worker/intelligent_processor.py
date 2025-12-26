#!/usr/bin/env python3
"""
NubHQ Intelligent Video Processor
==================================
Self-learning video pipeline that:
1. Watches input folder for new videos
2. Analyzes quality and content
3. Prompts user when uncertain
4. Learns preferences over time
5. Eventually runs fully autonomously

🦭 The walrus gets smarter with every video!
"""

import os
import sys
import json
import time
import hashlib
import logging
import sqlite3
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
from queue import Queue, Empty
import signal

# Import engagement scorer
try:
    from .engagement_scorer import EngagementScorer, EngagementScore
    HAS_ENGAGEMENT_SCORER = True
except ImportError:
    try:
        from engagement_scorer import EngagementScorer, EngagementScore
        HAS_ENGAGEMENT_SCORER = True
    except ImportError:
        HAS_ENGAGEMENT_SCORER = False
        logging.warning("Engagement scorer not available")

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Pipeline configuration"""
    # Folders
    INPUT_DIR = Path(os.environ.get('NUBHQ_INPUT', '/Volumes/NUB_Workspace/input'))
    PROCESSING_DIR = Path(os.environ.get('NUBHQ_PROCESSING', '/Volumes/NUB_Workspace/processing'))
    OUTPUT_DIR = Path(os.environ.get('NUBHQ_OUTPUT', '/Volumes/NUB_Workspace/output'))
    ARCHIVE_DIR = Path(os.environ.get('NUBHQ_ARCHIVE', '/Volumes/NUB_Workspace/archive'))
    DATA_DIR = Path(os.environ.get('NUBHQ_DATA', '/Volumes/NUB_Workspace/.nubhq'))

    # Output subdirectories
    AUTO_QUEUED_DIR = OUTPUT_DIR / 'auto-queued'
    REVIEW_DIR = OUTPUT_DIR / 'review'
    LIBRARY_DIR = OUTPUT_DIR / 'library'

    # Learning thresholds
    CONFIDENCE_THRESHOLD = 0.85  # Auto-apply when this confident
    MIN_SAMPLES_FOR_AUTO = 5     # Need this many examples before auto-applying
    SIMILARITY_THRESHOLD = 0.8   # How similar videos need to be for pattern match

    # Engagement thresholds
    ENGAGEMENT_AUTO_QUEUE_THRESHOLD = 0.85  # Auto-queue if engagement confidence >= this

    # Processing
    SUPPORTED_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mts', '.m2ts'}
    POLL_INTERVAL = 2.0  # Seconds between folder checks

    # Database
    DB_PATH = DATA_DIR / 'preferences.db'

    # API Configuration
    API_BASE_URL = os.environ.get('NUBHQ_API_URL', 'http://localhost:8000/api')
    API_TOKEN = os.environ.get('NUBHQ_API_TOKEN', '')  # JWT token for auth

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories (fails silently if permissions denied)"""
        for d in [cls.INPUT_DIR, cls.PROCESSING_DIR, cls.OUTPUT_DIR,
                  cls.ARCHIVE_DIR, cls.DATA_DIR, cls.AUTO_QUEUED_DIR,
                  cls.REVIEW_DIR, cls.LIBRARY_DIR]:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logging.warning(f"Cannot create directory {d} - permission denied")


# ============================================================
# DATA MODELS
# ============================================================

class DecisionType(Enum):
    """Types of decisions the system can learn"""
    OUTPUT_RESOLUTION = "output_resolution"
    OUTPUT_QUALITY = "output_quality"
    COLOR_GRADE = "color_grade"
    AUDIO_NORMALIZE = "audio_normalize"
    NOISE_REDUCTION = "noise_reduction"
    CROP_STYLE = "crop_style"
    ADD_INTRO = "add_intro"
    ADD_OUTRO = "add_outro"
    SUBTITLE_STYLE = "subtitle_style"
    HIGHLIGHT_STYLE = "highlight_style"
    EXPORT_PLATFORMS = "export_platforms"


@dataclass
class VideoAnalysis:
    """Analysis results for a video file"""
    path: str
    filename: str
    duration: float
    width: int
    height: int
    fps: float
    bitrate: int
    codec: str
    
    # Audio
    has_audio: bool
    audio_codec: Optional[str]
    audio_channels: int
    audio_sample_rate: int
    audio_bitrate: int
    
    # Quality metrics
    avg_brightness: float  # 0-255
    is_dark: bool
    is_overexposed: bool
    has_noise: bool
    noise_level: float  # 0-1
    
    # Audio quality
    peak_audio_db: float
    avg_audio_db: float
    has_clipping: bool
    has_background_noise: bool
    speech_detected: bool
    music_detected: bool
    
    # Content hints
    scene_changes: int
    motion_intensity: float  # 0-1
    
    # Fingerprint for similarity matching
    fingerprint: str = ""
    
    def __post_init__(self):
        """Generate fingerprint after initialization"""
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """Create a fingerprint for similarity matching"""
        # Bucket values into categories for matching
        res_bucket = "4k" if self.width >= 3840 else "1080p" if self.width >= 1920 else "720p" if self.width >= 1280 else "sd"
        duration_bucket = "short" if self.duration < 60 else "medium" if self.duration < 300 else "long"
        brightness_bucket = "dark" if self.is_dark else "bright" if self.is_overexposed else "normal"
        audio_bucket = "loud" if self.peak_audio_db > -3 else "quiet" if self.peak_audio_db < -20 else "normal"
        content_bucket = "static" if self.motion_intensity < 0.3 else "dynamic" if self.motion_intensity > 0.7 else "moderate"
        
        fp = f"{res_bucket}|{duration_bucket}|{brightness_bucket}|{audio_bucket}|{content_bucket}"
        return hashlib.md5(fp.encode()).hexdigest()[:12]


@dataclass
class UserDecision:
    """A decision made by the user"""
    decision_type: str
    choice: str
    video_fingerprint: str
    video_characteristics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    always_apply: bool = False  # User said "always do this"


@dataclass 
class ProcessingResult:
    """Result of processing a video"""
    success: bool
    input_path: str
    output_paths: List[str]
    decisions_made: Dict[str, str]
    processing_time: float
    error: Optional[str] = None


# ============================================================
# VIDEO ANALYZER
# ============================================================

class VideoAnalyzer:
    """Analyze video files for quality and content"""
    
    @staticmethod
    def analyze(video_path: Path) -> VideoAnalysis:
        """Full analysis of a video file"""
        path_str = str(video_path)
        
        # Get basic info with ffprobe
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', path_str
        ]
        
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=60)
            probe_data = json.loads(result.stdout)
        except Exception as e:
            logging.error(f"Failed to probe video: {e}")
            raise
        
        # Extract video stream info
        video_stream = next((s for s in probe_data.get('streams', []) 
                            if s['codec_type'] == 'video'), None)
        audio_stream = next((s for s in probe_data.get('streams', []) 
                            if s['codec_type'] == 'audio'), None)
        format_info = probe_data.get('format', {})
        
        if not video_stream:
            raise ValueError("No video stream found")
        
        # Parse video info
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        duration = float(format_info.get('duration', 0))
        
        # Parse FPS (can be fraction like "30000/1001")
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)
        
        bitrate = int(format_info.get('bit_rate', 0))
        codec = video_stream.get('codec_name', 'unknown')
        
        # Audio info
        has_audio = audio_stream is not None
        audio_codec = audio_stream.get('codec_name') if audio_stream else None
        audio_channels = int(audio_stream.get('channels', 0)) if audio_stream else 0
        audio_sample_rate = int(audio_stream.get('sample_rate', 0)) if audio_stream else 0
        audio_bitrate = int(audio_stream.get('bit_rate', 0)) if audio_stream else 0
        
        # Analyze quality (sample frames)
        brightness_info = VideoAnalyzer._analyze_brightness(path_str, duration)
        audio_info = VideoAnalyzer._analyze_audio(path_str) if has_audio else {
            'peak_db': -100, 'avg_db': -100, 'has_clipping': False,
            'has_background_noise': False, 'speech_detected': False, 'music_detected': False
        }
        motion_info = VideoAnalyzer._analyze_motion(path_str, duration)
        
        return VideoAnalysis(
            path=path_str,
            filename=video_path.name,
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            codec=codec,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_channels=audio_channels,
            audio_sample_rate=audio_sample_rate,
            audio_bitrate=audio_bitrate,
            avg_brightness=brightness_info['avg'],
            is_dark=brightness_info['is_dark'],
            is_overexposed=brightness_info['is_overexposed'],
            has_noise=brightness_info['has_noise'],
            noise_level=brightness_info['noise_level'],
            peak_audio_db=audio_info['peak_db'],
            avg_audio_db=audio_info['avg_db'],
            has_clipping=audio_info['has_clipping'],
            has_background_noise=audio_info['has_background_noise'],
            speech_detected=audio_info['speech_detected'],
            music_detected=audio_info['music_detected'],
            scene_changes=motion_info['scene_changes'],
            motion_intensity=motion_info['intensity'],
        )
    
    @staticmethod
    def _analyze_brightness(path: str, duration: float) -> Dict:
        """Sample frames for brightness analysis"""
        # Sample 5 frames evenly distributed
        sample_times = [duration * i / 6 for i in range(1, 6)]
        brightnesses = []
        
        for t in sample_times:
            cmd = [
                'ffmpeg', '-ss', str(t), '-i', path,
                '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'gray',
                '-'
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=10)
                if result.stdout:
                    # Calculate average brightness of frame
                    pixels = list(result.stdout)
                    if pixels:
                        avg = sum(pixels) / len(pixels)
                        brightnesses.append(avg)
            except:
                pass
        
        avg_brightness = sum(brightnesses) / len(brightnesses) if brightnesses else 128
        
        return {
            'avg': avg_brightness,
            'is_dark': avg_brightness < 50,
            'is_overexposed': avg_brightness > 220,
            'has_noise': False,  # Would need more complex analysis
            'noise_level': 0.0,
        }
    
    @staticmethod
    def _analyze_audio(path: str) -> Dict:
        """Analyze audio characteristics"""
        cmd = [
            'ffmpeg', '-i', path, '-af', 
            'volumedetect,silencedetect=noise=-30dB:d=0.5',
            '-f', 'null', '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            stderr = result.stderr
            
            # Parse volume info
            peak_db = -100.0
            avg_db = -100.0
            
            for line in stderr.split('\n'):
                if 'max_volume:' in line:
                    try:
                        peak_db = float(line.split('max_volume:')[1].split('dB')[0].strip())
                    except:
                        pass
                if 'mean_volume:' in line:
                    try:
                        avg_db = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                    except:
                        pass
            
            return {
                'peak_db': peak_db,
                'avg_db': avg_db,
                'has_clipping': peak_db > -0.5,
                'has_background_noise': avg_db > -20,  # Simplified
                'speech_detected': True,  # Would need ML for real detection
                'music_detected': True,
            }
        except Exception as e:
            logging.warning(f"Audio analysis failed: {e}")
            return {
                'peak_db': -20, 'avg_db': -30, 'has_clipping': False,
                'has_background_noise': False, 'speech_detected': False, 'music_detected': False
            }
    
    @staticmethod
    def _analyze_motion(path: str, duration: float) -> Dict:
        """Analyze motion and scene changes"""
        cmd = [
            'ffmpeg', '-i', path, '-vf', 
            'select=\'gt(scene,0.3)\',showinfo',
            '-f', 'null', '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            scene_changes = result.stderr.count('pts_time:')
            
            # Estimate motion intensity from scene changes
            changes_per_minute = (scene_changes / duration) * 60 if duration > 0 else 0
            intensity = min(1.0, changes_per_minute / 30)  # Normalize to 0-1
            
            return {
                'scene_changes': scene_changes,
                'intensity': intensity,
            }
        except:
            return {'scene_changes': 0, 'intensity': 0.5}


# ============================================================
# PREFERENCE LEARNER
# ============================================================

class PreferenceLearner:
    """Learn and apply user preferences"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logging.warning(f"Cannot create directory {self.db_path.parent} - permission denied")
            self._db_available = False
            return

        self._db_available = True
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_type TEXT NOT NULL,
                    choice TEXT NOT NULL,
                    video_fingerprint TEXT NOT NULL,
                    video_characteristics TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    always_apply INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS global_preferences (
                    decision_type TEXT PRIMARY KEY,
                    default_choice TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    sample_count INTEGER DEFAULT 0,
                    last_updated TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_fingerprint 
                ON decisions(video_fingerprint, decision_type)
            ''')
            
            conn.commit()
    
    def record_decision(self, decision: UserDecision):
        """Record a user decision for learning"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO decisions 
                (decision_type, choice, video_fingerprint, video_characteristics, timestamp, always_apply)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                decision.decision_type,
                decision.choice,
                decision.video_fingerprint,
                json.dumps(decision.video_characteristics),
                decision.timestamp,
                1 if decision.always_apply else 0
            ))
            
            # Update global preferences
            self._update_global_preference(conn, decision.decision_type)
            conn.commit()
    
    def _update_global_preference(self, conn: sqlite3.Connection, decision_type: str):
        """Update global preference stats for a decision type"""
        # Get all decisions of this type
        cursor = conn.execute('''
            SELECT choice, always_apply FROM decisions 
            WHERE decision_type = ?
        ''', (decision_type,))
        
        decisions = cursor.fetchall()
        if not decisions:
            return
        
        # Count choices
        choice_counts = {}
        has_always = None
        
        for choice, always in decisions:
            if always:
                has_always = choice
            choice_counts[choice] = choice_counts.get(choice, 0) + 1
        
        # If user said "always", use that
        if has_always:
            default_choice = has_always
            confidence = 1.0
        else:
            # Find most common choice
            default_choice = max(choice_counts, key=choice_counts.get)
            total = sum(choice_counts.values())
            confidence = choice_counts[default_choice] / total if total > 0 else 0
        
        conn.execute('''
            INSERT OR REPLACE INTO global_preferences 
            (decision_type, default_choice, confidence, sample_count, last_updated)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            decision_type,
            default_choice,
            confidence,
            len(decisions),
            datetime.now().isoformat()
        ))
    
    def get_recommendation(self, decision_type: str, analysis: VideoAnalysis) -> Tuple[Optional[str], float]:
        """
        Get a recommendation for a decision.
        Returns (choice, confidence) or (None, 0) if no recommendation.
        """
        with sqlite3.connect(self.db_path) as conn:
            # First check for "always apply" decisions
            cursor = conn.execute('''
                SELECT choice FROM decisions 
                WHERE decision_type = ? AND always_apply = 1
                LIMIT 1
            ''', (decision_type,))
            
            row = cursor.fetchone()
            if row:
                return row[0], 1.0
            
            # Check for similar videos
            cursor = conn.execute('''
                SELECT choice FROM decisions 
                WHERE decision_type = ? AND video_fingerprint = ?
            ''', (decision_type, analysis.fingerprint))
            
            similar_decisions = cursor.fetchall()
            if similar_decisions:
                # Use the most common choice for similar videos
                choices = [d[0] for d in similar_decisions]
                most_common = max(set(choices), key=choices.count)
                confidence = choices.count(most_common) / len(choices)
                
                if len(similar_decisions) >= Config.MIN_SAMPLES_FOR_AUTO:
                    return most_common, confidence
            
            # Fall back to global preference
            cursor = conn.execute('''
                SELECT default_choice, confidence, sample_count 
                FROM global_preferences 
                WHERE decision_type = ?
            ''', (decision_type,))
            
            row = cursor.fetchone()
            if row and row[2] >= Config.MIN_SAMPLES_FOR_AUTO:
                return row[0], row[1]
            
            return None, 0.0
    
    def get_stats(self) -> Dict:
        """Get learning statistics"""
        if not getattr(self, '_db_available', True):
            return {'total_decisions': 0, 'preferences': {}}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM decisions')
            total_decisions = cursor.fetchone()[0]

            cursor = conn.execute('''
                SELECT decision_type, confidence, sample_count
                FROM global_preferences
            ''')

            preferences = {}
            for row in cursor.fetchall():
                preferences[row[0]] = {
                    'confidence': row[1],
                    'samples': row[2],
                    'auto_enabled': row[1] >= Config.CONFIDENCE_THRESHOLD and row[2] >= Config.MIN_SAMPLES_FOR_AUTO
                }

            return {
                'total_decisions': total_decisions,
                'preferences': preferences,
            }

    def learn_from_approval(
        self,
        video_fingerprint: str,
        approved: bool,
        engagement_score: float,
        engagement_confidence: float,
        user_edits: Optional[Dict] = None
    ):
        """
        Learn from an approval queue decision.

        If approved without edits -> boost confidence for similar videos
        If approved with edits -> learn the edit patterns
        If rejected -> lower confidence for similar videos
        """
        with sqlite3.connect(self.db_path) as conn:
            # Create approval_feedback table if not exists
            conn.execute('''
                CREATE TABLE IF NOT EXISTS approval_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_fingerprint TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    engagement_score REAL,
                    engagement_confidence REAL,
                    user_edits TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')

            # Record the feedback
            conn.execute('''
                INSERT INTO approval_feedback
                (video_fingerprint, approved, engagement_score, engagement_confidence, user_edits, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                video_fingerprint,
                1 if approved else 0,
                engagement_score,
                engagement_confidence,
                json.dumps(user_edits) if user_edits else None,
                datetime.now().isoformat()
            ))

            # Update engagement threshold based on outcomes
            self._update_engagement_threshold(conn)

            conn.commit()

        logging.info(f"Learned from approval: fingerprint={video_fingerprint[:8]}, approved={approved}")

    def _update_engagement_threshold(self, conn: sqlite3.Connection):
        """
        Adjust engagement auto-queue threshold based on approval patterns.

        If many auto-queued videos are being rejected, raise the threshold.
        If most are approved, we can potentially lower it.
        """
        cursor = conn.execute('''
            SELECT approved, engagement_confidence
            FROM approval_feedback
            WHERE engagement_confidence >= ?
            ORDER BY id DESC
            LIMIT 20
        ''', (Config.ENGAGEMENT_AUTO_QUEUE_THRESHOLD,))

        recent = cursor.fetchall()
        if len(recent) < 5:
            return  # Not enough data

        approved_count = sum(1 for r in recent if r[0] == 1)
        approval_rate = approved_count / len(recent)

        # Store the learned threshold adjustment
        if approval_rate < 0.7:
            # Too many rejections - log recommendation to raise threshold
            logging.warning(f"Low approval rate ({approval_rate:.0%}) - consider raising ENGAGEMENT_AUTO_QUEUE_THRESHOLD")
        elif approval_rate > 0.95:
            # Very high approval - could potentially lower threshold
            logging.info(f"High approval rate ({approval_rate:.0%}) - threshold is working well")

    def get_approval_stats(self) -> Dict:
        """Get approval feedback statistics"""
        if not getattr(self, '_db_available', True):
            return {'total_feedback': 0, 'approved': 0, 'rejected': 0, 'approval_rate': 0, 'avg_engagement_score': 0}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT
                        COUNT(*) as total,
                        SUM(approved) as approved_count,
                        AVG(engagement_score) as avg_score
                    FROM approval_feedback
                ''')
                row = cursor.fetchone()

                if row and row[0] > 0:
                    return {
                        'total_feedback': row[0],
                        'approved': row[1] or 0,
                        'rejected': row[0] - (row[1] or 0),
                        'approval_rate': (row[1] or 0) / row[0],
                        'avg_engagement_score': row[2] or 0,
                    }
        except:
            pass

        return {
            'total_feedback': 0,
            'approved': 0,
            'rejected': 0,
            'approval_rate': 0,
            'avg_engagement_score': 0,
        }


# ============================================================
# APPROVAL QUEUE INTEGRATION
# ============================================================

class ApprovalQueueIntegration:
    """Push processed videos to the approval queue API"""

    def __init__(self):
        self.api_url = Config.API_BASE_URL
        self.token = Config.API_TOKEN

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth"""
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        return headers

    def _generate_thumbnail(self, video_path: Path) -> Optional[str]:
        """Generate base64 thumbnail from video"""
        import tempfile
        import base64

        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                # Extract frame at 25% of duration
                cmd = [
                    'ffprobe', '-v', 'quiet', '-print_format', 'json',
                    '-show_format', str(video_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                data = json.loads(result.stdout)
                duration = float(data.get('format', {}).get('duration', 10))

                # Extract thumbnail
                cmd = [
                    'ffmpeg', '-ss', str(duration * 0.25), '-i', str(video_path),
                    '-vframes', '1', '-q:v', '5', '-vf', 'scale=480:-1',
                    '-y', tmp.name
                ]
                subprocess.run(cmd, capture_output=True, timeout=30)

                if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                    with open(tmp.name, 'rb') as f:
                        thumb_b64 = base64.b64encode(f.read()).decode()
                    os.unlink(tmp.name)
                    return thumb_b64
        except Exception as e:
            logging.warning(f"Thumbnail generation failed: {e}")

        return None

    def push_to_queue(
        self,
        video_path: Path,
        analysis: 'VideoAnalysis',
        engagement: 'EngagementScore',
        decisions: Dict[str, str],
        recipient: str = "schedule"
    ) -> Tuple[bool, Optional[str]]:
        """
        Push a processed video to the approval queue.

        Returns (success, approval_id or error message)
        """
        try:
            # Generate thumbnail
            thumbnail = self._generate_thumbnail(video_path)

            # Build approval content
            content_data = {
                'filename': video_path.name,
                'path': str(video_path),
                'duration': analysis.duration,
                'resolution': f"{analysis.width}x{analysis.height}",
                'fps': analysis.fps,
                'engagement_score': engagement.overall_score,
                'engagement_confidence': engagement.confidence,
                'tags': engagement.tags,
                'best_moments': [
                    {
                        'start': m.start_time,
                        'end': m.end_time,
                        'score': m.score,
                        'reason': m.reason
                    }
                    for m in engagement.best_moments[:5]
                ],
                'processing_decisions': decisions,
                'thumbnail': thumbnail,
            }

            # Build approval payload
            payload = {
                'type': 'video',
                'content': json.dumps(content_data),
                'recipient': recipient,
            }

            # POST to approvals endpoint
            response = requests.post(
                f"{self.api_url}/approvals",
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )

            if response.status_code in [200, 201]:
                data = response.json()
                approval_id = data.get('id')
                logging.info(f"Video queued for approval: {approval_id}")
                return True, str(approval_id)
            else:
                error = f"API error {response.status_code}: {response.text[:200]}"
                logging.error(error)
                return False, error

        except requests.exceptions.ConnectionError:
            error = f"Cannot connect to API at {self.api_url}"
            logging.error(error)
            return False, error
        except Exception as e:
            error = f"Queue push failed: {str(e)}"
            logging.exception(error)
            return False, error

    def check_approval_status(self, approval_id: str) -> Optional[Dict]:
        """Check the status of an approval"""
        try:
            response = requests.get(
                f"{self.api_url}/approvals/{approval_id}",
                headers=self._get_headers(),
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logging.warning(f"Status check failed: {e}")
        return None


# ============================================================
# USER PROMPT SYSTEM
# ============================================================

class PromptSystem:
    """Interactive prompts for uncertain decisions"""
    
    # Decision options with descriptions
    OPTIONS = {
        DecisionType.OUTPUT_RESOLUTION: {
            'title': '📺 Output Resolution',
            'description': 'What resolution should the output be?',
            'choices': [
                ('source', 'Keep source resolution'),
                ('4k', '4K (3840x2160)'),
                ('1080p', '1080p (1920x1080)'),
                ('720p', '720p (1280x720)'),
                ('square_1080', 'Square 1080x1080 (Instagram)'),
                ('vertical_1080', 'Vertical 1080x1920 (Stories/TikTok)'),
            ]
        },
        DecisionType.OUTPUT_QUALITY: {
            'title': '🎬 Output Quality',
            'description': 'Video quality vs file size trade-off',
            'choices': [
                ('max', 'Maximum quality (largest file)'),
                ('high', 'High quality (recommended)'),
                ('medium', 'Medium quality (balanced)'),
                ('low', 'Low quality (smallest file)'),
                ('web', 'Web optimized (fast loading)'),
            ]
        },
        DecisionType.COLOR_GRADE: {
            'title': '🎨 Color Grading',
            'description': 'How should colors be processed?',
            'choices': [
                ('none', 'No color grading'),
                ('auto_correct', 'Auto-correct exposure/white balance'),
                ('warm', 'Warm & vibrant'),
                ('cool', 'Cool & moody'),
                ('vintage', 'Vintage film look'),
                ('high_contrast', 'High contrast punchy'),
            ]
        },
        DecisionType.AUDIO_NORMALIZE: {
            'title': '🔊 Audio Normalization',
            'description': 'How should audio levels be adjusted?',
            'choices': [
                ('none', 'No normalization'),
                ('light', 'Light normalization (-16 LUFS)'),
                ('standard', 'Standard broadcast (-14 LUFS)'),
                ('loud', 'Loud/punchy (-11 LUFS)'),
                ('youtube', 'YouTube optimized (-13 LUFS)'),
            ]
        },
        DecisionType.NOISE_REDUCTION: {
            'title': '🔇 Noise Reduction',
            'description': 'Background noise was detected. How to handle?',
            'choices': [
                ('none', 'Leave audio as-is'),
                ('light', 'Light noise reduction'),
                ('medium', 'Medium noise reduction'),
                ('aggressive', 'Aggressive noise reduction'),
            ]
        },
        DecisionType.CROP_STYLE: {
            'title': '✂️ Crop Style',
            'description': 'How should the video be framed?',
            'choices': [
                ('none', 'No cropping'),
                ('center', 'Center crop'),
                ('smart', 'Smart crop (follow action)'),
                ('face', 'Face tracking crop'),
            ]
        },
        DecisionType.ADD_INTRO: {
            'title': '🎬 Intro',
            'description': 'Add branded intro?',
            'choices': [
                ('none', 'No intro'),
                ('short', 'Short intro (2s)'),
                ('standard', 'Standard intro (5s)'),
            ]
        },
        DecisionType.ADD_OUTRO: {
            'title': '🎬 Outro',
            'description': 'Add branded outro?',
            'choices': [
                ('none', 'No outro'),
                ('short', 'Short outro (3s)'),
                ('standard', 'Standard outro with CTA (5s)'),
                ('subscribe', 'Subscribe reminder outro (8s)'),
            ]
        },
        DecisionType.SUBTITLE_STYLE: {
            'title': '💬 Subtitles',
            'description': 'Generate and burn in subtitles?',
            'choices': [
                ('none', 'No subtitles'),
                ('minimal', 'Minimal style'),
                ('karaoke', 'Karaoke word-by-word'),
                ('bold', 'Bold attention-grabbing'),
            ]
        },
        DecisionType.HIGHLIGHT_STYLE: {
            'title': '⭐ Highlight Detection',
            'description': 'How to select best moments?',
            'choices': [
                ('all', 'Keep full video'),
                ('top_moments', 'Top 5 moments only'),
                ('energy', 'High energy moments'),
                ('audio_peaks', 'Audio peak moments'),
            ]
        },
        DecisionType.EXPORT_PLATFORMS: {
            'title': '📱 Export Platforms',
            'description': 'Which platform versions to create?',
            'choices': [
                ('all', 'All platforms'),
                ('youtube', 'YouTube only'),
                ('instagram', 'Instagram (feed + stories)'),
                ('tiktok', 'TikTok/Shorts'),
                ('youtube,instagram', 'YouTube + Instagram'),
            ]
        },
    }
    
    @classmethod
    def prompt(cls, decision_type: DecisionType, analysis: VideoAnalysis, 
               recommendation: Optional[str] = None) -> Tuple[str, bool]:
        """
        Prompt user for a decision.
        Returns (choice, always_apply)
        """
        opts = cls.OPTIONS.get(decision_type)
        if not opts:
            return 'default', False
        
        print("\n" + "=" * 60)
        print(f"🦭 {opts['title']}")
        print("=" * 60)
        print(f"\n📁 File: {analysis.filename}")
        print(f"📐 {analysis.width}x{analysis.height} @ {analysis.fps:.1f}fps")
        print(f"⏱️  Duration: {analysis.duration:.1f}s")
        
        if decision_type in [DecisionType.AUDIO_NORMALIZE, DecisionType.NOISE_REDUCTION]:
            print(f"🔊 Audio: Peak {analysis.peak_audio_db:.1f}dB, Avg {analysis.avg_audio_db:.1f}dB")
        
        if decision_type == DecisionType.COLOR_GRADE:
            status = "🌑 Dark" if analysis.is_dark else "☀️ Overexposed" if analysis.is_overexposed else "✅ Normal"
            print(f"💡 Brightness: {status} ({analysis.avg_brightness:.0f}/255)")
        
        print(f"\n{opts['description']}\n")
        
        for i, (key, desc) in enumerate(opts['choices'], 1):
            rec_marker = " ← recommended" if key == recommendation else ""
            print(f"  [{i}] {desc}{rec_marker}")
        
        print(f"\n  [A] Always use my choice for similar videos")
        print(f"  [S] Skip this video")
        print()
        
        while True:
            try:
                choice = input("Your choice: ").strip().upper()
                
                if choice == 'S':
                    return 'skip', False
                
                always = False
                if choice.endswith('A'):
                    always = True
                    choice = choice[:-1].strip()
                elif choice == 'A':
                    # Just "A" means use recommendation with always
                    if recommendation:
                        return recommendation, True
                    choice = '1'
                    always = True
                
                idx = int(choice) - 1
                if 0 <= idx < len(opts['choices']):
                    return opts['choices'][idx][0], always
                
                print("Invalid choice. Try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nUsing default...")
                return opts['choices'][0][0], False


# ============================================================
# VIDEO PROCESSOR
# ============================================================

class VideoProcessor:
    """Process videos with learned preferences"""

    def __init__(self, learner: PreferenceLearner, enable_queue: bool = True):
        self.learner = learner
        self.enable_queue = enable_queue

        # Initialize engagement scorer
        if HAS_ENGAGEMENT_SCORER:
            self.engagement_scorer = EngagementScorer()
        else:
            self.engagement_scorer = None

        # Initialize queue integration
        if enable_queue:
            self.queue_integration = ApprovalQueueIntegration()
        else:
            self.queue_integration = None
    
    def process(self, video_path: Path, interactive: bool = True) -> ProcessingResult:
        """Process a single video"""
        start_time = time.time()
        decisions = {}
        
        try:
            # Analyze the video
            print(f"\n🦭 Analyzing: {video_path.name}")
            analysis = VideoAnalyzer.analyze(video_path)
            
            # Determine which decisions to prompt for
            decisions_needed = self._determine_decisions(analysis)
            
            for decision_type in decisions_needed:
                rec, confidence = self.learner.get_recommendation(decision_type.value, analysis)
                
                if confidence >= Config.CONFIDENCE_THRESHOLD:
                    # Auto-apply with high confidence
                    print(f"  ✓ {decision_type.value}: {rec} (auto, {confidence:.0%} confident)")
                    decisions[decision_type.value] = rec
                elif interactive:
                    # Need to ask user
                    choice, always = PromptSystem.prompt(decision_type, analysis, rec)
                    
                    if choice == 'skip':
                        return ProcessingResult(
                            success=False,
                            input_path=str(video_path),
                            output_paths=[],
                            decisions_made={},
                            processing_time=time.time() - start_time,
                            error="Skipped by user"
                        )
                    
                    decisions[decision_type.value] = choice
                    
                    # Record for learning
                    self.learner.record_decision(UserDecision(
                        decision_type=decision_type.value,
                        choice=choice,
                        video_fingerprint=analysis.fingerprint,
                        video_characteristics={
                            'width': analysis.width,
                            'height': analysis.height,
                            'duration': analysis.duration,
                            'is_dark': analysis.is_dark,
                            'has_audio': analysis.has_audio,
                        },
                        always_apply=always
                    ))
                else:
                    # Non-interactive, use recommendation or default
                    decisions[decision_type.value] = rec or 'default'
            
            # Process the video with decided settings
            output_paths = self._execute_processing(video_path, analysis, decisions)

            # Run engagement scoring
            engagement = None
            if self.engagement_scorer and output_paths:
                print(f"\n📊 Scoring engagement...")
                output_path = Path(output_paths[0])
                engagement = self.engagement_scorer.score(output_path, analysis)
                print(f"   Score: {engagement.overall_score:.2f} (confidence: {engagement.confidence:.2f})")
                print(f"   Tags: {', '.join(engagement.tags)}")

            # Determine destination based on engagement confidence
            final_output_path = None
            queued = False

            if output_paths:
                output_path = Path(output_paths[0])

                if engagement and engagement.confidence >= Config.ENGAGEMENT_AUTO_QUEUE_THRESHOLD:
                    # High confidence - auto-queue
                    final_dest = Config.AUTO_QUEUED_DIR / output_path.name
                    if final_dest.exists():
                        final_dest = Config.AUTO_QUEUED_DIR / f"{output_path.stem}_{int(time.time())}{output_path.suffix}"
                    output_path.rename(final_dest)
                    final_output_path = final_dest
                    output_paths = [str(final_dest)]

                    # Push to approval queue
                    if self.queue_integration and self.enable_queue:
                        print(f"\n🚀 Auto-queuing for approval...")
                        success, result = self.queue_integration.push_to_queue(
                            final_dest, analysis, engagement, decisions
                        )
                        if success:
                            queued = True
                            print(f"   ✅ Queued! Approval ID: {result}")
                        else:
                            print(f"   ⚠️ Queue failed: {result}")
                else:
                    # Low confidence - hold for review
                    final_dest = Config.REVIEW_DIR / output_path.name
                    if final_dest.exists():
                        final_dest = Config.REVIEW_DIR / f"{output_path.stem}_{int(time.time())}{output_path.suffix}"
                    output_path.rename(final_dest)
                    final_output_path = final_dest
                    output_paths = [str(final_dest)]

                    confidence_pct = engagement.confidence * 100 if engagement else 0
                    print(f"\n📋 Held for review (confidence: {confidence_pct:.0f}%)")
                    print(f"   → {final_dest}")

            # Move original to archive
            archive_path = Config.ARCHIVE_DIR / video_path.name
            if archive_path.exists():
                archive_path = Config.ARCHIVE_DIR / f"{video_path.stem}_{int(time.time())}{video_path.suffix}"
            video_path.rename(archive_path)

            return ProcessingResult(
                success=True,
                input_path=str(video_path),
                output_paths=output_paths,
                decisions_made=decisions,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logging.exception(f"Processing failed for {video_path}")
            return ProcessingResult(
                success=False,
                input_path=str(video_path),
                output_paths=[],
                decisions_made=decisions,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    def _determine_decisions(self, analysis: VideoAnalysis) -> List[DecisionType]:
        """Determine which decisions need to be made based on video analysis"""
        decisions = []
        
        # Always ask about resolution and quality for now
        decisions.append(DecisionType.OUTPUT_RESOLUTION)
        decisions.append(DecisionType.OUTPUT_QUALITY)
        
        # Color grading if exposure issues detected
        if analysis.is_dark or analysis.is_overexposed:
            decisions.append(DecisionType.COLOR_GRADE)
        
        # Audio decisions if has audio
        if analysis.has_audio:
            decisions.append(DecisionType.AUDIO_NORMALIZE)
            
            if analysis.has_background_noise:
                decisions.append(DecisionType.NOISE_REDUCTION)
        
        # Always ask about platforms for now
        decisions.append(DecisionType.EXPORT_PLATFORMS)
        
        return decisions
    
    def _execute_processing(self, video_path: Path, analysis: VideoAnalysis, 
                           decisions: Dict[str, str]) -> List[str]:
        """Execute the actual video processing"""
        output_paths = []
        
        # Build FFmpeg command based on decisions
        output_name = f"{video_path.stem}_processed"
        
        # Resolution mapping
        res_map = {
            'source': (analysis.width, analysis.height),
            '4k': (3840, 2160),
            '1080p': (1920, 1080),
            '720p': (1280, 720),
            'square_1080': (1080, 1080),
            'vertical_1080': (1080, 1920),
        }
        
        # Quality mapping (CRF values - lower = better)
        quality_map = {
            'max': 16,
            'high': 20,
            'medium': 23,
            'low': 28,
            'web': 26,
        }
        
        # Audio normalization
        audio_filters = []
        audio_norm = decisions.get('audio_normalize', 'none')
        if audio_norm != 'none':
            lufs_map = {'light': -16, 'standard': -14, 'loud': -11, 'youtube': -13}
            lufs = lufs_map.get(audio_norm, -14)
            audio_filters.append(f"loudnorm=I={lufs}:TP=-1.5:LRA=11")
        
        # Noise reduction
        noise_red = decisions.get('noise_reduction', 'none')
        if noise_red != 'none':
            strength_map = {'light': 0.2, 'medium': 0.5, 'aggressive': 0.8}
            strength = strength_map.get(noise_red, 0.3)
            audio_filters.append(f"afftdn=nf=-{int(strength * 40)}")
        
        # Build video filters
        video_filters = []
        
        # Resolution
        res_choice = decisions.get('output_resolution', 'source')
        target_w, target_h = res_map.get(res_choice, (analysis.width, analysis.height))
        
        if (target_w, target_h) != (analysis.width, analysis.height):
            video_filters.append(f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease")
            video_filters.append(f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2")
        
        # Color grading
        color = decisions.get('color_grade', 'none')
        if color == 'auto_correct':
            video_filters.append("eq=brightness=0.06:contrast=1.1:saturation=1.1")
        elif color == 'warm':
            video_filters.append("colorbalance=rs=0.1:gs=0.05:bs=-0.1,eq=saturation=1.2")
        elif color == 'cool':
            video_filters.append("colorbalance=rs=-0.1:gs=0:bs=0.1,eq=saturation=0.9")
        elif color == 'vintage':
            video_filters.append("curves=vintage,eq=contrast=1.1:saturation=0.8")
        elif color == 'high_contrast':
            video_filters.append("eq=contrast=1.3:saturation=1.15:brightness=0.02")
        
        # Build output path
        output_path = Config.OUTPUT_DIR / f"{output_name}.mp4"
        
        # Construct FFmpeg command
        cmd = ['ffmpeg', '-y', '-i', str(video_path)]
        
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])
        
        if audio_filters:
            cmd.extend(['-af', ','.join(audio_filters)])
        
        # Quality
        quality = decisions.get('output_quality', 'high')
        crf = quality_map.get(quality, 20)
        cmd.extend(['-c:v', 'libx264', '-crf', str(crf), '-preset', 'medium'])
        cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        cmd.append(str(output_path))
        
        print(f"\n🎬 Processing with settings:")
        for k, v in decisions.items():
            print(f"   • {k}: {v}")
        
        print(f"\n⏳ Running FFmpeg...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f"✅ Created: {output_path}")
                output_paths.append(str(output_path))
            else:
                logging.error(f"FFmpeg failed: {result.stderr}")
                raise Exception(f"FFmpeg error: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            raise Exception("Processing timed out after 1 hour")
        
        return output_paths


# ============================================================
# FOLDER WATCHER
# ============================================================

class FolderWatcher:
    """Watch input folder for new videos"""
    
    def __init__(self, processor: VideoProcessor):
        self.processor = processor
        self.queue = Queue()
        self.processed_files = set()
        self.running = False
        self._lock = threading.Lock()
    
    def start(self):
        """Start watching for new videos"""
        self.running = True
        
        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("\n" + "=" * 60)
        print("🦭 NubHQ Intelligent Video Processor")
        print("=" * 60)
        print(f"\n📁 Watching: {Config.INPUT_DIR}")
        print(f"📤 Output:   {Config.OUTPUT_DIR}")
        print(f"🗄️  Archive:  {Config.ARCHIVE_DIR}")
        print("\nDrop videos into the input folder to process them!")
        print("The walrus learns your preferences over time 🧠")
        print("\nPress Ctrl+C to stop\n")
        
        # Start watcher thread
        watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        watcher_thread.start()
        
        # Process queue in main thread
        self._process_loop()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signal"""
        print("\n\n🦭 Shutting down gracefully...")
        self.running = False
    
    def _watch_loop(self):
        """Watch for new files"""
        while self.running:
            try:
                # Scan input directory
                for file_path in Config.INPUT_DIR.iterdir():
                    if not file_path.is_file():
                        continue
                    
                    if file_path.suffix.lower() not in Config.SUPPORTED_FORMATS:
                        continue
                    
                    with self._lock:
                        if str(file_path) in self.processed_files:
                            continue
                        self.processed_files.add(str(file_path))
                    
                    # Wait a moment to ensure file is fully written
                    initial_size = file_path.stat().st_size
                    time.sleep(1)
                    
                    if file_path.exists() and file_path.stat().st_size == initial_size:
                        self.queue.put(file_path)
                        print(f"\n📥 Found: {file_path.name}")
                
            except Exception as e:
                logging.error(f"Watch error: {e}")
            
            time.sleep(Config.POLL_INTERVAL)
    
    def _process_loop(self):
        """Process queued videos"""
        while self.running:
            try:
                video_path = self.queue.get(timeout=1.0)
                
                # Move to processing folder
                processing_path = Config.PROCESSING_DIR / video_path.name
                video_path.rename(processing_path)
                
                # Process
                result = self.processor.process(processing_path, interactive=True)
                
                if result.success:
                    print(f"\n✅ Completed: {result.input_path}")
                    print(f"   Time: {result.processing_time:.1f}s")
                    print(f"   Outputs: {len(result.output_paths)}")
                else:
                    print(f"\n❌ Failed: {result.error}")
                    # Move failed file back
                    if processing_path.exists():
                        failed_dir = Config.OUTPUT_DIR / "failed"
                        failed_dir.mkdir(exist_ok=True)
                        processing_path.rename(failed_dir / processing_path.name)
                
                # Show learning stats
                stats = self.processor.learner.get_stats()
                auto_count = sum(1 for p in stats['preferences'].values() if p.get('auto_enabled'))
                if stats['total_decisions'] > 0:
                    print(f"\n📊 Learning: {stats['total_decisions']} decisions recorded, {auto_count} now automatic")
                
            except Empty:
                continue
            except Exception as e:
                logging.exception(f"Process loop error: {e}")
        
        print("\n🦭 Goodbye!")


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.DATA_DIR / 'processor.log'),
            logging.StreamHandler()
        ]
    )
    
    # Ensure directories exist
    Config.ensure_dirs()
    
    # Initialize components
    learner = PreferenceLearner(Config.DB_PATH)
    processor = VideoProcessor(learner)
    watcher = FolderWatcher(processor)
    
    # Show current learning stats
    stats = learner.get_stats()
    if stats['total_decisions'] > 0:
        print(f"\n📊 Loaded {stats['total_decisions']} previous decisions")
        for dtype, info in stats['preferences'].items():
            status = "✅ AUTO" if info['auto_enabled'] else f"📝 {info['confidence']:.0%}"
            print(f"   • {dtype}: {status} ({info['samples']} samples)")
    
    # Start watching
    watcher.start()


if __name__ == '__main__':
    main()
