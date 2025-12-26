#!/usr/bin/env python3
"""
NubHQ Engagement Scorer
========================
Analyzes video content for engagement potential:
1. Technical metrics (motion, audio energy)
2. Face/performance detection
3. AI content analysis (optional)
4. Best moments extraction

🦭 The walrus knows what slaps!
"""

import os
import json
import logging
import subprocess
import tempfile
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# Optional AI imports
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================
# DATA MODELS
# ============================================================

class ContentTag(Enum):
    """Tags for categorizing content"""
    HIGH_ENERGY = "high-energy"
    PERFORMANCE = "performance"
    CROWD_SHOT = "crowd-shot"
    INTERVIEW = "interview"
    B_ROLL = "b-roll"
    SLOW_BURN = "slow-burn"
    STATIC = "static"
    MUSIC_HEAVY = "music-heavy"
    DIALOGUE = "dialogue"
    HIGHLIGHT = "highlight"
    TRANSITION = "transition"


@dataclass
class Moment:
    """A notable moment in the video"""
    start_time: float  # seconds
    end_time: float    # seconds
    score: float       # 0-1 engagement score
    reason: str        # Why this moment scored high
    tags: List[str] = field(default_factory=list)


@dataclass
class EngagementScore:
    """Complete engagement analysis for a video"""
    # Component scores (0-1)
    technical_score: float      # Motion, audio energy, visual complexity
    performance_score: float    # Face detection, center-frame activity
    ai_score: Optional[float]   # Vision AI analysis (if enabled)

    # Overall
    overall_score: float        # Weighted combination
    confidence: float           # How confident in this assessment

    # Metadata
    tags: List[str]             # Content tags
    best_moments: List[Moment]  # Top moments with timestamps

    # Analysis details
    analysis_time: float        # How long analysis took
    ai_enabled: bool            # Whether AI was used

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'technical_score': self.technical_score,
            'performance_score': self.performance_score,
            'ai_score': self.ai_score,
            'overall_score': self.overall_score,
            'confidence': self.confidence,
            'tags': self.tags,
            'best_moments': [
                {
                    'start_time': m.start_time,
                    'end_time': m.end_time,
                    'score': m.score,
                    'reason': m.reason,
                    'tags': m.tags
                }
                for m in self.best_moments
            ],
            'analysis_time': self.analysis_time,
            'ai_enabled': self.ai_enabled
        }


# ============================================================
# CONFIGURATION
# ============================================================

class EngagementConfig:
    """Engagement scoring configuration"""

    # Score weights
    TECHNICAL_WEIGHT = 0.4
    PERFORMANCE_WEIGHT = 0.4
    AI_WEIGHT = 0.2  # Only if AI enabled

    # Thresholds
    HIGH_MOTION_THRESHOLD = 0.6    # Above this = high energy
    LOW_MOTION_THRESHOLD = 0.2     # Below this = static
    AUDIO_ENERGY_THRESHOLD = -15   # dB above this = energetic
    FACE_CONFIDENCE_MIN = 0.7      # Minimum for counting face detection

    # Moment detection
    MOMENT_MIN_DURATION = 3.0      # Minimum moment length (seconds)
    MOMENT_MAX_DURATION = 30.0     # Maximum moment length (seconds)
    MOMENT_SAMPLE_INTERVAL = 2.0   # Sample every N seconds
    TOP_MOMENTS_COUNT = 10         # Max moments to return

    # AI settings
    AI_ENABLED = os.environ.get('NUBHQ_AI_SCORING', 'false').lower() == 'true'
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    AI_MODEL = "gpt-4o-mini"  # Cost-effective vision model
    AI_SAMPLE_FRAMES = 5     # Frames to send to AI


# ============================================================
# ENGAGEMENT SCORER
# ============================================================

class EngagementScorer:
    """Score video engagement potential"""

    def __init__(self, ai_enabled: bool = None):
        """
        Initialize scorer.
        ai_enabled: Override AI setting (default: use config)
        """
        self.ai_enabled = ai_enabled if ai_enabled is not None else EngagementConfig.AI_ENABLED

        if self.ai_enabled and not HAS_OPENAI:
            logging.warning("AI scoring requested but openai package not installed")
            self.ai_enabled = False

        if self.ai_enabled and not EngagementConfig.OPENAI_API_KEY:
            logging.warning("AI scoring requested but OPENAI_API_KEY not set")
            self.ai_enabled = False

    def score(self, video_path: Path, video_analysis: Any = None) -> EngagementScore:
        """
        Analyze video and return engagement score.

        video_path: Path to video file
        video_analysis: Optional pre-computed VideoAnalysis object
        """
        import time
        start_time = time.time()

        path_str = str(video_path)

        # Get video duration and basic info
        duration = self._get_duration(path_str)

        # Score technical metrics
        technical_score, technical_tags = self._score_technical(path_str, video_analysis, duration)

        # Score performance/face detection
        performance_score, performance_tags = self._score_performance(path_str, duration)

        # AI scoring (optional)
        ai_score = None
        ai_tags = []
        if self.ai_enabled:
            ai_score, ai_tags = self._score_with_ai(path_str, duration)

        # Find best moments
        best_moments = self._find_best_moments(path_str, duration)

        # Combine scores
        all_tags = list(set(technical_tags + performance_tags + ai_tags))
        overall_score, confidence = self._calculate_overall(
            technical_score, performance_score, ai_score, all_tags
        )

        analysis_time = time.time() - start_time

        return EngagementScore(
            technical_score=technical_score,
            performance_score=performance_score,
            ai_score=ai_score,
            overall_score=overall_score,
            confidence=confidence,
            tags=all_tags,
            best_moments=best_moments,
            analysis_time=analysis_time,
            ai_enabled=self.ai_enabled
        )

    def _get_duration(self, path: str) -> float:
        """Get video duration"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return float(data.get('format', {}).get('duration', 0))
        except:
            return 0.0

    def _score_technical(self, path: str, analysis: Any, duration: float) -> Tuple[float, List[str]]:
        """
        Score technical engagement metrics.
        Returns (score, tags)
        """
        tags = []
        scores = []

        # If we have pre-computed analysis, use it
        if analysis:
            motion = analysis.motion_intensity
            avg_db = analysis.avg_audio_db
            scene_changes = analysis.scene_changes
        else:
            # Compute from scratch
            motion = self._analyze_motion(path, duration)
            avg_db = self._analyze_audio_energy(path)
            scene_changes = self._count_scene_changes(path)

        # Motion score
        if motion > EngagementConfig.HIGH_MOTION_THRESHOLD:
            scores.append(0.9)
            tags.append(ContentTag.HIGH_ENERGY.value)
        elif motion < EngagementConfig.LOW_MOTION_THRESHOLD:
            scores.append(0.3)
            tags.append(ContentTag.STATIC.value)
        else:
            scores.append(0.5 + (motion - 0.3) * 1.5)  # Scale 0.3-0.6 to 0.5-0.95

        # Audio energy score
        if avg_db > EngagementConfig.AUDIO_ENERGY_THRESHOLD:
            scores.append(0.85)
            tags.append(ContentTag.MUSIC_HEAVY.value)
        elif avg_db > -25:
            scores.append(0.6)
        else:
            scores.append(0.4)
            tags.append(ContentTag.DIALOGUE.value) if avg_db > -40 else None

        # Scene change density (edits per minute)
        if duration > 0:
            edits_per_min = (scene_changes / duration) * 60
            if edits_per_min > 20:
                scores.append(0.9)  # Fast-paced editing
            elif edits_per_min > 8:
                scores.append(0.7)
            else:
                scores.append(0.4)
                tags.append(ContentTag.SLOW_BURN.value)

        # Clean up None tags
        tags = [t for t in tags if t]

        return sum(scores) / len(scores) if scores else 0.5, tags

    def _analyze_motion(self, path: str, duration: float) -> float:
        """Analyze motion intensity (0-1)"""
        cmd = [
            'ffmpeg', '-i', path, '-vf',
            'select=\'gt(scene,0.3)\',showinfo',
            '-f', 'null', '-'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            scene_changes = result.stderr.count('pts_time:')
            changes_per_min = (scene_changes / duration) * 60 if duration > 0 else 0
            return min(1.0, changes_per_min / 30)
        except:
            return 0.5

    def _analyze_audio_energy(self, path: str) -> float:
        """Get average audio dB level"""
        cmd = [
            'ffmpeg', '-i', path, '-af', 'volumedetect',
            '-f', 'null', '-'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            for line in result.stderr.split('\n'):
                if 'mean_volume:' in line:
                    return float(line.split('mean_volume:')[1].split('dB')[0].strip())
            return -30.0
        except:
            return -30.0

    def _count_scene_changes(self, path: str) -> int:
        """Count number of scene changes"""
        cmd = [
            'ffmpeg', '-i', path, '-vf',
            'select=\'gt(scene,0.3)\',showinfo',
            '-f', 'null', '-'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.stderr.count('pts_time:')
        except:
            return 0

    def _score_performance(self, path: str, duration: float) -> Tuple[float, List[str]]:
        """
        Score performance/face presence using FFmpeg.
        Returns (score, tags)
        """
        tags = []

        # Sample frames for face detection
        face_scores = []
        sample_count = min(10, max(3, int(duration / 15)))  # 3-10 samples

        for i in range(sample_count):
            t = (duration / (sample_count + 1)) * (i + 1)
            face_present, confidence = self._detect_face_at_time(path, t)
            face_scores.append(confidence if face_present else 0.0)

        avg_face_score = sum(face_scores) / len(face_scores) if face_scores else 0.0

        if avg_face_score > 0.6:
            tags.append(ContentTag.PERFORMANCE.value)
        elif avg_face_score > 0.3:
            tags.append(ContentTag.INTERVIEW.value)
        else:
            tags.append(ContentTag.B_ROLL.value)

        # Center-frame activity analysis
        center_activity = self._analyze_center_activity(path, duration)

        # Combine face presence and center activity
        performance_score = (avg_face_score * 0.7) + (center_activity * 0.3)

        return min(1.0, performance_score), tags

    def _detect_face_at_time(self, path: str, time_sec: float) -> Tuple[bool, float]:
        """
        Detect if face is present at a specific time.
        Returns (face_present, confidence)

        Note: This uses FFmpeg's metadata detection which is basic.
        For better results, opencv-python with haarcascades could be used.
        """
        # Extract frame and check for face-like patterns
        # This is a simplified heuristic - for production, use proper face detection
        cmd = [
            'ffmpeg', '-ss', str(time_sec), '-i', path,
            '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.stdout:
                # Simple skin tone detection heuristic
                pixels = result.stdout
                # Count pixels with skin-like RGB values
                skin_count = 0
                total_checked = min(len(pixels) // 3, 10000)  # Sample up to 10k pixels

                for i in range(0, total_checked * 3, 3):
                    if i + 2 < len(pixels):
                        r, g, b = pixels[i], pixels[i+1], pixels[i+2]
                        # Simplified skin tone check
                        if r > 95 and g > 40 and b > 20:
                            if max(r, g, b) - min(r, g, b) > 15:
                                if abs(r - g) > 15 and r > g and r > b:
                                    skin_count += 1

                skin_ratio = skin_count / total_checked if total_checked > 0 else 0

                # High skin ratio suggests face presence
                if skin_ratio > 0.15:
                    return True, min(1.0, skin_ratio * 4)

                return False, skin_ratio * 2
        except:
            pass

        return False, 0.0

    def _analyze_center_activity(self, path: str, duration: float) -> float:
        """
        Analyze how much activity happens in the center of frame.
        Higher = more performance-focused content.
        """
        # Use FFmpeg crop filter to analyze just the center region
        cmd = [
            'ffmpeg', '-i', path, '-vf',
            'crop=iw/2:ih/2:iw/4:ih/4,select=\'gt(scene,0.2)\',showinfo',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            center_changes = result.stderr.count('pts_time:')

            # Compare to full-frame changes
            full_changes = self._count_scene_changes(path)

            if full_changes > 0:
                # If center has proportionally more changes, activity is centered
                center_ratio = center_changes / full_changes
                return min(1.0, center_ratio * 1.5)

            return 0.5
        except:
            return 0.5

    def _score_with_ai(self, path: str, duration: float) -> Tuple[Optional[float], List[str]]:
        """
        Use vision AI to analyze content.
        Returns (score, tags)
        """
        if not self.ai_enabled or not HAS_OPENAI:
            return None, []

        try:
            client = openai.OpenAI(api_key=EngagementConfig.OPENAI_API_KEY)

            # Extract frames for AI analysis
            frames = self._extract_frames_for_ai(path, duration)

            if not frames:
                return None, []

            # Build message with frames
            content = [
                {
                    "type": "text",
                    "text": """Analyze these video frames for social media engagement potential.

Rate from 1-10:
- Visual interest (composition, colors, action)
- Performance energy (if people visible)
- Shareability (would people want to share this?)

Also identify what type of content this is:
- performance, crowd-shot, interview, b-roll, transition

Respond in JSON format:
{
    "visual_score": 1-10,
    "energy_score": 1-10,
    "shareability_score": 1-10,
    "content_type": "type",
    "brief_description": "one sentence"
}"""
                }
            ]

            # Add frames as images
            for frame_b64 in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "low"  # Use low detail for cost efficiency
                    }
                })

            response = client.chat.completions.create(
                model=EngagementConfig.AI_MODEL,
                messages=[{"role": "user", "content": content}],
                max_tokens=200
            )

            # Parse response
            response_text = response.choices[0].message.content

            # Try to extract JSON
            import re
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Calculate overall AI score
                visual = data.get('visual_score', 5) / 10
                energy = data.get('energy_score', 5) / 10
                share = data.get('shareability_score', 5) / 10

                ai_score = (visual + energy + share) / 3

                tags = []
                content_type = data.get('content_type', '').lower()
                if content_type in [t.value for t in ContentTag]:
                    tags.append(content_type)

                if share > 0.7:
                    tags.append(ContentTag.HIGHLIGHT.value)

                return ai_score, tags

        except Exception as e:
            logging.warning(f"AI scoring failed: {e}")

        return None, []

    def _extract_frames_for_ai(self, path: str, duration: float) -> List[str]:
        """Extract frames as base64 for AI analysis"""
        frames = []
        sample_times = [
            duration * i / (EngagementConfig.AI_SAMPLE_FRAMES + 1)
            for i in range(1, EngagementConfig.AI_SAMPLE_FRAMES + 1)
        ]

        for t in sample_times:
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cmd = [
                        'ffmpeg', '-ss', str(t), '-i', path,
                        '-vframes', '1', '-q:v', '5',
                        '-vf', 'scale=512:-1',  # Resize for efficiency
                        '-y', tmp.name
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=10)

                    if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                        with open(tmp.name, 'rb') as f:
                            frames.append(base64.b64encode(f.read()).decode())

                    os.unlink(tmp.name)
            except:
                pass

        return frames

    def _find_best_moments(self, path: str, duration: float) -> List[Moment]:
        """
        Find the best moments in the video.
        Uses audio peaks and motion spikes.
        """
        moments = []

        if duration < 5:
            # Video too short, just return the whole thing
            return [Moment(
                start_time=0,
                end_time=duration,
                score=1.0,
                reason="Full video",
                tags=[]
            )]

        # Analyze audio levels over time
        audio_peaks = self._find_audio_peaks(path, duration)

        # Analyze motion spikes
        motion_spikes = self._find_motion_spikes(path, duration)

        # Combine and deduplicate
        all_events = audio_peaks + motion_spikes
        all_events.sort(key=lambda x: x['time'])

        # Cluster nearby events into moments
        current_moment_start = None
        current_moment_score = 0
        current_moment_reasons = []

        for event in all_events:
            t = event['time']

            if current_moment_start is None:
                current_moment_start = max(0, t - 1)  # Start 1s before peak
                current_moment_score = event['score']
                current_moment_reasons = [event['reason']]
            elif t - current_moment_start < EngagementConfig.MOMENT_MAX_DURATION:
                # Extend current moment
                current_moment_score = max(current_moment_score, event['score'])
                current_moment_reasons.append(event['reason'])
            else:
                # Save current moment and start new one
                end_time = min(current_moment_start + EngagementConfig.MOMENT_MAX_DURATION,
                              t - 1, duration)

                if end_time - current_moment_start >= EngagementConfig.MOMENT_MIN_DURATION:
                    moments.append(Moment(
                        start_time=current_moment_start,
                        end_time=end_time,
                        score=current_moment_score,
                        reason=', '.join(set(current_moment_reasons)),
                        tags=[ContentTag.HIGHLIGHT.value]
                    ))

                current_moment_start = max(0, t - 1)
                current_moment_score = event['score']
                current_moment_reasons = [event['reason']]

        # Don't forget the last moment
        if current_moment_start is not None:
            end_time = min(current_moment_start + EngagementConfig.MOMENT_MAX_DURATION, duration)
            if end_time - current_moment_start >= EngagementConfig.MOMENT_MIN_DURATION:
                moments.append(Moment(
                    start_time=current_moment_start,
                    end_time=end_time,
                    score=current_moment_score,
                    reason=', '.join(set(current_moment_reasons)),
                    tags=[ContentTag.HIGHLIGHT.value]
                ))

        # Sort by score and take top N
        moments.sort(key=lambda m: m.score, reverse=True)
        return moments[:EngagementConfig.TOP_MOMENTS_COUNT]

    def _find_audio_peaks(self, path: str, duration: float) -> List[Dict]:
        """Find timestamps of audio peaks"""
        peaks = []

        # Use FFmpeg's volumedetect with segment analysis
        segment_duration = 5.0  # Analyze in 5-second chunks

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

                        # High peak = interesting moment
                        if max_vol > -10:
                            peaks.append({
                                'time': start + segment_duration / 2,
                                'score': min(1.0, (max_vol + 20) / 15),
                                'reason': 'audio peak'
                            })
            except:
                pass

        return peaks

    def _find_motion_spikes(self, path: str, duration: float) -> List[Dict]:
        """Find timestamps of high motion"""
        spikes = []

        # Use scene detection to find motion spikes
        cmd = [
            'ffmpeg', '-i', path, '-vf',
            'select=\'gt(scene,0.4)\',showinfo',
            '-f', 'null', '-'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            for line in result.stderr.split('\n'):
                if 'pts_time:' in line:
                    # Extract timestamp
                    import re
                    match = re.search(r'pts_time:(\d+\.?\d*)', line)
                    if match:
                        t = float(match.group(1))
                        spikes.append({
                            'time': t,
                            'score': 0.7,
                            'reason': 'motion spike'
                        })
        except:
            pass

        return spikes

    def _calculate_overall(self, technical: float, performance: float,
                          ai_score: Optional[float], tags: List[str]) -> Tuple[float, float]:
        """
        Calculate overall score and confidence.
        Returns (overall_score, confidence)
        """
        scores = [
            (technical, EngagementConfig.TECHNICAL_WEIGHT),
            (performance, EngagementConfig.PERFORMANCE_WEIGHT),
        ]

        if ai_score is not None:
            scores.append((ai_score, EngagementConfig.AI_WEIGHT))
        else:
            # Redistribute AI weight
            total_weight = EngagementConfig.TECHNICAL_WEIGHT + EngagementConfig.PERFORMANCE_WEIGHT
            scores = [
                (technical, EngagementConfig.TECHNICAL_WEIGHT / total_weight),
                (performance, EngagementConfig.PERFORMANCE_WEIGHT / total_weight),
            ]

        overall = sum(score * weight for score, weight in scores)

        # Confidence based on:
        # - Consistency between scores
        # - Number of tags (more = more certain about classification)
        # - Whether AI was available

        score_variance = abs(technical - performance)
        consistency = 1.0 - (score_variance * 0.5)

        tag_confidence = min(1.0, len(tags) * 0.15)  # More tags = more confident

        ai_bonus = 0.1 if ai_score is not None else 0

        confidence = (consistency * 0.5) + (tag_confidence * 0.3) + (0.1 + ai_bonus)

        return round(overall, 3), round(min(1.0, confidence), 3)


# ============================================================
# MAIN (for testing)
# ============================================================

def main():
    """Test engagement scoring"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python engagement_scorer.py <video_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"File not found: {video_path}")
        sys.exit(1)

    print(f"\n🦭 Analyzing engagement: {video_path.name}")
    print("=" * 60)

    scorer = EngagementScorer()
    result = scorer.score(video_path)

    print(f"\n📊 Results:")
    print(f"   Technical Score:   {result.technical_score:.2f}")
    print(f"   Performance Score: {result.performance_score:.2f}")
    if result.ai_score is not None:
        print(f"   AI Score:          {result.ai_score:.2f}")
    print(f"   ─────────────────────")
    print(f"   Overall Score:     {result.overall_score:.2f}")
    print(f"   Confidence:        {result.confidence:.2f}")

    print(f"\n🏷️  Tags: {', '.join(result.tags)}")

    if result.best_moments:
        print(f"\n⭐ Best Moments:")
        for i, m in enumerate(result.best_moments[:5], 1):
            print(f"   {i}. {m.start_time:.1f}s - {m.end_time:.1f}s (score: {m.score:.2f}) - {m.reason}")

    print(f"\n⏱️  Analysis time: {result.analysis_time:.1f}s")


if __name__ == '__main__':
    main()
