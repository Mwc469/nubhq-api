"""
AI Video Tagging

Automatically tag videos with:
- Scene/object classification
- Activity detection
- Mood/tone analysis
- Color palette extraction
- Text/logo detection

Uses OpenAI Vision API or local models.
"""

import os
import json
import logging
import subprocess
import tempfile
import base64
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


class TagCategory(Enum):
    """Categories of tags"""
    SCENE = "scene"           # indoor, outdoor, beach, city, etc.
    ACTIVITY = "activity"     # dancing, talking, sports, cooking, etc.
    OBJECTS = "objects"       # car, dog, food, instrument, etc.
    PEOPLE = "people"         # solo, group, crowd, interview, etc.
    MOOD = "mood"             # energetic, calm, dramatic, funny, etc.
    STYLE = "style"           # cinematic, vlog, tutorial, music video, etc.
    COLORS = "colors"         # dominant color palette
    TEXT = "text"             # detected text/logos


@dataclass
class VideoTag:
    """A single tag with metadata"""
    name: str
    category: TagCategory
    confidence: float  # 0-1
    timestamp: Optional[float] = None  # Specific moment (None = whole video)


@dataclass
class TaggingResult:
    """Complete tagging result"""
    video_path: Path
    tags: List[VideoTag]
    description: str
    dominant_colors: List[str]
    detected_text: List[str]
    suggested_title: Optional[str]
    suggested_hashtags: List[str]


class AITagger:
    """AI-powered video tagging"""

    def __init__(self):
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.model = os.environ.get('NUBHQ_AI_MODEL', 'gpt-4o-mini')

    def is_available(self) -> bool:
        """Check if AI tagging is available"""
        return bool(self.api_key)

    def tag_video(self, video_path: Path, sample_count: int = 5) -> TaggingResult:
        """
        Analyze video and generate tags.

        Args:
            video_path: Path to video
            sample_count: Number of frames to analyze

        Returns:
            TaggingResult with all detected tags
        """
        if not self.is_available():
            return self._fallback_tagging(video_path)

        try:
            # Extract sample frames
            frames = self._extract_frames(video_path, sample_count)

            if not frames:
                logging.warning("No frames extracted, using fallback tagging")
                return self._fallback_tagging(video_path)

            # Analyze with AI
            analysis = self._analyze_with_ai(frames, video_path.name)

            # Parse results
            return self._parse_ai_response(video_path, analysis)

        except Exception as e:
            logging.exception(f"AI tagging failed: {e}")
            return self._fallback_tagging(video_path)

    def _extract_frames(self, video_path: Path, count: int) -> List[str]:
        """Extract frames as base64 for AI analysis"""
        duration = self._get_duration(video_path)
        if duration <= 0:
            return []

        frames = []
        times = [duration * (i + 1) / (count + 1) for i in range(count)]

        for t in times:
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cmd = [
                        'ffmpeg', '-ss', str(t), '-i', str(video_path),
                        '-vframes', '1', '-q:v', '3',
                        '-vf', 'scale=768:-1',  # Resize for efficiency
                        '-y', tmp.name
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=10)

                    if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
                        with open(tmp.name, 'rb') as f:
                            frames.append(base64.b64encode(f.read()).decode())

                    os.unlink(tmp.name)
            except Exception as e:
                logging.debug(f"Frame extraction failed at {t}: {e}")

        return frames

    def _analyze_with_ai(self, frames: List[str], filename: str) -> dict:
        """Send frames to OpenAI Vision API"""
        import openai

        client = openai.OpenAI(api_key=self.api_key)

        # Build message with frames
        content = [
            {
                "type": "text",
                "text": f"""Analyze these video frames from "{filename}" and provide:

1. Scene tags (indoor/outdoor, location type)
2. Activity tags (what's happening)
3. Object tags (notable items visible)
4. People tags (solo/group/crowd, demographics hints)
5. Mood tags (energy level, tone)
6. Style tags (video type/genre)
7. Dominant colors (hex codes)
8. Any visible text or logos
9. A one-sentence description
10. Suggested title (catchy, YouTube-style)
11. Suggested hashtags (10-15)

Respond in JSON format:
{{
    "scene": ["tag1", "tag2"],
    "activity": ["tag1", "tag2"],
    "objects": ["tag1", "tag2"],
    "people": ["tag1", "tag2"],
    "mood": ["tag1", "tag2"],
    "style": ["tag1", "tag2"],
    "colors": ["#hex1", "#hex2", "#hex3"],
    "text": ["detected text"],
    "description": "One sentence description",
    "title": "Suggested Title",
    "hashtags": ["#tag1", "#tag2"]
}}"""
            }
        ]

        # Add frames
        for frame_b64 in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low"
                }
            })

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000
        )

        # Parse JSON from response
        response_text = response.choices[0].message.content

        # Extract JSON
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())

        return {}

    def _parse_ai_response(self, video_path: Path, analysis: dict) -> TaggingResult:
        """Parse AI response into TaggingResult"""
        tags = []

        # Parse each category
        category_map = {
            'scene': TagCategory.SCENE,
            'activity': TagCategory.ACTIVITY,
            'objects': TagCategory.OBJECTS,
            'people': TagCategory.PEOPLE,
            'mood': TagCategory.MOOD,
            'style': TagCategory.STYLE,
        }

        for key, category in category_map.items():
            for tag_name in analysis.get(key, []):
                tags.append(VideoTag(
                    name=tag_name.lower().strip(),
                    category=category,
                    confidence=0.8  # AI typically high confidence
                ))

        return TaggingResult(
            video_path=video_path,
            tags=tags,
            description=analysis.get('description', ''),
            dominant_colors=analysis.get('colors', []),
            detected_text=analysis.get('text', []),
            suggested_title=analysis.get('title'),
            suggested_hashtags=analysis.get('hashtags', [])
        )

    def _fallback_tagging(self, video_path: Path) -> TaggingResult:
        """Basic tagging without AI"""
        tags = []

        # Analyze duration
        duration = self._get_duration(video_path)
        if duration < 60:
            tags.append(VideoTag("short-form", TagCategory.STYLE, 0.9))
        elif duration < 300:
            tags.append(VideoTag("medium-length", TagCategory.STYLE, 0.9))
        else:
            tags.append(VideoTag("long-form", TagCategory.STYLE, 0.9))

        # Analyze resolution
        width, height = self._get_dimensions(video_path)
        if width > height:
            tags.append(VideoTag("landscape", TagCategory.STYLE, 1.0))
        elif height > width:
            tags.append(VideoTag("vertical", TagCategory.STYLE, 1.0))
        else:
            tags.append(VideoTag("square", TagCategory.STYLE, 1.0))

        if width >= 3840:
            tags.append(VideoTag("4k", TagCategory.STYLE, 1.0))
        elif width >= 1920:
            tags.append(VideoTag("hd", TagCategory.STYLE, 1.0))

        # Analyze audio
        has_audio = self._has_audio(video_path)
        if has_audio:
            tags.append(VideoTag("has-audio", TagCategory.STYLE, 1.0))

        return TaggingResult(
            video_path=video_path,
            tags=tags,
            description="",
            dominant_colors=[],
            detected_text=[],
            suggested_title=None,
            suggested_hashtags=[]
        )

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration"""
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return float(data.get('format', {}).get('duration', 0))
        except:
            return 0

    def _get_dimensions(self, video_path: Path) -> tuple:
        """Get video dimensions"""
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream['codec_type'] == 'video':
                    return int(stream['width']), int(stream['height'])
        except:
            pass
        return 1920, 1080

    def _has_audio(self, video_path: Path) -> bool:
        """Check if video has audio"""
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(video_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)
            return any(s['codec_type'] == 'audio' for s in data.get('streams', []))
        except:
            return False


# ============================================================
# TAG DATABASE
# ============================================================

class TagDatabase:
    """Store and search video tags"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize tag database"""
        import sqlite3

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    description TEXT,
                    title_suggestion TEXT,
                    tagged_at TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL,
                    FOREIGN KEY (video_id) REFERENCES videos(id),
                    UNIQUE(video_id, name, category)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS colors (
                    video_id INTEGER NOT NULL,
                    color TEXT NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos(id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS hashtags (
                    video_id INTEGER NOT NULL,
                    hashtag TEXT NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES videos(id)
                )
            ''')

            conn.execute('CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category)')

            conn.commit()

    def save(self, result: TaggingResult) -> int:
        """Save tagging result to database"""
        import sqlite3
        from datetime import datetime

        with sqlite3.connect(self.db_path) as conn:
            # Insert or update video
            cursor = conn.execute('''
                INSERT OR REPLACE INTO videos (path, filename, description, title_suggestion, tagged_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                str(result.video_path),
                result.video_path.name,
                result.description,
                result.suggested_title,
                datetime.now().isoformat()
            ))

            video_id = cursor.lastrowid

            # Clear existing tags
            conn.execute('DELETE FROM tags WHERE video_id = ?', (video_id,))
            conn.execute('DELETE FROM colors WHERE video_id = ?', (video_id,))
            conn.execute('DELETE FROM hashtags WHERE video_id = ?', (video_id,))

            # Insert tags
            for tag in result.tags:
                conn.execute('''
                    INSERT INTO tags (video_id, name, category, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (video_id, tag.name, tag.category.value, tag.confidence))

            # Insert colors
            for color in result.dominant_colors:
                conn.execute('INSERT INTO colors (video_id, color) VALUES (?, ?)', (video_id, color))

            # Insert hashtags
            for hashtag in result.suggested_hashtags:
                conn.execute('INSERT INTO hashtags (video_id, hashtag) VALUES (?, ?)', (video_id, hashtag))

            conn.commit()
            return video_id

    def search(self, query: str, category: Optional[TagCategory] = None, limit: int = 50) -> List[dict]:
        """Search videos by tag"""
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            if category:
                cursor = conn.execute('''
                    SELECT DISTINCT v.path, v.filename, v.description, v.title_suggestion
                    FROM videos v
                    JOIN tags t ON v.id = t.video_id
                    WHERE t.name LIKE ? AND t.category = ?
                    LIMIT ?
                ''', (f'%{query}%', category.value, limit))
            else:
                cursor = conn.execute('''
                    SELECT DISTINCT v.path, v.filename, v.description, v.title_suggestion
                    FROM videos v
                    JOIN tags t ON v.id = t.video_id
                    WHERE t.name LIKE ? OR v.description LIKE ?
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'path': row[0],
                    'filename': row[1],
                    'description': row[2],
                    'title': row[3]
                })

            return results

    def get_tags_for_video(self, video_path: Path) -> List[VideoTag]:
        """Get all tags for a video"""
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT t.name, t.category, t.confidence
                FROM tags t
                JOIN videos v ON t.video_id = v.id
                WHERE v.path = ?
            ''', (str(video_path),))

            tags = []
            for row in cursor.fetchall():
                tags.append(VideoTag(
                    name=row[0],
                    category=TagCategory(row[1]),
                    confidence=row[2]
                ))

            return tags

    def get_all_tags(self) -> Dict[str, int]:
        """Get all unique tags with counts"""
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT name, COUNT(*) as count
                FROM tags
                GROUP BY name
                ORDER BY count DESC
            ''')

            return {row[0]: row[1] for row in cursor.fetchall()}
