"""
Video Search

Search videos by:
- Tags and categories
- Content description
- Filename
- Duration/resolution
- Date range
- Full-text search

Combines tag database with file metadata for comprehensive search.
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


@dataclass
class SearchResult:
    """A search result"""
    path: Path
    filename: str
    score: float           # Relevance score 0-1
    matched_tags: List[str]
    matched_text: List[str]
    description: str
    thumbnail: Optional[Path]
    duration: Optional[float]
    resolution: Optional[str]
    created_at: Optional[str]


@dataclass
class SearchQuery:
    """A search query with filters"""
    text: Optional[str] = None          # Free text search
    tags: Optional[List[str]] = None    # Required tags
    exclude_tags: Optional[List[str]] = None  # Tags to exclude
    category: Optional[str] = None      # Tag category filter
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    resolution: Optional[str] = None    # '4k', '1080p', '720p'
    aspect_ratio: Optional[str] = None  # 'landscape', 'portrait', 'square'
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 50


class VideoSearch:
    """
    Search engine for video library.

    Indexes video metadata and tags for fast searching.
    """

    def __init__(self, db_path: Path, library_dir: Path):
        self.db_path = db_path
        self.library_dir = library_dir
        self._init_db()

    def _init_db(self):
        """Initialize search database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Video index
            conn.execute('''
                CREATE TABLE IF NOT EXISTS video_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    description TEXT,
                    duration REAL,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    codec TEXT,
                    created_at TEXT,
                    indexed_at TEXT,
                    thumbnail_path TEXT
                )
            ''')

            # Tags (linked to videos)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS search_tags (
                    video_id INTEGER NOT NULL,
                    tag TEXT NOT NULL,
                    category TEXT,
                    confidence REAL,
                    FOREIGN KEY (video_id) REFERENCES video_index(id),
                    UNIQUE(video_id, tag)
                )
            ''')

            # Full-text search
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS video_fts USING fts5(
                    filename, description, tags,
                    content='video_index',
                    content_rowid='id'
                )
            ''')

            # Indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_search_tags ON search_tags(tag)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_video_duration ON video_index(duration)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_video_created ON video_index(created_at)')

            conn.commit()

    def index_video(
        self,
        video_path: Path,
        description: str = "",
        tags: List[str] = None,
        thumbnail_path: Path = None
    ) -> int:
        """
        Index a video for searching.

        Args:
            video_path: Path to video
            description: Text description
            tags: List of tags
            thumbnail_path: Path to thumbnail

        Returns:
            Video ID
        """
        tags = tags or []

        # Get video metadata
        metadata = self._get_video_metadata(video_path)

        with sqlite3.connect(self.db_path) as conn:
            # Insert/update video
            cursor = conn.execute('''
                INSERT OR REPLACE INTO video_index
                (path, filename, description, duration, width, height, fps, codec, created_at, indexed_at, thumbnail_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(video_path),
                video_path.name,
                description,
                metadata.get('duration'),
                metadata.get('width'),
                metadata.get('height'),
                metadata.get('fps'),
                metadata.get('codec'),
                metadata.get('created_at'),
                datetime.now().isoformat(),
                str(thumbnail_path) if thumbnail_path else None
            ))

            video_id = cursor.lastrowid

            # Clear old tags
            conn.execute('DELETE FROM search_tags WHERE video_id = ?', (video_id,))

            # Insert tags
            for tag in tags:
                conn.execute('''
                    INSERT INTO search_tags (video_id, tag, category, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (video_id, tag.lower(), None, 1.0))

            # Update FTS
            tags_str = ' '.join(tags)
            conn.execute('''
                INSERT OR REPLACE INTO video_fts (rowid, filename, description, tags)
                VALUES (?, ?, ?, ?)
            ''', (video_id, video_path.name, description, tags_str))

            conn.commit()
            return video_id

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Search for videos matching query.

        Args:
            query: SearchQuery with filters

        Returns:
            List of SearchResult sorted by relevance
        """
        results = []
        seen_paths = set()

        with sqlite3.connect(self.db_path) as conn:
            # Build query parts
            conditions = []
            params = []

            # Text search using FTS
            if query.text:
                # First get FTS matches
                fts_cursor = conn.execute('''
                    SELECT rowid, rank FROM video_fts
                    WHERE video_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ''', (query.text, query.limit))

                fts_ids = {row[0]: row[1] for row in fts_cursor.fetchall()}

                if fts_ids:
                    conditions.append(f"v.id IN ({','.join('?' * len(fts_ids))})")
                    params.extend(fts_ids.keys())

            # Tag filters
            if query.tags:
                tag_placeholders = ','.join('?' * len(query.tags))
                conditions.append(f'''
                    v.id IN (
                        SELECT video_id FROM search_tags
                        WHERE tag IN ({tag_placeholders})
                        GROUP BY video_id
                        HAVING COUNT(DISTINCT tag) = ?
                    )
                ''')
                params.extend([t.lower() for t in query.tags])
                params.append(len(query.tags))

            # Exclude tags
            if query.exclude_tags:
                tag_placeholders = ','.join('?' * len(query.exclude_tags))
                conditions.append(f'''
                    v.id NOT IN (
                        SELECT video_id FROM search_tags WHERE tag IN ({tag_placeholders})
                    )
                ''')
                params.extend([t.lower() for t in query.exclude_tags])

            # Duration filters
            if query.min_duration:
                conditions.append('v.duration >= ?')
                params.append(query.min_duration)

            if query.max_duration:
                conditions.append('v.duration <= ?')
                params.append(query.max_duration)

            # Resolution filter
            if query.resolution:
                res_map = {'4k': 3840, '1080p': 1920, '720p': 1280}
                min_width = res_map.get(query.resolution, 0)
                if min_width:
                    conditions.append('v.width >= ?')
                    params.append(min_width)

            # Aspect ratio filter
            if query.aspect_ratio:
                if query.aspect_ratio == 'landscape':
                    conditions.append('v.width > v.height')
                elif query.aspect_ratio == 'portrait':
                    conditions.append('v.height > v.width')
                elif query.aspect_ratio == 'square':
                    conditions.append('ABS(v.width - v.height) < v.width * 0.1')

            # Date filters
            if query.date_from:
                conditions.append('v.created_at >= ?')
                params.append(query.date_from.isoformat())

            if query.date_to:
                conditions.append('v.created_at <= ?')
                params.append(query.date_to.isoformat())

            # Build and execute query
            where_clause = ' AND '.join(conditions) if conditions else '1=1'

            sql = f'''
                SELECT v.id, v.path, v.filename, v.description, v.duration,
                       v.width, v.height, v.created_at, v.thumbnail_path
                FROM video_index v
                WHERE {where_clause}
                LIMIT ?
            '''
            params.append(query.limit)

            cursor = conn.execute(sql, params)

            for row in cursor.fetchall():
                video_id = row[0]
                path = Path(row[1])

                if str(path) in seen_paths:
                    continue
                seen_paths.add(str(path))

                # Get tags for this video
                tag_cursor = conn.execute(
                    'SELECT tag FROM search_tags WHERE video_id = ?',
                    (video_id,)
                )
                video_tags = [r[0] for r in tag_cursor.fetchall()]

                # Calculate relevance score
                score = self._calculate_score(query, row, video_tags)

                # Build resolution string
                width, height = row[5], row[6]
                resolution = None
                if width and height:
                    if width >= 3840:
                        resolution = '4K'
                    elif width >= 1920:
                        resolution = '1080p'
                    elif width >= 1280:
                        resolution = '720p'
                    else:
                        resolution = f'{width}x{height}'

                results.append(SearchResult(
                    path=path,
                    filename=row[2],
                    score=score,
                    matched_tags=video_tags,
                    matched_text=[],
                    description=row[3] or '',
                    thumbnail=Path(row[8]) if row[8] else None,
                    duration=row[4],
                    resolution=resolution,
                    created_at=row[7]
                ))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:query.limit]

    def _calculate_score(self, query: SearchQuery, row: tuple, tags: List[str]) -> float:
        """Calculate relevance score for a result"""
        score = 0.5  # Base score

        # Text match boost
        if query.text:
            text_lower = query.text.lower()
            if text_lower in row[2].lower():  # Filename match
                score += 0.3
            if row[3] and text_lower in row[3].lower():  # Description match
                score += 0.2

        # Tag match boost
        if query.tags:
            matched = sum(1 for t in query.tags if t.lower() in tags)
            score += 0.1 * matched

        # Recency boost
        if row[7]:  # created_at
            try:
                created = datetime.fromisoformat(row[7])
                days_old = (datetime.now() - created).days
                if days_old < 7:
                    score += 0.1
                elif days_old < 30:
                    score += 0.05
            except:
                pass

        return min(1.0, score)

    def _get_video_metadata(self, video_path: Path) -> Dict:
        """Get video metadata using ffprobe"""
        import subprocess

        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)

            video_stream = next(
                (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
                {}
            )
            format_info = data.get('format', {})

            # Get file creation time
            stat = video_path.stat()
            created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()

            return {
                'duration': float(format_info.get('duration', 0)),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'created_at': created_at
            }

        except Exception as e:
            logging.warning(f"Failed to get metadata for {video_path}: {e}")
            return {}

    def reindex_library(self) -> int:
        """Reindex all videos in library directory"""
        count = 0
        extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

        for video_path in self.library_dir.rglob('*'):
            if video_path.suffix.lower() in extensions:
                try:
                    self.index_video(video_path)
                    count += 1
                except Exception as e:
                    logging.warning(f"Failed to index {video_path}: {e}")

        logging.info(f"Indexed {count} videos")
        return count

    def get_all_tags(self) -> Dict[str, int]:
        """Get all unique tags with video counts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT tag, COUNT(DISTINCT video_id) as count
                FROM search_tags
                GROUP BY tag
                ORDER BY count DESC
            ''')

            return {row[0]: row[1] for row in cursor.fetchall()}

    def get_stats(self) -> Dict:
        """Get search index statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM video_index')
            total_videos = cursor.fetchone()[0]

            cursor = conn.execute('SELECT COUNT(DISTINCT tag) FROM search_tags')
            unique_tags = cursor.fetchone()[0]

            cursor = conn.execute('SELECT SUM(duration) FROM video_index')
            total_duration = cursor.fetchone()[0] or 0

            return {
                'total_videos': total_videos,
                'unique_tags': unique_tags,
                'total_duration_hours': total_duration / 3600,
            }
