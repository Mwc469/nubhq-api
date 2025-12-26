"""
Proxy Workflow

Fast previews with full quality on approval:
1. Generate low-res proxy quickly for preview
2. User reviews and approves proxy
3. Generate full quality only for approved videos

Saves processing time by not encoding rejected videos at full quality.
"""

import os
import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import sqlite3


@dataclass
class ProxyVideo:
    """A proxy video with metadata"""
    proxy_path: Path
    source_path: Path
    status: str  # 'pending', 'approved', 'rejected', 'processing', 'done'
    created_at: str
    reviewed_at: Optional[str]
    full_quality_path: Optional[Path]


class ProxyWorkflow:
    """
    Proxy-based video workflow.

    Fast previews first, full quality only on approval.
    """

    # Proxy settings (fast, low quality)
    PROXY_CRF = 35          # Very compressed
    PROXY_SCALE = 480       # Low resolution
    PROXY_PRESET = 'ultrafast'

    def __init__(self, proxy_dir: Path, db_path: Path):
        self.proxy_dir = proxy_dir
        self.db_path = db_path

        self.proxy_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize proxy tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS proxies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT UNIQUE NOT NULL,
                    proxy_path TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    profile TEXT,
                    settings TEXT,
                    created_at TEXT,
                    reviewed_at TEXT,
                    full_quality_path TEXT,
                    notes TEXT
                )
            ''')
            conn.commit()

    def generate_proxy(
        self,
        source_path: Path,
        profile: str = None,
        settings: Dict = None
    ) -> Optional[ProxyVideo]:
        """
        Generate fast proxy for preview.

        Args:
            source_path: Original video
            profile: Processing profile to use for full quality
            settings: Additional settings

        Returns:
            ProxyVideo or None if failed
        """
        proxy_name = f"{source_path.stem}_proxy.mp4"
        proxy_path = self.proxy_dir / proxy_name

        logging.info(f"Generating proxy for: {source_path.name}")

        # Build fast encode command
        cmd = [
            'ffmpeg', '-y', '-i', str(source_path),
            '-vf', f'scale={self.PROXY_SCALE}:-2',
            '-c:v', 'libx264',
            '-crf', str(self.PROXY_CRF),
            '-preset', self.PROXY_PRESET,
            '-c:a', 'aac', '-b:a', '64k',
            str(proxy_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)

            if result.returncode == 0 and proxy_path.exists():
                # Record in database
                created_at = datetime.now().isoformat()

                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO proxies
                        (source_path, proxy_path, status, profile, settings, created_at)
                        VALUES (?, ?, 'pending', ?, ?, ?)
                    ''', (
                        str(source_path),
                        str(proxy_path),
                        profile,
                        json.dumps(settings) if settings else None,
                        created_at
                    ))
                    conn.commit()

                logging.info(f"Proxy generated: {proxy_path.name}")

                return ProxyVideo(
                    proxy_path=proxy_path,
                    source_path=source_path,
                    status='pending',
                    created_at=created_at,
                    reviewed_at=None,
                    full_quality_path=None
                )
            else:
                logging.error(f"Proxy generation failed: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            logging.error("Proxy generation timed out")
        except Exception as e:
            logging.error(f"Proxy generation error: {e}")

        return None

    def approve(self, source_path: Path, notes: str = None) -> bool:
        """Mark a proxy as approved for full-quality processing"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE proxies
                SET status = 'approved', reviewed_at = ?, notes = ?
                WHERE source_path = ?
            ''', (datetime.now().isoformat(), notes, str(source_path)))
            conn.commit()

            return conn.total_changes > 0

    def reject(self, source_path: Path, notes: str = None) -> bool:
        """Mark a proxy as rejected"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE proxies
                SET status = 'rejected', reviewed_at = ?, notes = ?
                WHERE source_path = ?
            ''', (datetime.now().isoformat(), notes, str(source_path)))
            conn.commit()

            return conn.total_changes > 0

    def get_pending(self) -> List[ProxyVideo]:
        """Get all pending proxies for review"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT source_path, proxy_path, status, created_at, reviewed_at, full_quality_path
                FROM proxies
                WHERE status = 'pending'
                ORDER BY created_at ASC
            ''')

            return [
                ProxyVideo(
                    source_path=Path(row[0]),
                    proxy_path=Path(row[1]),
                    status=row[2],
                    created_at=row[3],
                    reviewed_at=row[4],
                    full_quality_path=Path(row[5]) if row[5] else None
                )
                for row in cursor.fetchall()
            ]

    def get_approved(self) -> List[ProxyVideo]:
        """Get all approved proxies ready for full-quality processing"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT source_path, proxy_path, status, created_at, reviewed_at, full_quality_path
                FROM proxies
                WHERE status = 'approved'
                ORDER BY reviewed_at ASC
            ''')

            return [
                ProxyVideo(
                    source_path=Path(row[0]),
                    proxy_path=Path(row[1]),
                    status=row[2],
                    created_at=row[3],
                    reviewed_at=row[4],
                    full_quality_path=Path(row[5]) if row[5] else None
                )
                for row in cursor.fetchall()
            ]

    def mark_processing(self, source_path: Path) -> bool:
        """Mark proxy as being processed to full quality"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE proxies SET status = 'processing' WHERE source_path = ?
            ''', (str(source_path),))
            conn.commit()
            return conn.total_changes > 0

    def mark_done(self, source_path: Path, full_quality_path: Path) -> bool:
        """Mark proxy as done with full quality path"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE proxies
                SET status = 'done', full_quality_path = ?
                WHERE source_path = ?
            ''', (str(full_quality_path), str(source_path)))
            conn.commit()
            return conn.total_changes > 0

    def get_profile(self, source_path: Path) -> Optional[str]:
        """Get the profile for a source video"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT profile FROM proxies WHERE source_path = ?',
                (str(source_path),)
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def get_settings(self, source_path: Path) -> Optional[Dict]:
        """Get settings for a source video"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT settings FROM proxies WHERE source_path = ?',
                (str(source_path),)
            )
            row = cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
        return None

    def cleanup_rejected(self, delete_sources: bool = False) -> int:
        """Clean up rejected proxies"""
        count = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT source_path, proxy_path FROM proxies WHERE status = ?',
                ('rejected',)
            )

            for row in cursor.fetchall():
                proxy_path = Path(row[1])
                source_path = Path(row[0])

                # Delete proxy
                if proxy_path.exists():
                    proxy_path.unlink()
                    count += 1

                # Optionally delete source
                if delete_sources and source_path.exists():
                    source_path.unlink()

            # Remove from database
            conn.execute('DELETE FROM proxies WHERE status = ?', ('rejected',))
            conn.commit()

        return count

    def get_stats(self) -> Dict:
        """Get proxy workflow statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT status, COUNT(*) FROM proxies GROUP BY status
            ''')

            stats = {'pending': 0, 'approved': 0, 'rejected': 0, 'processing': 0, 'done': 0}
            for row in cursor.fetchall():
                stats[row[0]] = row[1]

            stats['total'] = sum(stats.values())
            stats['approval_rate'] = (
                stats['approved'] + stats['done']
            ) / max(1, stats['approved'] + stats['done'] + stats['rejected'])

            return stats
