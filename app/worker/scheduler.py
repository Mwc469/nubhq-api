"""
Batch Processing Scheduler

Schedule video processing for specific times:
- Process overnight when not using computer
- Queue videos for batch processing
- Priority-based scheduling
- Time-window processing

Uses macOS launchd or cron for scheduling.
"""

import os
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import sqlite3


class ScheduleType(Enum):
    """Types of schedules"""
    IMMEDIATE = "immediate"      # Process now
    OVERNIGHT = "overnight"       # Process between 12am-6am
    IDLE = "idle"                 # Process when system is idle
    SPECIFIC = "specific"         # Process at specific time
    WEEKENDS = "weekends"         # Process only on weekends


@dataclass
class ScheduledJob:
    """A scheduled processing job"""
    id: int
    video_path: Path
    profile: str
    schedule_type: ScheduleType
    scheduled_time: Optional[datetime]
    priority: int
    status: str  # 'scheduled', 'processing', 'completed', 'failed'
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


class ProcessingScheduler:
    """
    Schedule batch video processing.

    Supports time-based scheduling and priority queues.
    """

    # Default overnight window
    OVERNIGHT_START = dt_time(0, 0)   # Midnight
    OVERNIGHT_END = dt_time(6, 0)     # 6 AM

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize scheduler database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_path TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    schedule_type TEXT NOT NULL,
                    scheduled_time TEXT,
                    priority INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'scheduled',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    settings TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS schedule_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            ''')

            conn.commit()

    def schedule(
        self,
        video_path: Path,
        profile: str,
        schedule_type: ScheduleType = ScheduleType.IMMEDIATE,
        scheduled_time: datetime = None,
        priority: int = 0,
        settings: Dict = None
    ) -> int:
        """
        Schedule a video for processing.

        Args:
            video_path: Video to process
            profile: Processing profile
            schedule_type: When to process
            scheduled_time: Specific time (for SPECIFIC type)
            priority: Higher = process first (0-10)
            settings: Additional settings

        Returns:
            Job ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO scheduled_jobs
                (video_path, profile, schedule_type, scheduled_time, priority, status, created_at, settings)
                VALUES (?, ?, ?, ?, ?, 'scheduled', ?, ?)
            ''', (
                str(video_path),
                profile,
                schedule_type.value,
                scheduled_time.isoformat() if scheduled_time else None,
                priority,
                datetime.now().isoformat(),
                json.dumps(settings) if settings else None
            ))
            conn.commit()

            job_id = cursor.lastrowid
            logging.info(f"Scheduled job {job_id}: {video_path.name} ({schedule_type.value})")

            return job_id

    def schedule_batch(
        self,
        video_paths: List[Path],
        profile: str,
        schedule_type: ScheduleType = ScheduleType.OVERNIGHT
    ) -> List[int]:
        """Schedule multiple videos for batch processing"""
        job_ids = []
        for i, path in enumerate(video_paths):
            job_id = self.schedule(
                path,
                profile,
                schedule_type,
                priority=len(video_paths) - i  # Earlier in list = higher priority
            )
            job_ids.append(job_id)
        return job_ids

    def get_ready_jobs(self) -> List[ScheduledJob]:
        """Get jobs ready to process now"""
        now = datetime.now()
        current_time = now.time()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, video_path, profile, schedule_type, scheduled_time,
                       priority, status, created_at, started_at, completed_at, error
                FROM scheduled_jobs
                WHERE status = 'scheduled'
                ORDER BY priority DESC, created_at ASC
            ''')

            ready = []
            for row in cursor.fetchall():
                job = ScheduledJob(
                    id=row[0],
                    video_path=Path(row[1]),
                    profile=row[2],
                    schedule_type=ScheduleType(row[3]),
                    scheduled_time=datetime.fromisoformat(row[4]) if row[4] else None,
                    priority=row[5],
                    status=row[6],
                    created_at=row[7],
                    started_at=row[8],
                    completed_at=row[9],
                    error=row[10]
                )

                # Check if job is ready
                if self._is_ready(job, now, current_time):
                    ready.append(job)

            return ready

    def _is_ready(self, job: ScheduledJob, now: datetime, current_time: dt_time) -> bool:
        """Check if a job is ready to run"""
        if job.schedule_type == ScheduleType.IMMEDIATE:
            return True

        elif job.schedule_type == ScheduleType.OVERNIGHT:
            return self.OVERNIGHT_START <= current_time <= self.OVERNIGHT_END

        elif job.schedule_type == ScheduleType.SPECIFIC:
            if job.scheduled_time:
                return now >= job.scheduled_time
            return False

        elif job.schedule_type == ScheduleType.WEEKENDS:
            return now.weekday() >= 5  # Saturday = 5, Sunday = 6

        elif job.schedule_type == ScheduleType.IDLE:
            return self._is_system_idle()

        return False

    def _is_system_idle(self) -> bool:
        """Check if system is idle (macOS)"""
        try:
            # Get idle time using ioreg
            cmd = ['ioreg', '-c', 'IOHIDSystem']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            # Parse HIDIdleTime (in nanoseconds)
            for line in result.stdout.split('\n'):
                if 'HIDIdleTime' in line:
                    import re
                    match = re.search(r'= (\d+)', line)
                    if match:
                        idle_ns = int(match.group(1))
                        idle_seconds = idle_ns / 1_000_000_000

                        # Consider idle if > 5 minutes
                        return idle_seconds > 300

        except Exception as e:
            logging.debug(f"Idle check failed: {e}")

        return False

    def mark_started(self, job_id: int) -> bool:
        """Mark job as started"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE scheduled_jobs
                SET status = 'processing', started_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), job_id))
            conn.commit()
            return conn.total_changes > 0

    def mark_completed(self, job_id: int) -> bool:
        """Mark job as completed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE scheduled_jobs
                SET status = 'completed', completed_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), job_id))
            conn.commit()
            return conn.total_changes > 0

    def mark_failed(self, job_id: int, error: str) -> bool:
        """Mark job as failed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE scheduled_jobs
                SET status = 'failed', completed_at = ?, error = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), error, job_id))
            conn.commit()
            return conn.total_changes > 0

    def cancel(self, job_id: int) -> bool:
        """Cancel a scheduled job"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'DELETE FROM scheduled_jobs WHERE id = ? AND status = ?',
                (job_id, 'scheduled')
            )
            conn.commit()
            return conn.total_changes > 0

    def get_queue(self) -> List[ScheduledJob]:
        """Get all scheduled jobs"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, video_path, profile, schedule_type, scheduled_time,
                       priority, status, created_at, started_at, completed_at, error
                FROM scheduled_jobs
                WHERE status IN ('scheduled', 'processing')
                ORDER BY priority DESC, created_at ASC
            ''')

            return [
                ScheduledJob(
                    id=row[0],
                    video_path=Path(row[1]),
                    profile=row[2],
                    schedule_type=ScheduleType(row[3]),
                    scheduled_time=datetime.fromisoformat(row[4]) if row[4] else None,
                    priority=row[5],
                    status=row[6],
                    created_at=row[7],
                    started_at=row[8],
                    completed_at=row[9],
                    error=row[10]
                )
                for row in cursor.fetchall()
            ]

    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT status, COUNT(*) FROM scheduled_jobs GROUP BY status
            ''')

            stats = {
                'scheduled': 0,
                'processing': 0,
                'completed': 0,
                'failed': 0
            }
            for row in cursor.fetchall():
                stats[row[0]] = row[1]

            stats['total'] = sum(stats.values())

            # Get next job time
            cursor = conn.execute('''
                SELECT schedule_type, scheduled_time FROM scheduled_jobs
                WHERE status = 'scheduled'
                ORDER BY priority DESC
                LIMIT 1
            ''')
            row = cursor.fetchone()
            if row:
                stats['next_job_type'] = row[0]
                stats['next_job_time'] = row[1]

            return stats

    def set_overnight_window(self, start: dt_time, end: dt_time):
        """Configure overnight processing window"""
        self.OVERNIGHT_START = start
        self.OVERNIGHT_END = end

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO schedule_settings (key, value)
                VALUES ('overnight_start', ?)
            ''', (start.isoformat(),))
            conn.execute('''
                INSERT OR REPLACE INTO schedule_settings (key, value)
                VALUES ('overnight_end', ?)
            ''', (end.isoformat(),))
            conn.commit()

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """Remove old completed/failed jobs"""
        cutoff = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                DELETE FROM scheduled_jobs
                WHERE status IN ('completed', 'failed')
                  AND completed_at < ?
            ''', (cutoff.isoformat(),))
            conn.commit()
            return conn.total_changes


# ============================================================
# LAUNCHD INTEGRATION
# ============================================================

class LaunchdScheduler:
    """Create macOS launchd jobs for scheduled processing"""

    PLIST_DIR = Path.home() / 'Library' / 'LaunchAgents'
    PLIST_PREFIX = 'com.nubhq.scheduled'

    def __init__(self, python_path: Path, script_path: Path):
        self.python_path = python_path
        self.script_path = script_path

    def create_overnight_job(self, hour: int = 1, minute: int = 0) -> Path:
        """Create launchd job for overnight processing"""
        plist_name = f"{self.PLIST_PREFIX}.overnight.plist"
        plist_path = self.PLIST_DIR / plist_name

        plist_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{self.PLIST_PREFIX}.overnight</string>

    <key>ProgramArguments</key>
    <array>
        <string>{self.python_path}</string>
        <string>{self.script_path}</string>
        <string>--scheduled</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/tmp/nubhq-scheduled.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/nubhq-scheduled-error.log</string>
</dict>
</plist>
'''

        plist_path.write_text(plist_content)
        logging.info(f"Created launchd job: {plist_path}")

        return plist_path

    def load_job(self, plist_path: Path) -> bool:
        """Load a launchd job"""
        try:
            subprocess.run(['launchctl', 'load', str(plist_path)], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def unload_job(self, plist_path: Path) -> bool:
        """Unload a launchd job"""
        try:
            subprocess.run(['launchctl', 'unload', str(plist_path)], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def list_jobs(self) -> List[str]:
        """List NubHQ scheduled jobs"""
        try:
            result = subprocess.run(
                ['launchctl', 'list'],
                capture_output=True, text=True
            )
            return [
                line.split()[-1]
                for line in result.stdout.split('\n')
                if self.PLIST_PREFIX in line
            ]
        except:
            return []
