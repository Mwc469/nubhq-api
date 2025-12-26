"""
Enhanced Preference Learning System

Tracks user preferences and adjusts defaults over time:
- Records all processing decisions
- Learns patterns based on video characteristics
- Adjusts confidence thresholds based on approval rates
- Suggests profile overrides based on history
- Exports/imports learned preferences
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict


@dataclass
class LearningPattern:
    """A learned pattern from user behavior"""
    pattern_type: str       # 'profile_override', 'setting_preference', 'approval_pattern'
    condition: Dict         # When this pattern applies
    action: str             # What to do
    confidence: float       # How confident (0-1)
    sample_count: int       # How many examples
    last_seen: str          # ISO timestamp


@dataclass
class ProcessingFeedback:
    """User feedback on processed video"""
    video_id: str
    approved: bool
    edits_made: Optional[Dict]      # Changes user made
    time_to_approve: Optional[float]  # Seconds to approve (None if rejected)
    profile_used: str
    video_characteristics: Dict
    timestamp: str


class EnhancedPreferenceLearner:
    """
    Enhanced learning system that tracks patterns and adjusts behavior.

    Key features:
    - Pattern recognition across video types
    - Approval rate tracking per profile
    - Auto-adjustment of confidence thresholds
    - Export/import of learned preferences
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database with enhanced schema"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Processing decisions (original)
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

            # Processing feedback (new)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    edits_made TEXT,
                    time_to_approve REAL,
                    profile_used TEXT NOT NULL,
                    video_characteristics TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')

            # Learned patterns (new)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    condition_json TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sample_count INTEGER DEFAULT 1,
                    last_seen TEXT NOT NULL,
                    UNIQUE(pattern_type, condition_json, action)
                )
            ''')

            # Profile performance stats (new)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS profile_stats (
                    profile_name TEXT PRIMARY KEY,
                    total_uses INTEGER DEFAULT 0,
                    approvals INTEGER DEFAULT 0,
                    rejections INTEGER DEFAULT 0,
                    avg_time_to_approve REAL,
                    last_used TEXT
                )
            ''')

            # System settings (new)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')

            conn.commit()

    def record_feedback(self, feedback: ProcessingFeedback) -> None:
        """Record user feedback on a processed video"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO processing_feedback
                (video_id, approved, edits_made, time_to_approve, profile_used, video_characteristics, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.video_id,
                1 if feedback.approved else 0,
                json.dumps(feedback.edits_made) if feedback.edits_made else None,
                feedback.time_to_approve,
                feedback.profile_used,
                json.dumps(feedback.video_characteristics),
                feedback.timestamp
            ))

            # Update profile stats
            self._update_profile_stats(conn, feedback)

            # Learn patterns from this feedback
            self._learn_from_feedback(conn, feedback)

            conn.commit()

    def _update_profile_stats(self, conn: sqlite3.Connection, feedback: ProcessingFeedback) -> None:
        """Update profile performance statistics"""
        # Get current stats
        cursor = conn.execute(
            'SELECT total_uses, approvals, rejections, avg_time_to_approve FROM profile_stats WHERE profile_name = ?',
            (feedback.profile_used,)
        )
        row = cursor.fetchone()

        if row:
            total, approvals, rejections, avg_time = row
            total += 1
            if feedback.approved:
                approvals += 1
                if feedback.time_to_approve:
                    avg_time = (avg_time * (approvals - 1) + feedback.time_to_approve) / approvals if avg_time else feedback.time_to_approve
            else:
                rejections += 1

            conn.execute('''
                UPDATE profile_stats
                SET total_uses = ?, approvals = ?, rejections = ?, avg_time_to_approve = ?, last_used = ?
                WHERE profile_name = ?
            ''', (total, approvals, rejections, avg_time, datetime.now().isoformat(), feedback.profile_used))
        else:
            conn.execute('''
                INSERT INTO profile_stats (profile_name, total_uses, approvals, rejections, avg_time_to_approve, last_used)
                VALUES (?, 1, ?, ?, ?, ?)
            ''', (
                feedback.profile_used,
                1 if feedback.approved else 0,
                0 if feedback.approved else 1,
                feedback.time_to_approve if feedback.approved else None,
                datetime.now().isoformat()
            ))

    def _learn_from_feedback(self, conn: sqlite3.Connection, feedback: ProcessingFeedback) -> None:
        """Extract and store learned patterns from feedback"""

        # Learn profile preference based on video characteristics
        chars = feedback.video_characteristics

        # Pattern: content_type + aspect_ratio -> profile
        if 'content_type' in chars and 'aspect_ratio' in chars:
            condition = {
                'content_type': chars.get('content_type'),
                'aspect_ratio': chars.get('aspect_ratio')
            }

            self._update_pattern(
                conn,
                pattern_type='profile_preference',
                condition=condition,
                action=feedback.profile_used,
                success=feedback.approved
            )

        # Pattern: duration bucket -> profile
        if 'duration' in chars:
            duration = chars['duration']
            duration_bucket = 'short' if duration < 60 else 'medium' if duration < 300 else 'long'

            self._update_pattern(
                conn,
                pattern_type='duration_profile',
                condition={'duration_bucket': duration_bucket},
                action=feedback.profile_used,
                success=feedback.approved
            )

        # Learn from edits - if user consistently makes same edit, learn it
        if feedback.edits_made:
            for setting, new_value in feedback.edits_made.items():
                self._update_pattern(
                    conn,
                    pattern_type='setting_override',
                    condition={'profile': feedback.profile_used},
                    action=f"{setting}={new_value}",
                    success=True  # Edits are always "correct"
                )

    def _update_pattern(
        self,
        conn: sqlite3.Connection,
        pattern_type: str,
        condition: Dict,
        action: str,
        success: bool
    ) -> None:
        """Update or create a learned pattern"""
        condition_json = json.dumps(condition, sort_keys=True)

        cursor = conn.execute('''
            SELECT id, confidence, sample_count FROM learned_patterns
            WHERE pattern_type = ? AND condition_json = ? AND action = ?
        ''', (pattern_type, condition_json, action))

        row = cursor.fetchone()

        if row:
            pattern_id, confidence, sample_count = row
            # Update confidence using exponential moving average
            new_sample = 1.0 if success else 0.0
            alpha = 0.2  # Learning rate
            new_confidence = confidence * (1 - alpha) + new_sample * alpha
            new_count = sample_count + 1

            conn.execute('''
                UPDATE learned_patterns
                SET confidence = ?, sample_count = ?, last_seen = ?
                WHERE id = ?
            ''', (new_confidence, new_count, datetime.now().isoformat(), pattern_id))
        else:
            # Create new pattern
            conn.execute('''
                INSERT INTO learned_patterns (pattern_type, condition_json, action, confidence, sample_count, last_seen)
                VALUES (?, ?, ?, ?, 1, ?)
            ''', (pattern_type, condition_json, action, 0.5 if success else 0.3, datetime.now().isoformat()))

    def get_profile_recommendation(self, video_characteristics: Dict) -> Tuple[Optional[str], float]:
        """
        Get a profile recommendation based on learned patterns.

        Returns: (profile_name, confidence) or (None, 0)
        """
        with sqlite3.connect(self.db_path) as conn:
            # Check content_type + aspect_ratio pattern
            if 'content_type' in video_characteristics and 'aspect_ratio' in video_characteristics:
                condition = {
                    'content_type': video_characteristics.get('content_type'),
                    'aspect_ratio': video_characteristics.get('aspect_ratio')
                }
                condition_json = json.dumps(condition, sort_keys=True)

                cursor = conn.execute('''
                    SELECT action, confidence, sample_count FROM learned_patterns
                    WHERE pattern_type = 'profile_preference' AND condition_json = ?
                    ORDER BY confidence DESC, sample_count DESC
                    LIMIT 1
                ''', (condition_json,))

                row = cursor.fetchone()
                if row and row[1] > 0.6 and row[2] >= 3:
                    return row[0], row[1]

            # Check duration pattern
            if 'duration' in video_characteristics:
                duration = video_characteristics['duration']
                duration_bucket = 'short' if duration < 60 else 'medium' if duration < 300 else 'long'

                condition_json = json.dumps({'duration_bucket': duration_bucket}, sort_keys=True)

                cursor = conn.execute('''
                    SELECT action, confidence, sample_count FROM learned_patterns
                    WHERE pattern_type = 'duration_profile' AND condition_json = ?
                    ORDER BY confidence DESC, sample_count DESC
                    LIMIT 1
                ''', (condition_json,))

                row = cursor.fetchone()
                if row and row[1] > 0.5 and row[2] >= 5:
                    return row[0], row[1] * 0.8  # Slightly lower confidence for duration-only

        return None, 0

    def get_setting_overrides(self, profile_name: str) -> Dict[str, str]:
        """Get learned setting overrides for a profile"""
        overrides = {}

        with sqlite3.connect(self.db_path) as conn:
            condition_json = json.dumps({'profile': profile_name}, sort_keys=True)

            cursor = conn.execute('''
                SELECT action, confidence, sample_count FROM learned_patterns
                WHERE pattern_type = 'setting_override' AND condition_json = ?
                  AND confidence > 0.7 AND sample_count >= 3
            ''', (condition_json,))

            for row in cursor.fetchall():
                action = row[0]
                if '=' in action:
                    setting, value = action.split('=', 1)
                    overrides[setting] = value

        return overrides

    def get_profile_stats(self) -> Dict[str, Dict]:
        """Get performance stats for all profiles"""
        stats = {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT profile_name, total_uses, approvals, rejections, avg_time_to_approve
                FROM profile_stats
                ORDER BY total_uses DESC
            ''')

            for row in cursor.fetchall():
                name, total, approvals, rejections, avg_time = row
                approval_rate = approvals / total if total > 0 else 0

                stats[name] = {
                    'total_uses': total,
                    'approvals': approvals,
                    'rejections': rejections,
                    'approval_rate': approval_rate,
                    'avg_time_to_approve': avg_time,
                    'performance': 'good' if approval_rate > 0.8 else 'okay' if approval_rate > 0.6 else 'poor'
                }

        return stats

    def get_learning_summary(self) -> Dict:
        """Get overall learning summary"""
        with sqlite3.connect(self.db_path) as conn:
            # Total decisions
            cursor = conn.execute('SELECT COUNT(*) FROM decisions')
            total_decisions = cursor.fetchone()[0]

            # Total feedback
            cursor = conn.execute('SELECT COUNT(*) FROM processing_feedback')
            total_feedback = cursor.fetchone()[0]

            # Learned patterns
            cursor = conn.execute('SELECT COUNT(*) FROM learned_patterns WHERE confidence > 0.6')
            confident_patterns = cursor.fetchone()[0]

            # Recent approval rate (last 50)
            cursor = conn.execute('''
                SELECT AVG(approved) FROM (
                    SELECT approved FROM processing_feedback
                    ORDER BY id DESC LIMIT 50
                )
            ''')
            recent_approval_rate = cursor.fetchone()[0] or 0

            # Best and worst profiles
            profile_stats = self.get_profile_stats()
            best_profile = max(profile_stats.items(), key=lambda x: x[1]['approval_rate'])[0] if profile_stats else None
            worst_profile = min(profile_stats.items(), key=lambda x: x[1]['approval_rate'])[0] if profile_stats else None

        return {
            'total_decisions': total_decisions,
            'total_feedback': total_feedback,
            'confident_patterns': confident_patterns,
            'recent_approval_rate': recent_approval_rate,
            'best_profile': best_profile,
            'worst_profile': worst_profile,
            'learning_status': 'active' if total_feedback > 10 else 'warming_up'
        }

    def export_preferences(self) -> Dict:
        """Export all learned preferences for backup/sharing"""
        with sqlite3.connect(self.db_path) as conn:
            # Export patterns
            cursor = conn.execute('''
                SELECT pattern_type, condition_json, action, confidence, sample_count
                FROM learned_patterns
                WHERE confidence > 0.5 AND sample_count >= 2
            ''')

            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'type': row[0],
                    'condition': json.loads(row[1]),
                    'action': row[2],
                    'confidence': row[3],
                    'samples': row[4]
                })

            # Export profile stats
            profile_stats = self.get_profile_stats()

        return {
            'version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'patterns': patterns,
            'profile_stats': profile_stats
        }

    def import_preferences(self, data: Dict) -> int:
        """Import preferences from backup. Returns count of imported patterns."""
        imported = 0

        with sqlite3.connect(self.db_path) as conn:
            for pattern in data.get('patterns', []):
                try:
                    condition_json = json.dumps(pattern['condition'], sort_keys=True)

                    conn.execute('''
                        INSERT OR REPLACE INTO learned_patterns
                        (pattern_type, condition_json, action, confidence, sample_count, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        pattern['type'],
                        condition_json,
                        pattern['action'],
                        pattern['confidence'],
                        pattern['samples'],
                        datetime.now().isoformat()
                    ))
                    imported += 1
                except Exception as e:
                    logging.warning(f"Failed to import pattern: {e}")

            conn.commit()

        return imported

    def reset_learning(self, keep_stats: bool = True) -> None:
        """Reset learned patterns (optionally keep stats)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM learned_patterns')

            if not keep_stats:
                conn.execute('DELETE FROM profile_stats')
                conn.execute('DELETE FROM processing_feedback')

            conn.commit()

        logging.info("Learning reset complete")
