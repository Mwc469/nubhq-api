from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from ..database import get_db

router = APIRouter(prefix="/api/activity", tags=["activity"])


# Mock activity data
mock_activities = [
    {
        "id": 1,
        "type": "approval",
        "action": "Message approved",
        "description": "Reply to @superfan123 approved and sent",
        "actor": "You",
        "actor_type": "user",
        "status": "completed",
        "timestamp": (datetime.utcnow() - timedelta(minutes=2)).isoformat(),
    },
    {
        "id": 2,
        "type": "ai",
        "action": "AI generated response",
        "description": "Created reply for message from @newbie_fan",
        "actor": "Nub AI",
        "actor_type": "ai",
        "status": "pending",
        "timestamp": (datetime.utcnow() - timedelta(minutes=15)).isoformat(),
    },
    {
        "id": 3,
        "type": "post",
        "action": "Post scheduled",
        "description": "Instagram post scheduled for tomorrow at 2:00 PM",
        "actor": "You",
        "actor_type": "user",
        "status": "scheduled",
        "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
    },
    {
        "id": 4,
        "type": "message",
        "action": "New fan message",
        "description": "Received message from @loyal_supporter",
        "actor": "System",
        "actor_type": "system",
        "status": "completed",
        "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
    },
    {
        "id": 5,
        "type": "approval",
        "action": "Message rejected",
        "description": "Reply to @spam_user rejected",
        "actor": "You",
        "actor_type": "user",
        "status": "rejected",
        "timestamp": (datetime.utcnow() - timedelta(hours=3)).isoformat(),
    },
    {
        "id": 6,
        "type": "ai",
        "action": "AI training updated",
        "description": "Voice profile updated with 5 new samples",
        "actor": "Nub AI",
        "actor_type": "ai",
        "status": "completed",
        "timestamp": (datetime.utcnow() - timedelta(hours=5)).isoformat(),
    },
    {
        "id": 7,
        "type": "post",
        "action": "Post published",
        "description": 'Twitter post "Behind the scenes 📸" went live',
        "actor": "System",
        "actor_type": "system",
        "status": "completed",
        "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat(),
    },
    {
        "id": 8,
        "type": "system",
        "action": "System backup",
        "description": "Daily backup completed successfully",
        "actor": "System",
        "actor_type": "system",
        "status": "completed",
        "timestamp": (datetime.utcnow() - timedelta(days=1)).isoformat(),
    },
]


@router.get("", response_model=List[dict])
def get_activities(
    activity_type: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get activity log with optional filtering"""
    activities = mock_activities

    if activity_type:
        activities = [a for a in activities if a["type"] == activity_type]

    return activities[offset:offset + limit]


@router.get("/stats")
def get_activity_stats(db: Session = Depends(get_db)):
    """Get activity statistics"""
    return {
        "total_today": 12,
        "approvals_pending": 3,
        "ai_actions": 8,
        "posts_scheduled": 5,
        "messages_received": 24,
    }


@router.get("/recent")
def get_recent_activity(
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get most recent activities for dashboard"""
    return mock_activities[:limit]
