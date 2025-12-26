from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta

from ..database import get_db
from ..models.activity import Activity

router = APIRouter(prefix="/api/activity", tags=["activity"])


@router.get("", response_model=List[dict])
def get_activities(
    activity_type: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get activity log with optional filtering and pagination"""
    query = db.query(Activity)

    if activity_type:
        query = query.filter(Activity.activity_type == activity_type)

    activities = query.order_by(Activity.created_at.desc()).offset(offset).limit(limit).all()

    return [
        {
            "id": a.id,
            "type": a.activity_type,
            "title": a.title,
            "description": a.description,
            "metadata": a.extra_data,
            "timestamp": a.created_at.isoformat(),
        }
        for a in activities
    ]


@router.get("/stats")
def get_activity_stats(db: Session = Depends(get_db)):
    """Get activity statistics"""
    total = db.query(func.count(Activity.id)).scalar() or 0

    # Count by type
    type_counts = db.query(
        Activity.activity_type,
        func.count(Activity.id)
    ).group_by(Activity.activity_type).all()

    # Last 24 hours
    day_ago = datetime.utcnow() - timedelta(days=1)
    today_count = db.query(func.count(Activity.id)).filter(
        Activity.created_at >= day_ago
    ).scalar() or 0

    # Build stats from type counts
    stats = {
        "total_today": today_count,
        "approvals_pending": 0,
        "ai_actions": 0,
        "posts_scheduled": 0,
        "messages_received": 0,
    }

    for activity_type, count in type_counts:
        if activity_type == "approval":
            stats["approvals_pending"] = count
        elif activity_type == "ai":
            stats["ai_actions"] = count
        elif activity_type == "post":
            stats["posts_scheduled"] = count
        elif activity_type == "message":
            stats["messages_received"] = count

    return stats


@router.get("/recent")
def get_recent_activity(limit: int = 5, db: Session = Depends(get_db)):
    """Get recent activity for dashboard widget"""
    activities = db.query(Activity).order_by(Activity.created_at.desc()).limit(limit).all()

    return [
        {
            "id": a.id,
            "type": a.activity_type,
            "title": a.title,
            "description": a.description,
            "metadata": a.extra_data,
            "timestamp": a.created_at.isoformat(),
        }
        for a in activities
    ]


@router.post("", response_model=dict)
def create_activity(activity_data: dict, db: Session = Depends(get_db)):
    """Create a new activity log entry"""
    activity = Activity(
        activity_type=activity_data.get("type", "other"),
        title=activity_data.get("title", ""),
        description=activity_data.get("description"),
        extra_data=activity_data.get("metadata"),
    )
    db.add(activity)
    db.commit()
    db.refresh(activity)

    return {
        "id": activity.id,
        "type": activity.activity_type,
        "title": activity.title,
        "description": activity.description,
        "metadata": activity.metadata,
        "timestamp": activity.created_at.isoformat(),
    }
