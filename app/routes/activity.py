"""
Activity routes for activity log operations.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel

from ..database import get_db
from ..models.activity import Activity
from ..models.user import User
from ..auth import get_required_user

router = APIRouter(prefix="/api/activity", tags=["activity"])


class ActivityCreate(BaseModel):
    """Schema for creating an activity log entry."""
    type: str = "other"
    title: str
    description: Optional[str] = None
    metadata: Optional[dict] = None


def activity_to_dict(activity: Activity) -> dict:
    """Convert an Activity model to a dictionary response."""
    return {
        "id": activity.id,
        "type": activity.activity_type,
        "title": activity.title,
        "description": activity.description,
        "metadata": activity.extra_data,
        "timestamp": activity.created_at.isoformat(),
    }


@router.get("", response_model=List[dict])
def get_activities(
    activity_type: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get activity log for the current user with optional filtering and pagination."""
    query = db.query(Activity).filter(Activity.user_id == current_user.id)

    if activity_type:
        query = query.filter(Activity.activity_type == activity_type)

    activities = query.order_by(Activity.created_at.desc()).offset(offset).limit(limit).all()
    return [activity_to_dict(a) for a in activities]


@router.get("/stats")
def get_activity_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get activity statistics for the current user."""
    base_query = db.query(Activity).filter(Activity.user_id == current_user.id)

    total = base_query.count()

    type_counts = db.query(
        Activity.activity_type,
        func.count(Activity.id)
    ).filter(
        Activity.user_id == current_user.id
    ).group_by(Activity.activity_type).all()

    day_ago = datetime.now(timezone.utc) - timedelta(days=1)
    today_count = base_query.filter(Activity.created_at >= day_ago).count()

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
def get_recent_activity(
    limit: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get recent activity for dashboard widget for the current user."""
    activities = db.query(Activity).filter(
        Activity.user_id == current_user.id
    ).order_by(Activity.created_at.desc()).limit(limit).all()

    return [activity_to_dict(a) for a in activities]


@router.post("", response_model=dict)
def create_activity(
    activity_data: ActivityCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Create a new activity log entry for the current user."""
    activity = Activity(
        user_id=current_user.id,
        activity_type=activity_data.type,
        title=activity_data.title,
        description=activity_data.description,
        extra_data=activity_data.metadata,
    )
    db.add(activity)
    db.commit()
    db.refresh(activity)

    return activity_to_dict(activity)
