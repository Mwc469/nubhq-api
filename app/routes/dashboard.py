"""
Dashboard routes for dashboard statistics.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.approval import Approval
from ..models.fan_message import FanMessage
from ..models.user import User
from ..auth import get_required_user
from ..schemas.dashboard import DashboardStats, ActivityItem, ChartDataPoint

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardStats)
def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get dashboard statistics for the current user."""
    pending_count = db.query(Approval).filter(
        Approval.user_id == current_user.id,
        Approval.status == "pending"
    ).count()

    fan_count = db.query(FanMessage).filter(
        FanMessage.user_id == current_user.id
    ).distinct(FanMessage.sender_id).count()

    recent_approvals = db.query(Approval).filter(
        Approval.user_id == current_user.id
    ).order_by(Approval.updated_at.desc()).limit(5).all()

    activity = []
    for approval in recent_approvals:
        status_text = "Completed" if approval.status == "approved" else (
            "Rejected" if approval.status == "rejected" else "Pending"
        )
        activity.append(ActivityItem(
            id=approval.id,
            description=f"Message to {approval.recipient}",
            time_ago="2 hours ago",
            status=status_text
        ))

    engagement_chart = [
        ChartDataPoint(name="Mon", value=65, prev=55),
        ChartDataPoint(name="Tue", value=78, prev=62),
        ChartDataPoint(name="Wed", value=82, prev=71),
        ChartDataPoint(name="Thu", value=74, prev=68),
        ChartDataPoint(name="Fri", value=91, prev=75),
        ChartDataPoint(name="Sat", value=88, prev=82),
        ChartDataPoint(name="Sun", value=95, prev=78),
    ]

    messages_chart = [
        ChartDataPoint(name="Mon", value=24),
        ChartDataPoint(name="Tue", value=31),
        ChartDataPoint(name="Wed", value=28),
        ChartDataPoint(name="Thu", value=45),
        ChartDataPoint(name="Fri", value=52),
        ChartDataPoint(name="Sat", value=38),
        ChartDataPoint(name="Sun", value=41),
    ]

    return DashboardStats(
        pending_approvals=pending_count,
        active_fans=fan_count or 0,
        avg_response_time="0h",
        engagement_rate="0%",
        recent_activity=activity,
        engagement_chart=engagement_chart,
        messages_chart=messages_chart
    )
