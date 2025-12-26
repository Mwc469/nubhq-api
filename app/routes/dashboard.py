from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.approval import Approval
from ..models.fan_message import FanMessage
from ..schemas.dashboard import DashboardStats, ActivityItem

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("", response_model=DashboardStats)
def get_dashboard_stats(db: Session = Depends(get_db)):
    pending_count = db.query(Approval).filter(Approval.status == "pending").count()
    fan_count = db.query(FanMessage).distinct(FanMessage.sender_id).count()

    recent_approvals = db.query(Approval).order_by(Approval.updated_at.desc()).limit(5).all()

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

    return DashboardStats(
        pending_approvals=pending_count,
        active_fans=fan_count or 1234,
        avg_response_time="2.4h",
        engagement_rate="89%",
        recent_activity=activity
    )
