from pydantic import BaseModel
from typing import List, Optional


class StatItem(BaseModel):
    label: str
    value: str
    trend: Optional[str] = None


class ActivityItem(BaseModel):
    id: int
    description: str
    time_ago: str
    status: str


class DashboardStats(BaseModel):
    pending_approvals: int
    active_fans: int
    avg_response_time: str
    engagement_rate: str
    recent_activity: List[ActivityItem]
