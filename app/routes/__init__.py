from .approvals import router as approvals_router
from .dashboard import router as dashboard_router
from .fan_mail import router as fan_mail_router
from .settings import router as settings_router
from .calendar import router as calendar_router
from .ai_trainer import router as ai_trainer_router
from .auth import router as auth_router
from .search import router as search_router
from .posts import router as posts_router
from .templates import router as templates_router
from .analytics import router as analytics_router
from .activity import router as activity_router
from .email_campaigns import router as email_campaigns_router
from .media import router as media_router
from .video_pipeline import router as video_pipeline_router
from .webhooks import router as webhooks_router
from .video_tools import router as video_tools_router

__all__ = [
    "approvals_router",
    "dashboard_router",
    "fan_mail_router",
    "settings_router",
    "calendar_router",
    "ai_trainer_router",
    "auth_router",
    "search_router",
    "posts_router",
    "templates_router",
    "analytics_router",
    "activity_router",
    "email_campaigns_router",
    "media_router",
    "video_pipeline_router",
    "webhooks_router",
    "video_tools_router",
]
