from .approvals import router as approvals_router
from .dashboard import router as dashboard_router
from .fan_mail import router as fan_mail_router
from .settings import router as settings_router
from .calendar import router as calendar_router
from .ai_trainer import router as ai_trainer_router

__all__ = ["approvals_router", "dashboard_router", "fan_mail_router", "settings_router", "calendar_router", "ai_trainer_router"]
