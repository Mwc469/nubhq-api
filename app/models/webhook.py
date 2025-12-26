"""
Webhook model for event notifications.
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON
from datetime import datetime, timezone
from ..database import Base
import secrets


class Webhook(Base):
    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)  # Optional, for per-user webhooks
    name = Column(String(100), nullable=True)  # Friendly name
    url = Column(String(500), nullable=False)
    events = Column(JSON, nullable=False)  # ["job.completed", "job.failed", ...]
    secret = Column(String(64), default=lambda: secrets.token_hex(32))  # HMAC signing key
    active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime, nullable=True)
    last_status_code = Column(Integer, nullable=True)
    failure_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Available webhook events
    EVENTS = [
        "job.started",
        "job.progress",
        "job.completed",
        "job.failed",
        "review.pending",
        "approval.approved",
        "approval.rejected",
    ]

    def to_dict(self, include_secret=False):
        """Convert to dictionary for API responses"""
        data = {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "events": self.events,
            "active": self.active,
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "last_status_code": self.last_status_code,
            "failure_count": self.failure_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_secret:
            data["secret"] = self.secret
        return data
