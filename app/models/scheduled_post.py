"""
ScheduledPost model for calendar/scheduling features.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..database import Base


class ScheduledPost(Base):
    __tablename__ = "scheduled_posts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    platform = Column(String(50), nullable=False, default="general")
    content = Column(Text)
    scheduled_at = Column(DateTime, nullable=False)
    status = Column(String(20), default="scheduled")  # scheduled, published, cancelled
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="scheduled_posts")
