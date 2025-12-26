from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from datetime import datetime
from ..database import Base


class Activity(Base):
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, index=True)
    activity_type = Column(String(50), nullable=False)  # post, approval, message, login, settings, ai_training
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Additional context like post_id, platform, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
