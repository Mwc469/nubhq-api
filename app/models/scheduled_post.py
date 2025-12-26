from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from ..database import Base


class ScheduledPost(Base):
    __tablename__ = "scheduled_posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    scheduled_at = Column(DateTime, nullable=False)
    status = Column(String(20), default="scheduled")  # scheduled, published, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)
