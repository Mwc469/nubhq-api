"""
Post model for social media content.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..database import Base


class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    platform = Column(String(50), nullable=False)  # instagram, twitter, tiktok, youtube
    post_type = Column(String(50), default="image")  # image, video, carousel, text
    hashtags = Column(JSON, default=list)
    media_urls = Column(JSON, default=list)
    status = Column(String(20), default="draft", index=True)  # draft, scheduled, published, failed
    scheduled_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    engagement = Column(JSON, nullable=True)  # likes, comments, shares, views
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="posts")
