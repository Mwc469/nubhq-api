from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from datetime import datetime
from ..database import Base


class Template(Base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(50), nullable=False)  # engagement, promotional, announcement, educational
    platform = Column(String(50), nullable=True)  # instagram, twitter, tiktok, youtube, or null for all
    hashtags = Column(String(500), nullable=True)
    is_favorite = Column(Boolean, default=False)
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
