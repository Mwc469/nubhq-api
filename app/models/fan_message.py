"""
FanMessage model for incoming fan messages.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from ..database import Base


class FanMessage(Base):
    __tablename__ = "fan_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    sender_id = Column(String(100), nullable=False)
    sender_name = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    reply = Column(Text, nullable=True)
    replied_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="fan_messages")
