from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from datetime import datetime
from ..database import Base


class FanMessage(Base):
    __tablename__ = "fan_messages"

    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(String(100), nullable=False)
    sender_name = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
