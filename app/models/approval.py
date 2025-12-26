from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from ..database import Base


class Approval(Base):
    __tablename__ = "approvals"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String(50), default="message")
    content = Column(Text, nullable=False)
    recipient = Column(String(100), nullable=False)
    status = Column(String(20), default="pending")  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
