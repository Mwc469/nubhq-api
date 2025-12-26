from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from datetime import datetime
from ..database import Base


class EmailCampaign(Base):
    __tablename__ = "email_campaigns"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    subject = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    recipient_count = Column(Integer, default=0)
    status = Column(String(20), default="draft")  # draft, scheduled, sent
    scheduled_at = Column(DateTime, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    stats = Column(JSON, nullable=True)  # opens, clicks, bounces, unsubscribes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
