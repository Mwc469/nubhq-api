from sqlalchemy import Column, Integer, String, DateTime, BigInteger
from datetime import datetime
from ..database import Base


class Media(Base):
    __tablename__ = "media"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    url = Column(String(1000), nullable=False)
    media_type = Column(String(50), default="image")  # image, video
    size = Column(BigInteger, default=0)  # in bytes
    mime_type = Column(String(100), nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
