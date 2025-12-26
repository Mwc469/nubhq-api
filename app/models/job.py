"""
Job model for video pipeline job persistence.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from datetime import datetime, timezone
from ..database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(50), primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)  # Optional, for tracking who started job
    type = Column(String(50), nullable=False, index=True)  # batch, export, caption, thumbnail, watermark
    status = Column(String(20), default="pending", index=True)  # pending, processing, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    priority = Column(Integer, default=0, index=True)  # 0=normal, 1=high, 2=urgent
    batch_id = Column(String(50), nullable=True, index=True)  # Group related jobs
    input_data = Column(JSON, nullable=True)  # Request parameters
    output_data = Column(JSON, nullable=True)  # Results
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "job_id": self.id,
            "type": self.type,
            "status": self.status,
            "progress": self.progress,
            "priority": self.priority,
            "batch_id": self.batch_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
