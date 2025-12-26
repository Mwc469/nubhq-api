from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class ScheduledPostBase(BaseModel):
    title: str
    content: Optional[str] = None
    scheduled_at: datetime


class ScheduledPostCreate(ScheduledPostBase):
    pass


class ScheduledPostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    status: Optional[str] = None


class ScheduledPostResponse(ScheduledPostBase):
    id: int
    status: str
    created_at: datetime

    class Config:
        from_attributes = True
