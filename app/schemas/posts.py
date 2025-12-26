from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PostBase(BaseModel):
    content: str
    platform: str
    post_type: str = "image"
    hashtags: List[str] = []
    media_urls: List[str] = []


class PostCreate(PostBase):
    scheduled_at: Optional[datetime] = None


class PostUpdate(BaseModel):
    content: Optional[str] = None
    platform: Optional[str] = None
    post_type: Optional[str] = None
    hashtags: Optional[List[str]] = None
    media_urls: Optional[List[str]] = None
    scheduled_at: Optional[datetime] = None
    status: Optional[str] = None


class PostResponse(PostBase):
    id: int
    status: str
    scheduled_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    engagement: Optional[dict] = None
    created_at: datetime

    class Config:
        from_attributes = True
