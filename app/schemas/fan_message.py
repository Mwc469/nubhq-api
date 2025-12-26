from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class FanMessageBase(BaseModel):
    sender_id: str
    sender_name: str
    content: str


class FanMessageCreate(FanMessageBase):
    pass


class FanMessageReply(BaseModel):
    reply: str


class FanMessageResponse(FanMessageBase):
    id: int
    is_read: bool
    reply: Optional[str] = None
    replied_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True
