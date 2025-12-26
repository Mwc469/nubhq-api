from pydantic import BaseModel
from datetime import datetime


class FanMessageBase(BaseModel):
    sender_id: str
    sender_name: str
    content: str


class FanMessageCreate(FanMessageBase):
    pass


class FanMessageResponse(FanMessageBase):
    id: int
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True
