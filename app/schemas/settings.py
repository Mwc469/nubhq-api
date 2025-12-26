from pydantic import BaseModel
from typing import Optional


class SettingsUpdate(BaseModel):
    display_name: Optional[str] = None
    push_notifications: Optional[bool] = None
    email_notifications: Optional[bool] = None


class SettingsResponse(BaseModel):
    user_id: str
    display_name: Optional[str]
    push_notifications: bool
    email_notifications: bool

    class Config:
        from_attributes = True
