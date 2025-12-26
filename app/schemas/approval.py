from pydantic import BaseModel
from datetime import datetime


class ApprovalBase(BaseModel):
    type: str = "message"
    content: str
    recipient: str


class ApprovalCreate(ApprovalBase):
    pass


class ApprovalUpdate(BaseModel):
    status: str  # approved, rejected


class ApprovalResponse(ApprovalBase):
    id: int
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
