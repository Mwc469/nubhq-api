from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class TrainingExampleBase(BaseModel):
    category: str = "general"
    input_message: str
    response: str


class TrainingExampleCreate(TrainingExampleBase):
    pass


class TrainingExampleUpdate(BaseModel):
    category: Optional[str] = None
    input_message: Optional[str] = None
    response: Optional[str] = None


class TrainingExampleResponse(TrainingExampleBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingStats(BaseModel):
    total_examples: int
    by_category: dict
