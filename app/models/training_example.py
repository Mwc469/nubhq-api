from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from ..database import Base


class TrainingExample(Base):
    __tablename__ = "training_examples"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), default="general")  # greeting, thanks, question, promo, etc.
    input_message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
