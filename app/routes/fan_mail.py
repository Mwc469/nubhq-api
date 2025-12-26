from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from ..database import get_db
from ..models.fan_message import FanMessage
from ..schemas.fan_message import FanMessageCreate, FanMessageResponse, FanMessageReply

router = APIRouter(prefix="/api/fan-mail", tags=["fan-mail"])


@router.get("", response_model=List[FanMessageResponse])
def get_messages(unread_only: bool = False, db: Session = Depends(get_db)):
    query = db.query(FanMessage)
    if unread_only:
        query = query.filter(FanMessage.is_read == False)
    return query.order_by(FanMessage.created_at.desc()).all()


@router.get("/{message_id}", response_model=FanMessageResponse)
def get_message(message_id: int, db: Session = Depends(get_db)):
    message = db.query(FanMessage).filter(FanMessage.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@router.post("", response_model=FanMessageResponse)
def create_message(message: FanMessageCreate, db: Session = Depends(get_db)):
    db_message = FanMessage(**message.model_dump())
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


@router.patch("/{message_id}/read", response_model=FanMessageResponse)
def mark_as_read(message_id: int, db: Session = Depends(get_db)):
    message = db.query(FanMessage).filter(FanMessage.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    message.is_read = True
    db.commit()
    db.refresh(message)
    return message


@router.get("/unread/count")
def get_unread_count(db: Session = Depends(get_db)):
    count = db.query(FanMessage).filter(FanMessage.is_read == False).count()
    return {"count": count}


@router.post("/{message_id}/reply", response_model=FanMessageResponse)
def reply_to_message(message_id: int, reply_data: FanMessageReply, db: Session = Depends(get_db)):
    message = db.query(FanMessage).filter(FanMessage.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    message.reply = reply_data.reply
    message.replied_at = datetime.utcnow()
    message.is_read = True
    db.commit()
    db.refresh(message)
    return message
