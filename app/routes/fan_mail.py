"""
Fan mail routes for managing incoming fan messages.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timezone

from ..database import get_db
from ..models.fan_message import FanMessage
from ..models.user import User
from ..auth import get_required_user
from ..schemas.fan_message import FanMessageCreate, FanMessageResponse, FanMessageReply

router = APIRouter(prefix="/api/fan-mail", tags=["fan-mail"])


@router.get("", response_model=List[FanMessageResponse])
def get_messages(
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get all fan messages for the current user."""
    query = db.query(FanMessage).filter(FanMessage.user_id == current_user.id)
    if unread_only:
        query = query.filter(FanMessage.is_read == False)
    return query.order_by(FanMessage.created_at.desc()).all()


@router.get("/unread/count")
def get_unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get count of unread messages for the current user."""
    count = db.query(FanMessage).filter(
        FanMessage.user_id == current_user.id,
        FanMessage.is_read == False
    ).count()
    return {"count": count}


@router.get("/{message_id}", response_model=FanMessageResponse)
def get_message(
    message_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get a single fan message by ID (must belong to current user)."""
    message = db.query(FanMessage).filter(
        FanMessage.id == message_id,
        FanMessage.user_id == current_user.id
    ).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


@router.post("", response_model=FanMessageResponse)
def create_message(
    message: FanMessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Create a new fan message for the current user."""
    db_message = FanMessage(
        user_id=current_user.id,
        **message.model_dump()
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


@router.patch("/{message_id}/read", response_model=FanMessageResponse)
def mark_as_read(
    message_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Mark a fan message as read (must belong to current user)."""
    message = db.query(FanMessage).filter(
        FanMessage.id == message_id,
        FanMessage.user_id == current_user.id
    ).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    message.is_read = True
    db.commit()
    db.refresh(message)
    return message


@router.post("/{message_id}/reply", response_model=FanMessageResponse)
def reply_to_message(
    message_id: int,
    reply_data: FanMessageReply,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Reply to a fan message (must belong to current user)."""
    message = db.query(FanMessage).filter(
        FanMessage.id == message_id,
        FanMessage.user_id == current_user.id
    ).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    message.reply = reply_data.reply
    message.replied_at = datetime.now(timezone.utc)
    message.is_read = True
    db.commit()
    db.refresh(message)
    return message
