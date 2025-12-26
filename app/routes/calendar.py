from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta

from ..database import get_db
from ..models.scheduled_post import ScheduledPost
from ..schemas.scheduled_post import ScheduledPostCreate, ScheduledPostUpdate, ScheduledPostResponse

router = APIRouter(prefix="/api/calendar", tags=["calendar"])


@router.get("", response_model=List[ScheduledPostResponse])
def get_scheduled_posts(
    start: datetime = Query(None),
    end: datetime = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(ScheduledPost)
    if start:
        query = query.filter(ScheduledPost.scheduled_at >= start)
    if end:
        query = query.filter(ScheduledPost.scheduled_at <= end)
    return query.order_by(ScheduledPost.scheduled_at).all()


@router.get("/{post_id}", response_model=ScheduledPostResponse)
def get_scheduled_post(post_id: int, db: Session = Depends(get_db)):
    post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Scheduled post not found")
    return post


@router.post("", response_model=ScheduledPostResponse)
def create_scheduled_post(post: ScheduledPostCreate, db: Session = Depends(get_db)):
    db_post = ScheduledPost(**post.model_dump())
    db.add(db_post)
    db.commit()
    db.refresh(db_post)
    return db_post


@router.patch("/{post_id}", response_model=ScheduledPostResponse)
def update_scheduled_post(post_id: int, update: ScheduledPostUpdate, db: Session = Depends(get_db)):
    post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Scheduled post not found")

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(post, field, value)

    db.commit()
    db.refresh(post)
    return post


@router.delete("/{post_id}")
def delete_scheduled_post(post_id: int, db: Session = Depends(get_db)):
    post = db.query(ScheduledPost).filter(ScheduledPost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Scheduled post not found")

    db.delete(post)
    db.commit()
    return {"message": "Deleted"}
