from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..database import get_db
from ..models.post import Post

router = APIRouter(prefix="/api/posts", tags=["posts"])


@router.get("", response_model=List[dict])
def get_posts(
    status: Optional[str] = None,
    platform: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all posts with optional filtering"""
    query = db.query(Post)

    if status:
        query = query.filter(Post.status == status)
    if platform:
        query = query.filter(Post.platform == platform)

    posts = query.order_by(Post.created_at.desc()).all()
    return [
        {
            "id": p.id,
            "content": p.content,
            "platform": p.platform,
            "post_type": p.post_type,
            "status": p.status,
            "scheduled_at": p.scheduled_at.isoformat() if p.scheduled_at else None,
            "published_at": p.published_at.isoformat() if p.published_at else None,
            "hashtags": p.hashtags or [],
            "media_urls": p.media_urls or [],
            "engagement": p.engagement,
            "created_at": p.created_at.isoformat(),
        }
        for p in posts
    ]


@router.get("/{post_id}", response_model=dict)
def get_post(post_id: int, db: Session = Depends(get_db)):
    """Get a single post by ID"""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return {
        "id": post.id,
        "content": post.content,
        "platform": post.platform,
        "post_type": post.post_type,
        "status": post.status,
        "scheduled_at": post.scheduled_at.isoformat() if post.scheduled_at else None,
        "published_at": post.published_at.isoformat() if post.published_at else None,
        "hashtags": post.hashtags or [],
        "media_urls": post.media_urls or [],
        "engagement": post.engagement,
        "created_at": post.created_at.isoformat(),
    }


@router.post("", response_model=dict)
def create_post(post_data: dict, db: Session = Depends(get_db)):
    """Create a new post"""
    scheduled_at = post_data.get("scheduled_at")
    if scheduled_at and isinstance(scheduled_at, str):
        scheduled_at = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))

    post = Post(
        content=post_data.get("content", ""),
        platform=post_data.get("platform", "instagram"),
        post_type=post_data.get("post_type", "image"),
        hashtags=post_data.get("hashtags", []),
        media_urls=post_data.get("media_urls", []),
        status="scheduled" if scheduled_at else "draft",
        scheduled_at=scheduled_at,
    )
    db.add(post)
    db.commit()
    db.refresh(post)

    return {
        "id": post.id,
        "content": post.content,
        "platform": post.platform,
        "post_type": post.post_type,
        "status": post.status,
        "scheduled_at": post.scheduled_at.isoformat() if post.scheduled_at else None,
        "published_at": post.published_at.isoformat() if post.published_at else None,
        "hashtags": post.hashtags or [],
        "media_urls": post.media_urls or [],
        "engagement": post.engagement,
        "created_at": post.created_at.isoformat(),
    }


@router.patch("/{post_id}", response_model=dict)
def update_post(post_id: int, post_update: dict, db: Session = Depends(get_db)):
    """Update a post"""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    for key, value in post_update.items():
        if value is not None and hasattr(post, key):
            if key == "scheduled_at" and isinstance(value, str):
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            setattr(post, key, value)

    db.commit()
    db.refresh(post)

    return {
        "id": post.id,
        "content": post.content,
        "platform": post.platform,
        "post_type": post.post_type,
        "status": post.status,
        "scheduled_at": post.scheduled_at.isoformat() if post.scheduled_at else None,
        "published_at": post.published_at.isoformat() if post.published_at else None,
        "hashtags": post.hashtags or [],
        "media_urls": post.media_urls or [],
        "engagement": post.engagement,
        "created_at": post.created_at.isoformat(),
    }


@router.delete("/{post_id}")
def delete_post(post_id: int, db: Session = Depends(get_db)):
    """Delete a post"""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    db.delete(post)
    db.commit()
    return {"message": "Post deleted"}


@router.post("/{post_id}/publish")
def publish_post(post_id: int, db: Session = Depends(get_db)):
    """Publish a draft or scheduled post immediately"""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    post.status = "published"
    post.published_at = datetime.utcnow()
    post.scheduled_at = None

    db.commit()
    db.refresh(post)

    return {
        "id": post.id,
        "content": post.content,
        "platform": post.platform,
        "post_type": post.post_type,
        "status": post.status,
        "scheduled_at": post.scheduled_at.isoformat() if post.scheduled_at else None,
        "published_at": post.published_at.isoformat() if post.published_at else None,
        "hashtags": post.hashtags or [],
        "media_urls": post.media_urls or [],
        "engagement": post.engagement,
        "created_at": post.created_at.isoformat(),
    }
