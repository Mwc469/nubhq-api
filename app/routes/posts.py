"""
Posts routes for CRUD operations on social media posts.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel

from ..database import get_db
from ..models.post import Post
from ..models.user import User
from ..auth import get_required_user

router = APIRouter(prefix="/api/posts", tags=["posts"])


class PostCreate(BaseModel):
    """Schema for creating a post."""
    content: str
    platform: str = "instagram"
    post_type: str = "image"
    hashtags: List[str] = []
    media_urls: List[str] = []
    scheduled_at: Optional[str] = None


class PostUpdate(BaseModel):
    """Schema for updating a post."""
    content: Optional[str] = None
    platform: Optional[str] = None
    post_type: Optional[str] = None
    hashtags: Optional[List[str]] = None
    media_urls: Optional[List[str]] = None
    status: Optional[str] = None
    scheduled_at: Optional[str] = None


def post_to_dict(post: Post) -> dict:
    """Convert a Post model to a dictionary response."""
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


@router.get("", response_model=List[dict])
def get_posts(
    status: Optional[str] = None,
    platform: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get all posts for the current user with optional filtering."""
    query = db.query(Post).filter(Post.user_id == current_user.id)

    if status:
        query = query.filter(Post.status == status)
    if platform:
        query = query.filter(Post.platform == platform)

    posts = query.order_by(Post.created_at.desc()).all()
    return [post_to_dict(p) for p in posts]


@router.get("/{post_id}", response_model=dict)
def get_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get a single post by ID (must belong to current user)."""
    post = db.query(Post).filter(
        Post.id == post_id,
        Post.user_id == current_user.id
    ).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post_to_dict(post)


@router.post("", response_model=dict)
def create_post(
    post_data: PostCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Create a new post for the current user."""
    scheduled_at = None
    if post_data.scheduled_at:
        scheduled_at = datetime.fromisoformat(post_data.scheduled_at.replace("Z", "+00:00"))

    post = Post(
        user_id=current_user.id,
        content=post_data.content,
        platform=post_data.platform,
        post_type=post_data.post_type,
        hashtags=post_data.hashtags,
        media_urls=post_data.media_urls,
        status="scheduled" if scheduled_at else "draft",
        scheduled_at=scheduled_at,
    )
    db.add(post)
    db.commit()
    db.refresh(post)

    return post_to_dict(post)


@router.patch("/{post_id}", response_model=dict)
def update_post(
    post_id: int,
    post_update: PostUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Update a post (must belong to current user)."""
    post = db.query(Post).filter(
        Post.id == post_id,
        Post.user_id == current_user.id
    ).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    update_data = post_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            if key == "scheduled_at" and isinstance(value, str):
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            setattr(post, key, value)

    db.commit()
    db.refresh(post)

    return post_to_dict(post)


@router.delete("/{post_id}")
def delete_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Delete a post (must belong to current user)."""
    post = db.query(Post).filter(
        Post.id == post_id,
        Post.user_id == current_user.id
    ).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    db.delete(post)
    db.commit()
    return {"message": "Post deleted"}


@router.post("/{post_id}/publish")
def publish_post(
    post_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Publish a draft or scheduled post immediately."""
    post = db.query(Post).filter(
        Post.id == post_id,
        Post.user_id == current_user.id
    ).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    post.status = "published"
    post.published_at = datetime.now(timezone.utc)
    post.scheduled_at = None

    db.commit()
    db.refresh(post)

    return post_to_dict(post)
