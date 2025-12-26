from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..database import get_db
from ..schemas.posts import PostCreate, PostUpdate, PostResponse

router = APIRouter(prefix="/api/posts", tags=["posts"])


# Mock data for now - will be replaced with DB models
mock_posts = [
    {
        "id": 1,
        "content": "Behind the scenes 📸 #content #creator",
        "platform": "instagram",
        "post_type": "image",
        "status": "published",
        "scheduled_at": None,
        "published_at": "2024-01-15T14:00:00Z",
        "hashtags": ["content", "creator"],
        "media_urls": ["https://picsum.photos/400/400?random=1"],
        "engagement": {"likes": 1234, "comments": 56, "shares": 23},
        "created_at": "2024-01-15T10:00:00Z",
    },
    {
        "id": 2,
        "content": "New content dropping soon! 🔥",
        "platform": "twitter",
        "post_type": "text",
        "status": "scheduled",
        "scheduled_at": "2024-01-20T18:00:00Z",
        "published_at": None,
        "hashtags": [],
        "media_urls": [],
        "engagement": None,
        "created_at": "2024-01-14T12:00:00Z",
    },
    {
        "id": 3,
        "content": "Thank you all for 10K! 🎉",
        "platform": "instagram",
        "post_type": "reel",
        "status": "draft",
        "scheduled_at": None,
        "published_at": None,
        "hashtags": ["milestone", "thankyou"],
        "media_urls": [],
        "engagement": None,
        "created_at": "2024-01-13T09:00:00Z",
    },
]


@router.get("", response_model=List[dict])
def get_posts(
    status: Optional[str] = None,
    platform: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all posts with optional filtering"""
    posts = mock_posts

    if status:
        posts = [p for p in posts if p["status"] == status]
    if platform:
        posts = [p for p in posts if p["platform"] == platform]

    return posts


@router.get("/{post_id}", response_model=dict)
def get_post(post_id: int, db: Session = Depends(get_db)):
    """Get a single post by ID"""
    post = next((p for p in mock_posts if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return post


@router.post("", response_model=dict)
def create_post(post: dict, db: Session = Depends(get_db)):
    """Create a new post"""
    new_post = {
        "id": len(mock_posts) + 1,
        **post,
        "status": post.get("scheduled_at") and "scheduled" or "draft",
        "created_at": datetime.utcnow().isoformat(),
        "engagement": None,
    }
    mock_posts.append(new_post)
    return new_post


@router.patch("/{post_id}", response_model=dict)
def update_post(post_id: int, post_update: dict, db: Session = Depends(get_db)):
    """Update a post"""
    post = next((p for p in mock_posts if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    for key, value in post_update.items():
        if value is not None:
            post[key] = value

    return post


@router.delete("/{post_id}")
def delete_post(post_id: int, db: Session = Depends(get_db)):
    """Delete a post"""
    global mock_posts
    post = next((p for p in mock_posts if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    mock_posts = [p for p in mock_posts if p["id"] != post_id]
    return {"message": "Post deleted"}


@router.post("/{post_id}/publish")
def publish_post(post_id: int, db: Session = Depends(get_db)):
    """Publish a draft or scheduled post immediately"""
    post = next((p for p in mock_posts if p["id"] == post_id), None)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    post["status"] = "published"
    post["published_at"] = datetime.utcnow().isoformat()
    post["scheduled_at"] = None

    return post
