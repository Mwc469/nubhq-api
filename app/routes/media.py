"""
Media routes for CRUD operations on media files.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from pydantic import BaseModel

from ..database import get_db
from ..models.media import Media
from ..models.user import User
from ..auth import get_required_user

router = APIRouter(prefix="/api/media", tags=["media"])


class MediaCreate(BaseModel):
    """Schema for creating a media record."""
    name: str
    url: str
    media_type: str = "image"
    size: int = 0
    mime_type: str = "image/jpeg"
    width: Optional[int] = None
    height: Optional[int] = None


class MediaUpdate(BaseModel):
    """Schema for updating a media record."""
    name: Optional[str] = None


def media_to_dict(media: Media) -> dict:
    """Convert a Media model to a dictionary response."""
    return {
        "id": media.id,
        "name": media.name,
        "url": media.url,
        "type": media.media_type,
        "size": media.size,
        "mime_type": media.mime_type,
        "width": media.width,
        "height": media.height,
        "created_at": media.created_at.isoformat(),
    }


@router.get("", response_model=List[dict])
def get_media(
    media_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get all media files for the current user with optional type filter."""
    query = db.query(Media).filter(Media.user_id == current_user.id)

    if media_type:
        query = query.filter(Media.media_type == media_type)

    media_items = query.order_by(Media.created_at.desc()).all()
    return [media_to_dict(m) for m in media_items]


@router.get("/stats")
def get_media_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get media library statistics for the current user."""
    base_query = db.query(Media).filter(Media.user_id == current_user.id)

    total_files = base_query.count()
    total_size = db.query(func.sum(Media.size)).filter(
        Media.user_id == current_user.id
    ).scalar() or 0
    images = base_query.filter(Media.media_type == "image").count()
    videos = base_query.filter(Media.media_type == "video").count()

    return {
        "total_files": total_files,
        "total_size": total_size,
        "images": images,
        "videos": videos,
    }


@router.get("/{media_id}", response_model=dict)
def get_media_item(
    media_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get a single media item by ID (must belong to current user)."""
    media = db.query(Media).filter(
        Media.id == media_id,
        Media.user_id == current_user.id
    ).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    return media_to_dict(media)


@router.post("", response_model=dict)
def upload_media(
    media_data: MediaCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Upload a new media file (creates record with URL) for the current user."""
    media = Media(
        user_id=current_user.id,
        name=media_data.name,
        url=media_data.url,
        media_type=media_data.media_type,
        size=media_data.size,
        mime_type=media_data.mime_type,
        width=media_data.width,
        height=media_data.height,
    )
    db.add(media)
    db.commit()
    db.refresh(media)

    return media_to_dict(media)


@router.delete("/{media_id}")
def delete_media(
    media_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Delete a media file (must belong to current user)."""
    media = db.query(Media).filter(
        Media.id == media_id,
        Media.user_id == current_user.id
    ).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    db.delete(media)
    db.commit()
    return {"message": "Media deleted"}


@router.patch("/{media_id}", response_model=dict)
def update_media(
    media_id: int,
    update: MediaUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Update media metadata (must belong to current user)."""
    media = db.query(Media).filter(
        Media.id == media_id,
        Media.user_id == current_user.id
    ).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(media, key, value)

    db.commit()
    db.refresh(media)

    return media_to_dict(media)
