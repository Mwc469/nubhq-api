from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional

from ..database import get_db
from ..models.media import Media

router = APIRouter(prefix="/api/media", tags=["media"])


@router.get("", response_model=List[dict])
def get_media(
    media_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all media files with optional type filter"""
    query = db.query(Media)

    if media_type:
        query = query.filter(Media.media_type == media_type)

    media_items = query.order_by(Media.created_at.desc()).all()

    return [
        {
            "id": m.id,
            "name": m.name,
            "url": m.url,
            "type": m.media_type,
            "size": m.size,
            "mime_type": m.mime_type,
            "width": m.width,
            "height": m.height,
            "created_at": m.created_at.isoformat(),
        }
        for m in media_items
    ]


@router.get("/stats")
def get_media_stats(db: Session = Depends(get_db)):
    """Get media library statistics"""
    total_files = db.query(func.count(Media.id)).scalar() or 0
    total_size = db.query(func.sum(Media.size)).scalar() or 0
    images = db.query(func.count(Media.id)).filter(Media.media_type == "image").scalar() or 0
    videos = db.query(func.count(Media.id)).filter(Media.media_type == "video").scalar() or 0

    return {
        "total_files": total_files,
        "total_size": total_size,
        "images": images,
        "videos": videos,
    }


@router.get("/{media_id}", response_model=dict)
def get_media_item(media_id: int, db: Session = Depends(get_db)):
    """Get a single media item by ID"""
    media = db.query(Media).filter(Media.id == media_id).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

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


@router.post("", response_model=dict)
def upload_media(
    name: str,
    url: str,
    media_type: str = "image",
    size: int = 0,
    mime_type: str = "image/jpeg",
    width: Optional[int] = None,
    height: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Upload a new media file (creates record with URL)"""
    media = Media(
        name=name,
        url=url,
        media_type=media_type,
        size=size,
        mime_type=mime_type,
        width=width,
        height=height,
    )
    db.add(media)
    db.commit()
    db.refresh(media)

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


@router.delete("/{media_id}")
def delete_media(media_id: int, db: Session = Depends(get_db)):
    """Delete a media file"""
    media = db.query(Media).filter(Media.id == media_id).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    db.delete(media)
    db.commit()
    return {"message": "Media deleted"}


@router.patch("/{media_id}", response_model=dict)
def update_media(media_id: int, update: dict, db: Session = Depends(get_db)):
    """Update media metadata (e.g., rename)"""
    media = db.query(Media).filter(Media.id == media_id).first()
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    if "name" in update and update["name"] is not None:
        media.name = update["name"]

    db.commit()
    db.refresh(media)

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
