from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..database import get_db

router = APIRouter(prefix="/api/media", tags=["media"])


# Mock media data
mock_media = [
    {"id": 1, "name": "banner.jpg", "url": "https://picsum.photos/400/300?random=1", "type": "image", "size": 245000, "mime_type": "image/jpeg", "created_at": "2024-01-15T10:00:00Z"},
    {"id": 2, "name": "profile.png", "url": "https://picsum.photos/400/300?random=2", "type": "image", "size": 128000, "mime_type": "image/png", "created_at": "2024-01-14T14:00:00Z"},
    {"id": 3, "name": "post-1.jpg", "url": "https://picsum.photos/400/300?random=3", "type": "image", "size": 512000, "mime_type": "image/jpeg", "created_at": "2024-01-13T09:00:00Z"},
    {"id": 4, "name": "thumbnail.jpg", "url": "https://picsum.photos/400/300?random=4", "type": "image", "size": 89000, "mime_type": "image/jpeg", "created_at": "2024-01-12T16:00:00Z"},
    {"id": 5, "name": "cover.png", "url": "https://picsum.photos/400/300?random=5", "type": "image", "size": 324000, "mime_type": "image/png", "created_at": "2024-01-11T11:00:00Z"},
    {"id": 6, "name": "story.jpg", "url": "https://picsum.photos/400/300?random=6", "type": "image", "size": 198000, "mime_type": "image/jpeg", "created_at": "2024-01-10T08:00:00Z"},
]


@router.get("", response_model=List[dict])
def get_media(
    media_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all media files with optional type filter"""
    media = mock_media

    if media_type:
        media = [m for m in media if m["type"] == media_type]

    return media


@router.get("/stats")
def get_media_stats(db: Session = Depends(get_db)):
    """Get media library statistics"""
    total_size = sum(m["size"] for m in mock_media)
    return {
        "total_files": len(mock_media),
        "total_size": total_size,
        "images": len([m for m in mock_media if m["type"] == "image"]),
        "videos": len([m for m in mock_media if m["type"] == "video"]),
    }


@router.get("/{media_id}", response_model=dict)
def get_media_item(media_id: int, db: Session = Depends(get_db)):
    """Get a single media item by ID"""
    media = next((m for m in mock_media if m["id"] == media_id), None)
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")
    return media


@router.post("", response_model=dict)
def upload_media(
    name: str,
    url: str,
    media_type: str = "image",
    size: int = 0,
    mime_type: str = "image/jpeg",
    db: Session = Depends(get_db)
):
    """Upload a new media file (mock - just creates record)"""
    new_media = {
        "id": len(mock_media) + 1,
        "name": name,
        "url": url,
        "type": media_type,
        "size": size,
        "mime_type": mime_type,
        "created_at": datetime.utcnow().isoformat(),
    }
    mock_media.append(new_media)
    return new_media


@router.delete("/{media_id}")
def delete_media(media_id: int, db: Session = Depends(get_db)):
    """Delete a media file"""
    global mock_media
    media = next((m for m in mock_media if m["id"] == media_id), None)
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    mock_media = [m for m in mock_media if m["id"] != media_id]
    return {"message": "Media deleted"}


@router.patch("/{media_id}", response_model=dict)
def update_media(media_id: int, update: dict, db: Session = Depends(get_db)):
    """Update media metadata (e.g., rename)"""
    media = next((m for m in mock_media if m["id"] == media_id), None)
    if not media:
        raise HTTPException(status_code=404, detail="Media not found")

    for key, value in update.items():
        if key in ["name"] and value is not None:
            media[key] = value

    return media
