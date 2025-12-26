from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..database import get_db

router = APIRouter(prefix="/api/templates", tags=["templates"])


# Mock data
mock_templates = [
    {
        "id": 1,
        "name": "Flirty Response",
        "category": "responses",
        "content": "Hey cutie! Thanks for the love 💕 You always know how to make my day...",
        "uses": 234,
        "is_favorite": True,
        "created_at": "2024-01-10T10:00:00Z",
    },
    {
        "id": 2,
        "name": "New Post Alert",
        "category": "captions",
        "content": "🔥 Something spicy just dropped! Check my latest...",
        "uses": 189,
        "is_favorite": True,
        "created_at": "2024-01-09T14:00:00Z",
    },
    {
        "id": 3,
        "name": "Thank You Message",
        "category": "greetings",
        "content": "You are amazing! Thank you so much for your support...",
        "uses": 156,
        "is_favorite": False,
        "created_at": "2024-01-08T09:00:00Z",
    },
    {
        "id": 4,
        "name": "Special Offer",
        "category": "promotions",
        "content": "🎉 Exclusive deal just for you! For the next 24 hours...",
        "uses": 98,
        "is_favorite": False,
        "created_at": "2024-01-07T16:00:00Z",
    },
    {
        "id": 5,
        "name": "Welcome New Fan",
        "category": "greetings",
        "content": "Welcome to the crew! 🎉 So happy to have you here...",
        "uses": 312,
        "is_favorite": True,
        "created_at": "2024-01-06T11:00:00Z",
    },
]


@router.get("", response_model=List[dict])
def get_templates(
    category: Optional[str] = None,
    favorites_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get all templates with optional filtering"""
    templates = mock_templates

    if category:
        templates = [t for t in templates if t["category"] == category]
    if favorites_only:
        templates = [t for t in templates if t["is_favorite"]]

    return templates


@router.get("/{template_id}", response_model=dict)
def get_template(template_id: int, db: Session = Depends(get_db)):
    """Get a single template by ID"""
    template = next((t for t in mock_templates if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@router.post("", response_model=dict)
def create_template(template: dict, db: Session = Depends(get_db)):
    """Create a new template"""
    new_template = {
        "id": len(mock_templates) + 1,
        **template,
        "uses": 0,
        "is_favorite": False,
        "created_at": datetime.utcnow().isoformat(),
    }
    mock_templates.append(new_template)
    return new_template


@router.patch("/{template_id}", response_model=dict)
def update_template(template_id: int, template_update: dict, db: Session = Depends(get_db)):
    """Update a template"""
    template = next((t for t in mock_templates if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    for key, value in template_update.items():
        if value is not None:
            template[key] = value

    return template


@router.delete("/{template_id}")
def delete_template(template_id: int, db: Session = Depends(get_db)):
    """Delete a template"""
    global mock_templates
    template = next((t for t in mock_templates if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    mock_templates = [t for t in mock_templates if t["id"] != template_id]
    return {"message": "Template deleted"}


@router.post("/{template_id}/favorite")
def toggle_favorite(template_id: int, db: Session = Depends(get_db)):
    """Toggle favorite status of a template"""
    template = next((t for t in mock_templates if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    template["is_favorite"] = not template["is_favorite"]
    return template


@router.post("/{template_id}/use")
def use_template(template_id: int, db: Session = Depends(get_db)):
    """Increment use count of a template"""
    template = next((t for t in mock_templates if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    template["uses"] += 1
    return template
