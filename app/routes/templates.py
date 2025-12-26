from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from ..database import get_db
from ..models.template import Template

router = APIRouter(prefix="/api/templates", tags=["templates"])


@router.get("", response_model=List[dict])
def get_templates(
    category: Optional[str] = None,
    favorites_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get all templates with optional filtering"""
    query = db.query(Template)

    if category:
        query = query.filter(Template.category == category)
    if favorites_only:
        query = query.filter(Template.is_favorite == True)

    templates = query.order_by(Template.created_at.desc()).all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "content": t.content,
            "category": t.category,
            "platform": t.platform,
            "hashtags": t.hashtags,
            "is_favorite": t.is_favorite,
            "uses": t.use_count,
            "created_at": t.created_at.isoformat(),
        }
        for t in templates
    ]


@router.get("/{template_id}", response_model=dict)
def get_template(template_id: int, db: Session = Depends(get_db)):
    """Get a single template by ID"""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "id": template.id,
        "name": template.name,
        "content": template.content,
        "category": template.category,
        "platform": template.platform,
        "hashtags": template.hashtags,
        "is_favorite": template.is_favorite,
        "uses": template.use_count,
        "created_at": template.created_at.isoformat(),
    }


@router.post("", response_model=dict)
def create_template(template_data: dict, db: Session = Depends(get_db)):
    """Create a new template"""
    template = Template(
        name=template_data.get("name", "Untitled Template"),
        content=template_data.get("content", ""),
        category=template_data.get("category", "engagement"),
        platform=template_data.get("platform"),
        hashtags=template_data.get("hashtags", ""),
    )
    db.add(template)
    db.commit()
    db.refresh(template)

    return {
        "id": template.id,
        "name": template.name,
        "content": template.content,
        "category": template.category,
        "platform": template.platform,
        "hashtags": template.hashtags,
        "is_favorite": template.is_favorite,
        "uses": template.use_count,
        "created_at": template.created_at.isoformat(),
    }


@router.patch("/{template_id}", response_model=dict)
def update_template(template_id: int, template_update: dict, db: Session = Depends(get_db)):
    """Update a template"""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    for key, value in template_update.items():
        if value is not None and hasattr(template, key):
            setattr(template, key, value)

    db.commit()
    db.refresh(template)

    return {
        "id": template.id,
        "name": template.name,
        "content": template.content,
        "category": template.category,
        "platform": template.platform,
        "hashtags": template.hashtags,
        "is_favorite": template.is_favorite,
        "uses": template.use_count,
        "created_at": template.created_at.isoformat(),
    }


@router.delete("/{template_id}")
def delete_template(template_id: int, db: Session = Depends(get_db)):
    """Delete a template"""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    db.delete(template)
    db.commit()
    return {"message": "Template deleted"}


@router.post("/{template_id}/favorite")
def toggle_favorite(template_id: int, db: Session = Depends(get_db)):
    """Toggle favorite status of a template"""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    template.is_favorite = not template.is_favorite
    db.commit()
    db.refresh(template)

    return {
        "id": template.id,
        "is_favorite": template.is_favorite,
    }


@router.post("/{template_id}/use")
def use_template(template_id: int, db: Session = Depends(get_db)):
    """Increment usage count when template is used"""
    template = db.query(Template).filter(Template.id == template_id).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    template.use_count += 1
    db.commit()
    db.refresh(template)

    return {
        "id": template.id,
        "name": template.name,
        "content": template.content,
        "category": template.category,
        "platform": template.platform,
        "hashtags": template.hashtags,
        "is_favorite": template.is_favorite,
        "uses": template.use_count,
        "created_at": template.created_at.isoformat(),
    }
