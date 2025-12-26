"""
Templates routes for CRUD operations on content templates.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from ..database import get_db
from ..models.template import Template
from ..models.user import User
from ..auth import get_required_user

router = APIRouter(prefix="/api/templates", tags=["templates"])


class TemplateCreate(BaseModel):
    """Schema for creating a template."""
    name: str
    content: str
    category: str = "engagement"
    platform: Optional[str] = None
    hashtags: Optional[str] = None


class TemplateUpdate(BaseModel):
    """Schema for updating a template."""
    name: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    platform: Optional[str] = None
    hashtags: Optional[str] = None
    is_favorite: Optional[bool] = None


def template_to_dict(template: Template) -> dict:
    """Convert a Template model to a dictionary response."""
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


@router.get("", response_model=List[dict])
def get_templates(
    category: Optional[str] = None,
    favorites_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get all templates for the current user with optional filtering."""
    query = db.query(Template).filter(Template.user_id == current_user.id)

    if category:
        query = query.filter(Template.category == category)
    if favorites_only:
        query = query.filter(Template.is_favorite == True)

    templates = query.order_by(Template.created_at.desc()).all()
    return [template_to_dict(t) for t in templates]


@router.get("/{template_id}", response_model=dict)
def get_template(
    template_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get a single template by ID (must belong to current user)."""
    template = db.query(Template).filter(
        Template.id == template_id,
        Template.user_id == current_user.id
    ).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template_to_dict(template)


@router.post("", response_model=dict)
def create_template(
    template_data: TemplateCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Create a new template for the current user."""
    template = Template(
        user_id=current_user.id,
        name=template_data.name,
        content=template_data.content,
        category=template_data.category,
        platform=template_data.platform,
        hashtags=template_data.hashtags,
    )
    db.add(template)
    db.commit()
    db.refresh(template)

    return template_to_dict(template)


@router.patch("/{template_id}", response_model=dict)
def update_template(
    template_id: int,
    template_update: TemplateUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Update a template (must belong to current user)."""
    template = db.query(Template).filter(
        Template.id == template_id,
        Template.user_id == current_user.id
    ).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    update_data = template_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(template, key, value)

    db.commit()
    db.refresh(template)

    return template_to_dict(template)


@router.delete("/{template_id}")
def delete_template(
    template_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Delete a template (must belong to current user)."""
    template = db.query(Template).filter(
        Template.id == template_id,
        Template.user_id == current_user.id
    ).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    db.delete(template)
    db.commit()
    return {"message": "Template deleted"}


@router.post("/{template_id}/favorite")
def toggle_favorite(
    template_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Toggle favorite status of a template (must belong to current user)."""
    template = db.query(Template).filter(
        Template.id == template_id,
        Template.user_id == current_user.id
    ).first()
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
def use_template(
    template_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Increment usage count when template is used (must belong to current user)."""
    template = db.query(Template).filter(
        Template.id == template_id,
        Template.user_id == current_user.id
    ).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    template.use_count += 1
    db.commit()
    db.refresh(template)

    return template_to_dict(template)
