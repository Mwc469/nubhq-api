"""
Email campaigns routes for CRUD operations on email campaigns.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel

from ..database import get_db
from ..models.email_campaign import EmailCampaign
from ..models.user import User
from ..auth import get_required_user

router = APIRouter(prefix="/api/email-campaigns", tags=["email-campaigns"])


class CampaignCreate(BaseModel):
    """Schema for creating an email campaign."""
    name: str
    subject: str = ""
    content: str = ""
    recipients: int = 0
    scheduled_for: Optional[str] = None


class CampaignUpdate(BaseModel):
    """Schema for updating an email campaign."""
    name: Optional[str] = None
    subject: Optional[str] = None
    content: Optional[str] = None
    recipients: Optional[int] = None
    status: Optional[str] = None
    scheduled_for: Optional[str] = None


def campaign_to_dict(campaign: EmailCampaign) -> dict:
    """Convert an EmailCampaign model to a dictionary response."""
    return {
        "id": campaign.id,
        "name": campaign.name,
        "subject": campaign.subject,
        "content": campaign.content,
        "recipients": campaign.recipient_count,
        "status": campaign.status,
        "scheduled_for": campaign.scheduled_at.isoformat() if campaign.scheduled_at else None,
        "sent_at": campaign.sent_at.isoformat() if campaign.sent_at else None,
        "open_rate": campaign.stats.get("open_rate") if campaign.stats else None,
        "click_rate": campaign.stats.get("click_rate") if campaign.stats else None,
        "created_at": campaign.created_at.isoformat(),
    }


@router.get("/stats")
def get_email_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get email campaign statistics for the current user."""
    base_query = db.query(EmailCampaign).filter(EmailCampaign.user_id == current_user.id)

    total = base_query.count()
    sent = base_query.filter(EmailCampaign.status == "sent").count()
    scheduled = base_query.filter(EmailCampaign.status == "scheduled").count()
    drafts = base_query.filter(EmailCampaign.status == "draft").count()

    total_recipients = db.query(func.sum(EmailCampaign.recipient_count)).filter(
        EmailCampaign.user_id == current_user.id,
        EmailCampaign.status == "sent"
    ).scalar() or 0

    return {
        "total_campaigns": total,
        "sent": sent,
        "scheduled": scheduled,
        "drafts": drafts,
        "total_sent": total_recipients,
        "total_subscribers": 0,
        "avg_open_rate": 0,
        "avg_click_rate": 0,
    }


@router.get("", response_model=List[dict])
def get_campaigns(
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get all email campaigns for the current user with optional status filter."""
    query = db.query(EmailCampaign).filter(EmailCampaign.user_id == current_user.id)

    if status:
        query = query.filter(EmailCampaign.status == status)

    campaigns = query.order_by(EmailCampaign.created_at.desc()).all()
    return [campaign_to_dict(c) for c in campaigns]


@router.get("/{campaign_id}", response_model=dict)
def get_campaign(
    campaign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get a single campaign by ID (must belong to current user)."""
    campaign = db.query(EmailCampaign).filter(
        EmailCampaign.id == campaign_id,
        EmailCampaign.user_id == current_user.id
    ).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    return campaign_to_dict(campaign)


@router.post("", response_model=dict)
def create_campaign(
    campaign_data: CampaignCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Create a new email campaign for the current user."""
    scheduled_at = None
    if campaign_data.scheduled_for:
        scheduled_at = datetime.fromisoformat(campaign_data.scheduled_for.replace("Z", "+00:00"))

    campaign = EmailCampaign(
        user_id=current_user.id,
        name=campaign_data.name,
        subject=campaign_data.subject,
        content=campaign_data.content,
        recipient_count=campaign_data.recipients,
        status="scheduled" if scheduled_at else "draft",
        scheduled_at=scheduled_at,
    )
    db.add(campaign)
    db.commit()
    db.refresh(campaign)

    return campaign_to_dict(campaign)


@router.patch("/{campaign_id}", response_model=dict)
def update_campaign(
    campaign_id: int,
    campaign_update: CampaignUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Update a campaign (must belong to current user)."""
    campaign = db.query(EmailCampaign).filter(
        EmailCampaign.id == campaign_id,
        EmailCampaign.user_id == current_user.id
    ).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    update_data = campaign_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            if key == "scheduled_for":
                if isinstance(value, str):
                    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
                campaign.scheduled_at = value
            elif key == "recipients":
                campaign.recipient_count = value
            elif hasattr(campaign, key):
                setattr(campaign, key, value)

    db.commit()
    db.refresh(campaign)

    return campaign_to_dict(campaign)


@router.delete("/{campaign_id}")
def delete_campaign(
    campaign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Delete a campaign (must belong to current user)."""
    campaign = db.query(EmailCampaign).filter(
        EmailCampaign.id == campaign_id,
        EmailCampaign.user_id == current_user.id
    ).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    db.delete(campaign)
    db.commit()
    return {"message": "Campaign deleted"}


@router.post("/{campaign_id}/send")
def send_campaign(
    campaign_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Send a draft campaign immediately (must belong to current user)."""
    campaign = db.query(EmailCampaign).filter(
        EmailCampaign.id == campaign_id,
        EmailCampaign.user_id == current_user.id
    ).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if campaign.status == "sent":
        raise HTTPException(status_code=400, detail="Campaign already sent")

    campaign.status = "sent"
    campaign.sent_at = datetime.now(timezone.utc)
    campaign.stats = {
        "opens": 0,
        "clicks": 0,
        "bounces": 0,
        "unsubscribes": 0,
        "open_rate": 0,
        "click_rate": 0,
    }

    db.commit()
    db.refresh(campaign)

    return {
        "id": campaign.id,
        "name": campaign.name,
        "status": campaign.status,
        "sent_at": campaign.sent_at.isoformat(),
        "message": "Campaign sent successfully",
    }


@router.post("/{campaign_id}/schedule")
def schedule_campaign(
    campaign_id: int,
    scheduled_for: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Schedule a campaign for later (must belong to current user)."""
    campaign = db.query(EmailCampaign).filter(
        EmailCampaign.id == campaign_id,
        EmailCampaign.user_id == current_user.id
    ).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign.status = "scheduled"
    campaign.scheduled_at = datetime.fromisoformat(scheduled_for.replace("Z", "+00:00"))

    db.commit()
    db.refresh(campaign)

    return {
        "id": campaign.id,
        "name": campaign.name,
        "status": campaign.status,
        "scheduled_for": campaign.scheduled_at.isoformat(),
    }
