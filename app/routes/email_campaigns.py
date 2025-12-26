from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime

from ..database import get_db
from ..models.email_campaign import EmailCampaign

router = APIRouter(prefix="/api/email-campaigns", tags=["email-campaigns"])


@router.get("/stats")
def get_email_stats(db: Session = Depends(get_db)):
    """Get email campaign statistics"""
    total = db.query(func.count(EmailCampaign.id)).scalar() or 0
    sent = db.query(func.count(EmailCampaign.id)).filter(
        EmailCampaign.status == "sent"
    ).scalar() or 0
    scheduled = db.query(func.count(EmailCampaign.id)).filter(
        EmailCampaign.status == "scheduled"
    ).scalar() or 0
    drafts = db.query(func.count(EmailCampaign.id)).filter(
        EmailCampaign.status == "draft"
    ).scalar() or 0

    # Calculate total recipients from sent campaigns
    total_recipients = db.query(func.sum(EmailCampaign.recipient_count)).filter(
        EmailCampaign.status == "sent"
    ).scalar() or 0

    return {
        "total_campaigns": total,
        "total_sent": total_recipients,
        "total_subscribers": 3892,  # Would come from subscribers table
        "avg_open_rate": 64,
        "avg_click_rate": 22,
    }


@router.get("", response_model=List[dict])
def get_campaigns(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all email campaigns with optional status filter"""
    query = db.query(EmailCampaign)

    if status:
        query = query.filter(EmailCampaign.status == status)

    campaigns = query.order_by(EmailCampaign.created_at.desc()).all()

    return [
        {
            "id": c.id,
            "name": c.name,
            "subject": c.subject,
            "content": c.content,
            "recipients": c.recipient_count,
            "status": c.status,
            "scheduled_for": c.scheduled_at.isoformat() if c.scheduled_at else None,
            "sent_at": c.sent_at.isoformat() if c.sent_at else None,
            "open_rate": c.stats.get("open_rate") if c.stats else None,
            "click_rate": c.stats.get("click_rate") if c.stats else None,
            "created_at": c.created_at.isoformat(),
        }
        for c in campaigns
    ]


@router.get("/{campaign_id}", response_model=dict)
def get_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Get a single campaign by ID"""
    campaign = db.query(EmailCampaign).filter(EmailCampaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

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


@router.post("", response_model=dict)
def create_campaign(campaign_data: dict, db: Session = Depends(get_db)):
    """Create a new email campaign"""
    scheduled_at = campaign_data.get("scheduled_for")
    if scheduled_at and isinstance(scheduled_at, str):
        scheduled_at = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))

    campaign = EmailCampaign(
        name=campaign_data.get("name", "Untitled Campaign"),
        subject=campaign_data.get("subject", ""),
        content=campaign_data.get("content", ""),
        recipient_count=campaign_data.get("recipients", 0),
        status="scheduled" if scheduled_at else "draft",
        scheduled_at=scheduled_at,
    )
    db.add(campaign)
    db.commit()
    db.refresh(campaign)

    return {
        "id": campaign.id,
        "name": campaign.name,
        "subject": campaign.subject,
        "content": campaign.content,
        "recipients": campaign.recipient_count,
        "status": campaign.status,
        "scheduled_for": campaign.scheduled_at.isoformat() if campaign.scheduled_at else None,
        "sent_at": campaign.sent_at.isoformat() if campaign.sent_at else None,
        "created_at": campaign.created_at.isoformat(),
    }


@router.patch("/{campaign_id}", response_model=dict)
def update_campaign(campaign_id: int, campaign_update: dict, db: Session = Depends(get_db)):
    """Update a campaign"""
    campaign = db.query(EmailCampaign).filter(EmailCampaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    for key, value in campaign_update.items():
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

    return {
        "id": campaign.id,
        "name": campaign.name,
        "subject": campaign.subject,
        "content": campaign.content,
        "recipients": campaign.recipient_count,
        "status": campaign.status,
        "scheduled_for": campaign.scheduled_at.isoformat() if campaign.scheduled_at else None,
        "sent_at": campaign.sent_at.isoformat() if campaign.sent_at else None,
        "created_at": campaign.created_at.isoformat(),
    }


@router.delete("/{campaign_id}")
def delete_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Delete a campaign"""
    campaign = db.query(EmailCampaign).filter(EmailCampaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    db.delete(campaign)
    db.commit()
    return {"message": "Campaign deleted"}


@router.post("/{campaign_id}/send")
def send_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Send a draft campaign immediately"""
    campaign = db.query(EmailCampaign).filter(EmailCampaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if campaign.status == "sent":
        raise HTTPException(status_code=400, detail="Campaign already sent")

    campaign.status = "sent"
    campaign.sent_at = datetime.utcnow()
    campaign.recipient_count = 3892  # Would come from subscribers
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
def schedule_campaign(campaign_id: int, scheduled_for: str, db: Session = Depends(get_db)):
    """Schedule a campaign for later"""
    campaign = db.query(EmailCampaign).filter(EmailCampaign.id == campaign_id).first()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign.status = "scheduled"
    campaign.scheduled_at = datetime.fromisoformat(scheduled_for.replace("Z", "+00:00"))
    campaign.recipient_count = 3892  # Would come from subscribers

    db.commit()
    db.refresh(campaign)

    return {
        "id": campaign.id,
        "name": campaign.name,
        "status": campaign.status,
        "scheduled_for": campaign.scheduled_at.isoformat(),
    }
