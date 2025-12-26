from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from ..database import get_db

router = APIRouter(prefix="/api/email-campaigns", tags=["email-campaigns"])


# Mock data
mock_campaigns = [
    {
        "id": 1,
        "name": "Weekly Newsletter",
        "subject": "This Week's Exclusive Content",
        "status": "sent",
        "recipients": 2450,
        "open_rate": 68,
        "click_rate": 24,
        "sent_at": (datetime.utcnow() - timedelta(days=2)).isoformat(),
        "created_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
    },
    {
        "id": 2,
        "name": "New Content Alert",
        "subject": "Something Special Just Dropped!",
        "status": "scheduled",
        "recipients": 1890,
        "open_rate": None,
        "click_rate": None,
        "scheduled_for": (datetime.utcnow() + timedelta(days=1, hours=9)).isoformat(),
        "created_at": (datetime.utcnow() - timedelta(hours=5)).isoformat(),
    },
    {
        "id": 3,
        "name": "Exclusive Offer",
        "subject": "24 Hours Only - Don't Miss Out!",
        "status": "draft",
        "recipients": 0,
        "open_rate": None,
        "click_rate": None,
        "created_at": (datetime.utcnow() - timedelta(hours=5)).isoformat(),
    },
]

mock_stats = {
    "total_sent": 12400,
    "total_subscribers": 3892,
    "avg_open_rate": 64,
    "avg_click_rate": 22,
}


@router.get("/stats")
def get_email_stats(db: Session = Depends(get_db)):
    """Get email campaign statistics"""
    return mock_stats


@router.get("", response_model=List[dict])
def get_campaigns(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all email campaigns with optional status filter"""
    campaigns = mock_campaigns

    if status:
        campaigns = [c for c in campaigns if c["status"] == status]

    return campaigns


@router.get("/{campaign_id}", response_model=dict)
def get_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Get a single campaign by ID"""
    campaign = next((c for c in mock_campaigns if c["id"] == campaign_id), None)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return campaign


@router.post("", response_model=dict)
def create_campaign(campaign: dict, db: Session = Depends(get_db)):
    """Create a new email campaign"""
    new_campaign = {
        "id": len(mock_campaigns) + 1,
        **campaign,
        "status": "draft",
        "recipients": 0,
        "open_rate": None,
        "click_rate": None,
        "created_at": datetime.utcnow().isoformat(),
    }
    mock_campaigns.append(new_campaign)
    return new_campaign


@router.patch("/{campaign_id}", response_model=dict)
def update_campaign(campaign_id: int, campaign_update: dict, db: Session = Depends(get_db)):
    """Update a campaign"""
    campaign = next((c for c in mock_campaigns if c["id"] == campaign_id), None)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    for key, value in campaign_update.items():
        if value is not None:
            campaign[key] = value

    return campaign


@router.delete("/{campaign_id}")
def delete_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Delete a campaign"""
    global mock_campaigns
    campaign = next((c for c in mock_campaigns if c["id"] == campaign_id), None)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    mock_campaigns = [c for c in mock_campaigns if c["id"] != campaign_id]
    return {"message": "Campaign deleted"}


@router.post("/{campaign_id}/send")
def send_campaign(campaign_id: int, db: Session = Depends(get_db)):
    """Send a draft campaign immediately"""
    campaign = next((c for c in mock_campaigns if c["id"] == campaign_id), None)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    if campaign["status"] not in ["draft", "scheduled"]:
        raise HTTPException(status_code=400, detail="Campaign already sent")

    campaign["status"] = "sent"
    campaign["sent_at"] = datetime.utcnow().isoformat()
    campaign["recipients"] = mock_stats["total_subscribers"]

    return campaign


@router.post("/{campaign_id}/schedule")
def schedule_campaign(campaign_id: int, scheduled_for: str, db: Session = Depends(get_db)):
    """Schedule a campaign for later"""
    campaign = next((c for c in mock_campaigns if c["id"] == campaign_id), None)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign["status"] = "scheduled"
    campaign["scheduled_for"] = scheduled_for
    campaign["recipients"] = mock_stats["total_subscribers"]

    return campaign
