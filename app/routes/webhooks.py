"""
Webhook API Routes
==================
Endpoints for managing webhook subscriptions and event delivery.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import hmac
import hashlib
import json
import logging
import asyncio

import requests

from ..database import get_db
from ..models.webhook import Webhook
from ..auth import get_required_user
from ..models.user import User

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])
logger = logging.getLogger(__name__)


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class WebhookCreate(BaseModel):
    name: Optional[str] = None
    url: str
    events: List[str]


class WebhookUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    events: Optional[List[str]] = None
    active: Optional[bool] = None


class WebhookResponse(BaseModel):
    id: int
    name: Optional[str]
    url: str
    events: List[str]
    active: bool
    last_triggered_at: Optional[str]
    last_status_code: Optional[int]
    failure_count: int
    created_at: str

    class Config:
        from_attributes = True


# ============================================================
# WEBHOOK DELIVERY
# ============================================================

def sign_payload(payload: dict, secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload"""
    payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
    signature = hmac.new(
        secret.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    return f"sha256={signature}"


async def deliver_webhook(webhook: Webhook, event: str, data: dict, db: Session):
    """Deliver a webhook with retry logic"""
    payload = {
        "event": event,
        "data": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "webhook_id": webhook.id,
    }

    signature = sign_payload(payload, webhook.secret)
    headers = {
        "Content-Type": "application/json",
        "X-NubHQ-Signature": signature,
        "X-NubHQ-Event": event,
    }

    max_retries = 3
    backoff = 2

    for attempt in range(max_retries):
        try:
            response = requests.post(
                webhook.url,
                json=payload,
                headers=headers,
                timeout=10
            )

            # Update webhook status
            webhook.last_triggered_at = datetime.now(timezone.utc)
            webhook.last_status_code = response.status_code

            if response.status_code >= 200 and response.status_code < 300:
                webhook.failure_count = 0
                db.commit()
                logger.info(f"Webhook delivered: {webhook.id} -> {event}")
                return True
            else:
                webhook.failure_count += 1
                db.commit()
                logger.warning(f"Webhook failed: {webhook.id} status={response.status_code}")

        except Exception as e:
            webhook.failure_count += 1
            db.commit()
            logger.error(f"Webhook error: {webhook.id} error={str(e)}")

        # Retry with exponential backoff
        if attempt < max_retries - 1:
            await asyncio.sleep(backoff ** attempt)

    # Disable webhook after too many failures
    if webhook.failure_count >= 10:
        webhook.active = False
        db.commit()
        logger.warning(f"Webhook disabled due to failures: {webhook.id}")

    return False


async def trigger_webhooks(event: str, data: dict, db: Session, user_id: Optional[int] = None):
    """Trigger all active webhooks subscribed to an event"""
    query = db.query(Webhook).filter(
        Webhook.active.is_(True),
        Webhook.events.contains([event])
    )

    if user_id:
        query = query.filter(
            (Webhook.user_id == user_id) | (Webhook.user_id == None)
        )

    webhooks = query.all()

    for webhook in webhooks:
        await deliver_webhook(webhook, event, data, db)


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("", response_model=List[WebhookResponse])
async def list_webhooks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """List all webhooks for the current user"""
    webhooks = db.query(Webhook).filter(
        (Webhook.user_id == current_user.id) | (Webhook.user_id == None)
    ).all()

    return [
        WebhookResponse(
            id=w.id,
            name=w.name,
            url=w.url,
            events=w.events,
            active=w.active,
            last_triggered_at=w.last_triggered_at.isoformat() if w.last_triggered_at else None,
            last_status_code=w.last_status_code,
            failure_count=w.failure_count,
            created_at=w.created_at.isoformat(),
        )
        for w in webhooks
    ]


@router.post("", status_code=201)
async def create_webhook(
    webhook: WebhookCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """Register a new webhook"""
    # Validate events
    invalid_events = [e for e in webhook.events if e not in Webhook.EVENTS]
    if invalid_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid events: {invalid_events}. Valid events: {Webhook.EVENTS}"
        )

    new_webhook = Webhook(
        user_id=current_user.id,
        name=webhook.name,
        url=webhook.url,
        events=webhook.events,
    )

    db.add(new_webhook)
    db.commit()
    db.refresh(new_webhook)

    return {
        "status": "ok",
        "message": "Webhook created",
        "webhook": new_webhook.to_dict(include_secret=True)
    }


@router.get("/events")
async def list_webhook_events():
    """List all available webhook events"""
    return {
        "events": Webhook.EVENTS,
        "descriptions": {
            "job.started": "A processing job has started",
            "job.progress": "Job progress update (25%, 50%, 75%)",
            "job.completed": "A processing job completed successfully",
            "job.failed": "A processing job failed",
            "review.pending": "A video is pending manual review",
            "approval.approved": "Content was approved",
            "approval.rejected": "Content was rejected",
        }
    }


@router.get("/{webhook_id}")
async def get_webhook(
    webhook_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """Get a specific webhook"""
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        (Webhook.user_id == current_user.id) | (Webhook.user_id == None)
    ).first()

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    return webhook.to_dict()


@router.put("/{webhook_id}")
async def update_webhook(
    webhook_id: int,
    update: WebhookUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """Update a webhook"""
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.user_id == current_user.id
    ).first()

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    if update.name is not None:
        webhook.name = update.name
    if update.url is not None:
        webhook.url = update.url
    if update.events is not None:
        invalid_events = [e for e in update.events if e not in Webhook.EVENTS]
        if invalid_events:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid events: {invalid_events}"
            )
        webhook.events = update.events
    if update.active is not None:
        webhook.active = update.active
        if update.active:
            webhook.failure_count = 0  # Reset on re-enable

    db.commit()
    db.refresh(webhook)

    return {
        "status": "ok",
        "message": "Webhook updated",
        "webhook": webhook.to_dict()
    }


@router.delete("/{webhook_id}")
async def delete_webhook(
    webhook_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """Delete a webhook"""
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.user_id == current_user.id
    ).first()

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    db.delete(webhook)
    db.commit()

    return {"status": "ok", "message": f"Webhook {webhook_id} deleted"}


@router.post("/{webhook_id}/test")
async def test_webhook(
    webhook_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """Send a test event to a webhook"""
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        (Webhook.user_id == current_user.id) | (Webhook.user_id == None)
    ).first()

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    test_data = {
        "test": True,
        "message": "This is a test webhook from NubHQ",
        "webhook_id": webhook_id,
    }

    # Deliver synchronously for test so we can return the result
    success = await deliver_webhook(webhook, "test", test_data, db)

    return {
        "status": "ok" if success else "failed",
        "message": "Test webhook delivered" if success else "Webhook delivery failed",
        "last_status_code": webhook.last_status_code
    }


@router.post("/{webhook_id}/reset")
async def reset_webhook(
    webhook_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """Reset a webhook's failure count and re-enable it"""
    webhook = db.query(Webhook).filter(
        Webhook.id == webhook_id,
        Webhook.user_id == current_user.id
    ).first()

    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")

    webhook.failure_count = 0
    webhook.active = True
    db.commit()

    return {
        "status": "ok",
        "message": f"Webhook {webhook_id} reset and re-enabled"
    }
