from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from ..database import get_db
from ..models.approval import Approval
from ..models.fan_message import FanMessage
from ..models.scheduled_post import ScheduledPost
from pydantic import BaseModel


class SearchResult(BaseModel):
    id: int
    type: str
    title: str
    subtitle: str
    url: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int


router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(q: str, db: Session = Depends(get_db)):
    if not q or len(q) < 2:
        return SearchResponse(results=[], total=0)

    results = []
    query = f"%{q.lower()}%"

    # Search approvals
    approvals = db.query(Approval).filter(
        Approval.content.ilike(query) | Approval.recipient.ilike(query)
    ).limit(5).all()

    for item in approvals:
        results.append(SearchResult(
            id=item.id,
            type="approval",
            title=f"Message to {item.recipient}",
            subtitle=item.content[:50] + "..." if len(item.content) > 50 else item.content,
            url="/approvals"
        ))

    # Search fan messages
    messages = db.query(FanMessage).filter(
        FanMessage.content.ilike(query) | FanMessage.sender_name.ilike(query)
    ).limit(5).all()

    for item in messages:
        results.append(SearchResult(
            id=item.id,
            type="fan_mail",
            title=f"From {item.sender_name}",
            subtitle=item.content[:50] + "..." if len(item.content) > 50 else item.content,
            url="/fan-mail"
        ))

    # Search scheduled posts
    posts = db.query(ScheduledPost).filter(
        ScheduledPost.content.ilike(query) | ScheduledPost.platform.ilike(query)
    ).limit(5).all()

    for item in posts:
        results.append(SearchResult(
            id=item.id,
            type="calendar",
            title=f"Post on {item.platform}",
            subtitle=item.content[:50] + "..." if len(item.content) > 50 else item.content,
            url="/calendar"
        ))

    return SearchResponse(results=results[:10], total=len(results))
