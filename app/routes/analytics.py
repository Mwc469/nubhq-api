from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime, timedelta

from ..database import get_db

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview")
def get_analytics_overview(
    period: str = "30d",
    db: Session = Depends(get_db)
):
    """Get analytics overview with key metrics"""
    return {
        "total_followers": 24500,
        "followers_change": 12.3,
        "total_revenue": 8420,
        "revenue_change": 8.7,
        "post_impressions": 156000,
        "impressions_change": 24.1,
        "engagement_rate": 8.4,
        "engagement_change": -2.1,
        "period": period,
    }


@router.get("/engagement")
def get_engagement_data(
    period: str = "7d",
    db: Session = Depends(get_db)
):
    """Get engagement data over time"""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return [
        {"name": day, "likes": 2400 + i * 500, "comments": 400 + i * 50, "shares": 240 + i * 30}
        for i, day in enumerate(days)
    ]


@router.get("/revenue")
def get_revenue_data(
    period: str = "6m",
    db: Session = Depends(get_db)
):
    """Get revenue data over time"""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    values = [4000, 3000, 5000, 4500, 6000, 5500, 7000]
    return [
        {"name": month, "value": value}
        for month, value in zip(months, values)
    ]


@router.get("/platforms")
def get_platform_distribution(db: Session = Depends(get_db)):
    """Get follower distribution by platform"""
    return [
        {"name": "Instagram", "value": 45, "followers": 11025},
        {"name": "Twitter", "value": 25, "followers": 6125},
        {"name": "YouTube", "value": 20, "followers": 4900},
        {"name": "TikTok", "value": 10, "followers": 2450},
    ]


@router.get("/top-posts")
def get_top_posts(
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """Get top performing posts"""
    return [
        {"id": 1, "title": "Behind the scenes 📸", "engagement": 12400, "platform": "Instagram"},
        {"id": 2, "title": "Q&A Session highlights", "engagement": 8900, "platform": "Twitter"},
        {"id": 3, "title": "New collection reveal!", "engagement": 7200, "platform": "TikTok"},
        {"id": 4, "title": "Day in my life vlog", "engagement": 6800, "platform": "YouTube"},
        {"id": 5, "title": "Giveaway announcement", "engagement": 5500, "platform": "Instagram"},
    ][:limit]


@router.get("/demographics")
def get_demographics(db: Session = Depends(get_db)):
    """Get audience demographics"""
    return {
        "age_groups": [
            {"range": "18-24", "percentage": 35},
            {"range": "25-34", "percentage": 40},
            {"range": "35-44", "percentage": 15},
            {"range": "45+", "percentage": 10},
        ],
        "gender": [
            {"type": "Female", "percentage": 65},
            {"type": "Male", "percentage": 32},
            {"type": "Other", "percentage": 3},
        ],
        "top_locations": [
            {"country": "United States", "percentage": 45},
            {"country": "United Kingdom", "percentage": 15},
            {"country": "Canada", "percentage": 10},
            {"country": "Australia", "percentage": 8},
            {"country": "Germany", "percentage": 5},
        ],
    }
