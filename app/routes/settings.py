"""
Settings routes for user preferences.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.settings import UserSettings
from ..models.user import User
from ..auth import get_required_user
from ..schemas.settings import SettingsUpdate, SettingsResponse

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
def get_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Get settings for the current user."""
    settings = db.query(UserSettings).filter(
        UserSettings.user_id == current_user.id
    ).first()

    if not settings:
        settings = UserSettings(
            user_id=current_user.id,
            push_notifications=True,
            email_notifications=True,
            dark_mode=True,
            dry_run_mode=False,
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)

    return settings


@router.patch("", response_model=SettingsResponse)
def update_settings(
    update: SettingsUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user),
):
    """Update settings for the current user."""
    settings = db.query(UserSettings).filter(
        UserSettings.user_id == current_user.id
    ).first()

    if not settings:
        settings = UserSettings(user_id=current_user.id)
        db.add(settings)

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(settings, field, value)

    db.commit()
    db.refresh(settings)
    return settings
