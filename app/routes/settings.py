from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.settings import UserSettings
from ..schemas.settings import SettingsUpdate, SettingsResponse

router = APIRouter(prefix="/api/settings", tags=["settings"])

DEFAULT_USER_ID = "default"


@router.get("", response_model=SettingsResponse)
def get_settings(db: Session = Depends(get_db)):
    settings = db.query(UserSettings).filter(UserSettings.user_id == DEFAULT_USER_ID).first()
    if not settings:
        settings = UserSettings(
            user_id=DEFAULT_USER_ID,
            display_name="Creator",
            push_notifications=True,
            email_notifications=True
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


@router.patch("", response_model=SettingsResponse)
def update_settings(update: SettingsUpdate, db: Session = Depends(get_db)):
    settings = db.query(UserSettings).filter(UserSettings.user_id == DEFAULT_USER_ID).first()
    if not settings:
        settings = UserSettings(user_id=DEFAULT_USER_ID)
        db.add(settings)

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(settings, field, value)

    db.commit()
    db.refresh(settings)
    return settings
