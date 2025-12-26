"""
Authentication routes for login, register, and token management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..database import get_db
from ..models.user import User
from ..schemas.auth import UserCreate, UserLogin, UserResponse, Token
from ..auth import (
    verify_password,
    get_password_hash,
    create_tokens,
    get_required_user,
    refresh_access_token,
)
from ..config import get_settings

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class TokenResponse(BaseModel):
    """Response with both access and refresh tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    """Request to refresh tokens."""
    refresh_token: str


@router.post("/register", response_model=UserResponse)
@limiter.limit("3/minute")
def register(request: Request, user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account."""
    # Check if user exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    user = User(
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        display_name=user_data.display_name or user_data.email.split("@")[0],
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login with OAuth2 form (username/password)."""
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token, refresh_token = create_tokens(user.id)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/login/json", response_model=TokenResponse)
@limiter.limit("5/minute")
def login_json(request: Request, credentials: UserLogin, db: Session = Depends(get_db)):
    """Login with JSON body (email/password)."""
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token, refresh_token = create_tokens(user.id)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("10/minute")
def refresh_tokens(request: Request, refresh_request: RefreshRequest, db: Session = Depends(get_db)):
    """Get new access and refresh tokens using a valid refresh token."""
    tokens = refresh_access_token(refresh_request.refresh_token, db)
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    access_token, refresh_token = tokens
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_required_user)):
    """Get current authenticated user."""
    return current_user


@router.post("/logout")
def logout(current_user: User = Depends(get_required_user)):
    """
    Logout the current user.

    Note: Since JWTs are stateless, this is a client-side operation.
    The client should delete the tokens. For additional security,
    implement a token blacklist in Redis.
    """
    return {"message": "Successfully logged out"}
