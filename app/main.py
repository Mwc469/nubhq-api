"""
NubHQ API - FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .config import get_settings
from .database import engine, Base
from .middleware import SecurityHeadersMiddleware, RequestLoggingMiddleware
from .limiter import limiter
from .routes import (
    approvals_router,
    dashboard_router,
    fan_mail_router,
    settings_router,
    calendar_router,
    ai_trainer_router,
    auth_router,
    search_router,
    posts_router,
    templates_router,
    analytics_router,
    activity_router,
    email_campaigns_router,
    media_router,
    video_pipeline_router,
    webhooks_router,
)

settings = get_settings()

# Create tables (in production, use Alembic migrations instead)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="NubHQ API",
    description="Backend API for NubHQ dashboard",
    version="1.0.0",
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Request logging middleware (only in debug mode)
if settings.debug:
    app.add_middleware(RequestLoggingMiddleware)

# CORS - Properly configured with specific methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Routes
app.include_router(approvals_router)
app.include_router(dashboard_router)
app.include_router(fan_mail_router)
app.include_router(settings_router)
app.include_router(calendar_router)
app.include_router(ai_trainer_router)
app.include_router(auth_router)
app.include_router(search_router)
app.include_router(posts_router)
app.include_router(templates_router)
app.include_router(analytics_router)
app.include_router(activity_router)
app.include_router(email_campaigns_router)
app.include_router(media_router)
app.include_router(video_pipeline_router)
app.include_router(webhooks_router)


@app.get("/api/health")
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "1.0.0",
    }


@app.get("/")
def root():
    """Root endpoint redirects to API docs."""
    return {
        "message": "NubHQ API",
        "docs": "/api/docs" if settings.debug else "Disabled in production",
    }
