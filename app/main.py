from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
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
)

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="NubHQ API",
    description="Backend API for NubHQ dashboard",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "https://web-pi-livid.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
