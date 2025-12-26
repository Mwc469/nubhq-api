from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .routes import approvals_router, dashboard_router, fan_mail_router, settings_router, calendar_router, ai_trainer_router

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
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
