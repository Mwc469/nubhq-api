# NubHQ API

FastAPI backend for NubHQ - a gamified content approval dashboard.

## Features

- JWT authentication with refresh tokens
- Content approval workflow
- Fan mail management
- Calendar/scheduling
- Video processing pipeline
- AI-powered engagement scoring
- Rate limiting and security headers

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg (for video processing)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Run development server
uvicorn app.main:app --reload --port 8000
```

### Environment Variables

See `.env.example` for all available configuration options.

Required:
- `SECRET_KEY` - JWT signing key (generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
- `DATABASE_URL` - Database connection string

## API Documentation

When running in development mode, API docs are available at:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Project Structure

```
app/
├── main.py              # FastAPI application entry
├── config.py            # Configuration settings
├── database.py          # Database connection
├── auth.py              # JWT authentication
├── models/              # SQLAlchemy models
├── schemas/             # Pydantic schemas
├── routes/              # API endpoints
└── worker/              # Background workers
    ├── intelligent_processor.py  # Video processing
    ├── engagement_scorer.py      # AI scoring
    └── content_combiner.py       # Video combining
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## Docker

```bash
# Build image
docker build -t nubhq-api .

# Run container
docker run -p 8000:8000 --env-file .env nubhq-api
```

## Health Checks

- `GET /api/health` - Basic health check
- `GET /api/health/live` - Kubernetes liveness probe
- `GET /api/health/ready` - Readiness probe (checks DB)
- `GET /api/health/full` - Full diagnostics
