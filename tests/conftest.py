"""
Pytest configuration and fixtures for NubHQ API tests.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.limiter import limiter
from app.main import app
from app.models.user import User
from app.auth import get_password_hash, create_access_token

# Disable rate limiting for tests
limiter.enabled = False

# Use in-memory SQLite for tests with shared connection
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Global session for sharing across requests
_test_session = None


def get_test_db():
    """Get the shared test database session."""
    global _test_session
    try:
        yield _test_session
    finally:
        pass


@pytest.fixture(scope="function")
def db():
    """Create a fresh database for each test."""
    global _test_session

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create a session
    _test_session = TestingSessionLocal()

    # Override the get_db dependency
    app.dependency_overrides[get_db] = get_test_db

    yield _test_session

    # Cleanup
    app.dependency_overrides.clear()
    _test_session.close()
    _test_session = None

    # Drop all tables
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db):
    """Create a test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="function")
def test_user(db):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        display_name="Test User",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="function")
def auth_token(test_user):
    """Get an auth token for the test user."""
    return create_access_token({"sub": test_user.id})


@pytest.fixture(scope="function")
def auth_headers(auth_token):
    """Get auth headers for the test user."""
    return {"Authorization": f"Bearer {auth_token}"}
