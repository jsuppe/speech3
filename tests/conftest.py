"""Pytest configuration and fixtures for API tests."""
import os
import pytest
import httpx

# Test against local or production API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "sk-5b7fd66025ead6b8731ef73b2c970f26")

# Performance thresholds (in seconds)
# These are for warm requests after initial cold start
THRESHOLDS = {
    "auth_check": 0.5,        # /v1/auth/me
    "speeches_list": 0.5,      # /v1/speeches (paginated) - relaxed for 61K rows
    "speech_detail": 0.3,      # /v1/speeches/{id}
    "speech_score": 0.5,       # /v1/speeches/{id}/score
    "speakers_list": 0.3,      # /v1/speakers
    "projects_list": 0.3,      # /v1/projects
    "health_check": 0.3,       # / - first request can be slower
}


@pytest.fixture(scope="session")
def api_client():
    """HTTP client for API requests with warmup."""
    client = httpx.Client(
        base_url=API_BASE_URL,
        timeout=30.0,
        params={"api_key": API_KEY}  # API uses query param, not header
    )
    # Warmup - make a few requests to ensure server is ready
    for _ in range(3):
        try:
            client.get("/")
        except Exception:
            pass
    return client


@pytest.fixture(scope="session")
def api_key():
    return API_KEY


@pytest.fixture(scope="session")
def thresholds():
    return THRESHOLDS
