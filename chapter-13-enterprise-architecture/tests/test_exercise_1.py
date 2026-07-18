"""tests/test_exercise_1.py
Tests for Exercise 13.1 — service scaffold.
Tests import `app` from exercises.service_scaffold.
Run: pytest tests/test_exercise_1.py -v
"""
import base64
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set a test public key so verify_jwt_rs256 (test-mode) works
os.environ.setdefault("JWT_PUBLIC_KEY", "test-key-placeholder")


def _make_test_jwt(sub="test-user", role="CREDIT_ANALYST",
                   exp_offset=3600):
    """Build a minimal JWT for testing (unsigned — test mode only)."""
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps({
            "sub": sub,
            "iat": int(time.time()),
            "exp": int(time.time()) + exp_offset,
            "custom:roles": [role],
        }).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesig"


@pytest.fixture
def client():
    """Create a TestClient for the exercise service."""
    from fastapi.testclient import TestClient
    try:
        from exercises.service_scaffold import app
    except (ImportError, AttributeError):
        pytest.skip("exercise_scaffold.app not implemented yet")
    return TestClient(app)


def test_health_check_with_valid_jwt(client):
    """Valid JWT must return 200 with correct schema."""
    token = _make_test_jwt()
    response = client.post(
        "/v1/health-check",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "service_name" in body
    assert "version" in body
    assert "timestamp" in body
    assert body["status"] == "healthy"


def test_health_check_no_token_returns_401(client):
    """Missing Authorization header must return 401."""
    response = client.post("/v1/health-check")
    assert response.status_code == 401


def test_health_check_expired_token_returns_401(client):
    """Expired JWT must return 401."""
    token = _make_test_jwt(exp_offset=-1)
    response = client.post(
        "/v1/health-check",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401
