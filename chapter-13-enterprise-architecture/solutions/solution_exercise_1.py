"""solutions/solution_exercise_1.py
Exercise 13.1 — Reference solution.
Do not read before attempting the exercise!

Build a FastAPI service using awb_commons create_app().
"""
import os
import time

from fastapi import Request, HTTPException, status

from awb_commons.app_factory import HealthResponse, create_app
from awb_commons.auth import verify_jwt_rs256
from awb_commons.logging_client import get_structured_logger

logger = get_structured_logger(
    __name__, service_name="exercise-13-1"
)

app = create_app(
    service_name="exercise-13-1",
    version="1.0.0",
)


@app.post("/v1/health-check", response_model=HealthResponse)
async def health_check_authed(request: Request) -> HealthResponse:
    """JWT-authenticated health check endpoint.

    Returns HTTP 401 if Bearer token is missing or invalid.
    Returns HTTP 200 with HealthResponse on success.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = verify_jwt_rs256(auth[7:])
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    logger.info(
        "health_check_called",
        extra={
            "sub": payload.get("sub"),
            "service": "exercise-13-1",
            "version": "1.0.0",
        },
    )
    return HealthResponse(
        service_name="exercise-13-1",
        version="1.0.0",
        status="healthy",
        timestamp=time.time(),
    )
