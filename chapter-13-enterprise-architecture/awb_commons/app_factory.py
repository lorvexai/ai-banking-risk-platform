"""awb_commons/app_factory.py
AWB shared FastAPI application factory v2.3.1.
All AI services call create_app() as their entry point.
PRA SS1/23 | DORA Art.17 | FCA PS22/9
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .auth import verify_jwt_rs256
from .logging_client import get_structured_logger

logger = get_structured_logger(__name__)


class HealthResponse(BaseModel):
    """Standard AWB health-check response."""
    service_name: str
    version: str
    status: str = "healthy"
    timestamp: float


def create_app(
    service_name: str,
    version: str,
    routers: list | None = None,
    lifespan: Callable | None = None,
) -> FastAPI:
    """Create a production-ready AWB FastAPI service.

    Args:
        service_name: DORA ICT asset name (e.g. "credit-doc-analyser").
        version:      Semantic version string (e.g. "1.4.2").
        routers:      Optional list of APIRouter to include.
        lifespan:     Optional async context manager for startup/shutdown.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=service_name,
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    # ── CORS — internal AWB origins only ─────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://*.awb.internal",
            "https://portal.awb.co.uk",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # ── Rate limiting middleware (100 req/min per JWT sub) ────────────
    _rate_counters: dict[str, list[float]] = {}

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        token = request.headers.get("Authorization", "")
        if token.startswith("Bearer "):
            try:
                payload = verify_jwt_rs256(token[7:])
                sub = payload.get("sub", "anonymous")
            except Exception:
                sub = "anonymous"
        else:
            sub = "anonymous"

        now = time.time()
        hits = [t for t in _rate_counters.get(sub, [])
                if now - t < 60]
        if len(hits) >= 100:
            logger.warning(
                "rate_limit_exceeded",
                extra={"sub": sub, "service": service_name}
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "rate limit exceeded"},
            )
        hits.append(now)
        _rate_counters[sub] = hits
        return await call_next(request)

    # ── Health check — no auth required ──────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(
            service_name=service_name,
            version=version,
            timestamp=time.time(),
        )

    # ── Auth-protected health check (Exercise 13.1 target) ───────────
    @app.post("/v1/health-check",
              response_model=HealthResponse, tags=["ops"])
    async def health_check_authed(request: Request) -> HealthResponse:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token required",
            )
        verify_jwt_rs256(auth[7:])  # raises on invalid
        logger.info(
            "health_check_called",
            extra={"service": service_name, "version": version},
        )
        return HealthResponse(
            service_name=service_name,
            version=version,
            timestamp=time.time(),
        )

    # ── Register routers ──────────────────────────────────────────────
    for router in (routers or []):
        app.include_router(router)

    logger.info(
        "service_started",
        extra={"service": service_name, "version": version},
    )
    return app
