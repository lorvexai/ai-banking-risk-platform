"""awb_commons — AWB shared library v2.3.1
Mandatory foundation for all AWB-AI-2025 programme services.
War story: £2.3M rework from missing shared standards (Ch 13).

Components:
    create_app()        FastAPI factory (JWT, CORS, rate limiting)
    AWBDatabaseClient   Aurora Serverless v2 connection pool
    AWBLLMFactory       DORA Art.28 multi-provider LLM client
    CircuitBreaker      DORA Art.17 resilience pattern
    T24WriteClient      Idempotent T24 write with compensation
    get_structured_logger  CloudWatch-indexed JSON logger
    verify_jwt_rs256    RS256 JWT verification
"""
from .app_factory import create_app
from .auth import require_role, verify_jwt_rs256
from .circuit_breaker import CircuitBreaker, CircuitOpenError
from .db_client import AWBDatabaseClient
from .llm_factory import AWBLLMFactory, LLMResponse
from .logging_client import get_structured_logger
from .t24_client import T24WriteClient, T24WriteResult

__version__ = "2.3.1"
__all__ = [
    "create_app",
    "require_role",
    "verify_jwt_rs256",
    "CircuitBreaker",
    "CircuitOpenError",
    "AWBDatabaseClient",
    "AWBLLMFactory",
    "LLMResponse",
    "get_structured_logger",
    "T24WriteClient",
    "T24WriteResult",
]
