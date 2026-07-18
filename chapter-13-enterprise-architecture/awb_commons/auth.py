"""awb_commons/auth.py
RS256 JWT verification for all AWB AI services.
Issued by AWS Cognito; validated against public key.
Roles: CREDIT_ANALYST | COMPLIANCE_OFFICER | MLRO | TREASURY
PRA SS1/23 | FCA PS22/9 | DORA Art.9
"""
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal RS256 JWT verifier — no third-party JWT lib dependency.
# Production uses python-jose; this stub supports testing without Cognito.
# ---------------------------------------------------------------------------

try:
    from jose import jwt as jose_jwt
    from jose.exceptions import JWTError, ExpiredSignatureError

    def verify_jwt_rs256(token: str) -> dict[str, Any]:
        """Verify RS256 JWT; return payload dict.

        Args:
            token: Raw JWT string (no 'Bearer ' prefix).

        Returns:
            Decoded payload with sub, roles, exp, iat.

        Raises:
            ValueError: On invalid signature, expiry, or missing claims.
        """
        public_key = os.environ.get("JWT_PUBLIC_KEY", "")
        if not public_key:
            raise ValueError("JWT_PUBLIC_KEY env var not set")
        try:
            payload = jose_jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                options={"verify_aud": False},
            )
        except ExpiredSignatureError as exc:
            logger.warning("jwt_expired", extra={"error": str(exc)})
            raise ValueError("JWT expired") from exc
        except JWTError as exc:
            logger.warning(
                "jwt_invalid", extra={"error": str(exc)}
            )
            raise ValueError(f"JWT invalid: {exc}") from exc

        _validate_claims(payload)
        logger.info(
            "jwt_verified",
            extra={
                "sub": payload.get("sub"),
                "roles": payload.get("custom:roles", []),
            },
        )
        return payload

except ImportError:
    # Test-mode fallback — python-jose not installed
    import base64
    import json

    def verify_jwt_rs256(token: str) -> dict[str, Any]:  # type: ignore[misc]
        """Test-mode JWT decode (no signature check).

        Only used when python-jose is unavailable.
        Real production always uses the jose implementation above.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                raise ValueError("Malformed JWT")
            padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(
                base64.urlsafe_b64decode(padded).decode()
            )
        except Exception as exc:
            raise ValueError(f"JWT decode failed: {exc}") from exc

        _validate_claims(payload)
        return payload


def _validate_claims(payload: dict[str, Any]) -> None:
    """Assert mandatory AWB JWT claims are present and valid."""
    required = ("sub", "iat", "exp")
    for claim in required:
        if claim not in payload:
            raise ValueError(f"Missing required JWT claim: {claim}")

    if payload["exp"] < time.time():
        raise ValueError("JWT expired")

    roles = payload.get("custom:roles", [])
    valid_roles = {
        "CREDIT_ANALYST",
        "COMPLIANCE_OFFICER",
        "MLRO",
        "TREASURY",
        "ADMIN",
    }
    if roles and not set(roles).intersection(valid_roles):
        raise ValueError(
            f"JWT contains no recognised AWB role: {roles}"
        )


def require_role(payload: dict[str, Any], role: str) -> None:
    """Assert JWT payload contains required role.

    Args:
        payload: Decoded JWT payload from verify_jwt_rs256().
        role:    Required role string.

    Raises:
        PermissionError: If role not present in JWT.
    """
    roles = payload.get("custom:roles", [])
    if role not in roles:
        logger.warning(
            "role_check_failed",
            extra={
                "required": role,
                "present": roles,
                "sub": payload.get("sub"),
            },
        )
        raise PermissionError(
            f"Role '{role}' required; JWT has {roles}"
        )
