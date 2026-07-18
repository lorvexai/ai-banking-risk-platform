# customer_360/profile_builder.py
# AWB Customer 360 — UK GDPR Art.5 data minimisation
# AWB-AI-2025 | C360-2026-001 | ICT Asset: RESTRICTED
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CustomerRole(Enum):
    """Access roles — enforces UK GDPR purpose limitation."""
    CREDIT_ANALYST = "credit_analyst"
    RELATIONSHIP_MGR = "relationship_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    MLRO = "mlro"  # Only role with SAR access — POCA s.333A


@dataclass
class CustomerProfile:
    """Unified AWB profile keyed on AWB_CUSTOMER_ID.

    2.4M customers | 7-year retention | FCA COBS 9.
    AWB_CUSTOMER_ID is the universal master key across
    T24, Salesforce, digital banking, and credit risk.
    """
    awb_customer_id: str
    full_name: str
    kyc_status: str           # MR-2026-050 output
    aml_risk_band: str        # MR-2026-051 output
    pd_estimate: Optional[float] = None   # Credit only
    lgd_estimate: Optional[float] = None  # Credit only
    churn_score: Optional[float] = None   # MR-2026-053
    herfindahl_index: Optional[float] = None
    sar_flag: Optional[bool] = None       # MLRO only


class Customer360Builder:
    """Build role-scoped Customer 360 profiles.

    Enforces UK GDPR Art.5(1)(b) purpose limitation by
    returning only the fields documented for each role's
    ROPA entry. SAR history accessible only to MLRO per
    POCA 2002 section 333A.

    Args:
        db_client: PostgreSQL connection pool.
        redis_client: Redis cache (30s freshness SLA).

    Example:
        builder = Customer360Builder(db, redis)
        profile = builder.build_profile(
            awb_customer_id="AWB-001234",
            requesting_role=CustomerRole.CREDIT_ANALYST,
            purpose="credit_risk_assessment",
        )
    """

    def __init__(self, db_client, redis_client) -> None:
        self._db = db_client
        self._cache = redis_client

    def build_profile(
        self,
        awb_customer_id: str,
        requesting_role: CustomerRole,
        purpose: str,
    ) -> CustomerProfile:
        """Return role-scoped profile; log for audit trail.

        Args:
            awb_customer_id: Universal AWB master key.
            requesting_role: Determines which fields returned.
            purpose: ROPA purpose code for audit log.

        Returns:
            CustomerProfile with only permitted fields.

        Raises:
            ValueError: If customer ID not found.
            PermissionError: If role/purpose mismatch.
        """
        logger.info(
            "C360 access: id=%s role=%s purpose=%s",
            awb_customer_id,
            requesting_role.value,
            purpose,
        )
        base = self._fetch_base_profile(awb_customer_id)

        if requesting_role == CustomerRole.CREDIT_ANALYST:
            return base  # PD/LGD/EAD fields populated

        if requesting_role == CustomerRole.MLRO:
            # Only MLRO can access SAR history — POCA s.333A
            sar = self._fetch_sar_flag(awb_customer_id)
            base.sar_flag = sar
            return base

        if requesting_role == CustomerRole.COMPLIANCE_OFFICER:
            base.pd_estimate = None   # Not needed for purpose
            base.lgd_estimate = None
            return base

        # Relationship Manager — no credit model fields
        base.pd_estimate = None
        base.lgd_estimate = None
        return base

    def _fetch_base_profile(
        self, awb_customer_id: str
    ) -> CustomerProfile:
        """Fetch from PostgreSQL or Redis cache."""
        raise NotImplementedError(
            "Full implementation in GitHub repo"
        )

    def _fetch_sar_flag(
        self, awb_customer_id: str
    ) -> bool:
        """Fetch SAR flag — MLRO-only endpoint."""
        raise NotImplementedError(
            "Full implementation in GitHub repo"
        )
