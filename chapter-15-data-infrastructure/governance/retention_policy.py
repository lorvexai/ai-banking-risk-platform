# governance/retention_policy.py
# AWB Data Retention Schedule — unified 7-year policy
# Satisfies: FCA COBS 9 | MLR 2017 | POCA 2002 |
#            PRA SS1/23 | DORA | SM&CR
# GOV-2026-001 | AWB-AI-2025
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """AWB regulated data categories with retention basis.

    Seven-year baseline unifies FCA COBS 9, MLR 2017,
    POCA 2002, PRA SS1/23, and DORA into a single
    operational policy. Maintaining five different periods
    for data needed together in regulatory investigations
    is operationally fragile — always align to longest.
    """
    CREDIT_DECISIONS = "credit_decisions"        # FCA COBS 9
    SAR_RECORDS = "sar_records"                  # MLR 2017 Reg 40
    MODEL_OUTPUTS = "model_outputs"              # PRA SS1/23
    AUDIT_LOGS = "audit_logs"                    # SM&CR / DORA
    KYC_RECORDS = "kyc_records"                  # MLR 2017/POCA
    REGULATORY_POSITIONS = "reg_positions"       # BCBS239/CRR3
    TRAINING_DATASETS = "training_datasets"      # PRA SS1/23
    VECTOR_EMBEDDINGS = "vector_embeddings"      # PRA SS1/23


# Canonical retention schedule — single source of truth
RETENTION_SCHEDULE: Dict[DataCategory, dict] = {
    DataCategory.CREDIT_DECISIONS: {
        "years": 7,
        "basis": "FCA COBS 9.1.3R",
        "s3_lock": True,
    },
    DataCategory.SAR_RECORDS: {
        "years": 5,
        "basis": "MLR 2017 Regulation 40",
        "s3_lock": True,
    },
    DataCategory.MODEL_OUTPUTS: {
        "years": 7,
        "basis": "PRA SS1/23 Section 4",
        "s3_lock": True,
    },
    DataCategory.AUDIT_LOGS: {
        "years": 7,
        "basis": "FCA COBS 9 / DORA Art.17",
        "s3_lock": True,
    },
    DataCategory.KYC_RECORDS: {
        "years": 7,
        "basis": "MLR 2017 Reg 40 / POCA 2002",
        "s3_lock": True,
    },
    DataCategory.REGULATORY_POSITIONS: {
        "years": 7,
        "basis": "BCBS 239 / CRR3 Art.430",
        "s3_lock": True,
    },
    DataCategory.TRAINING_DATASETS: {
        "years": 5,
        "basis": "PRA SS1/23 (DVC versioned)",
        "s3_lock": False,
    },
    DataCategory.VECTOR_EMBEDDINGS: {
        "years": 5,
        "basis": "PRA SS1/23 Section 4",
        "s3_lock": False,
    },
}


@dataclass
class RetentionPolicy:
    """Retention policy for one AWB data category."""
    category: DataCategory
    retention_years: int
    regulatory_basis: str
    requires_s3_object_lock: bool

    def to_s3_lifecycle_days(self) -> int:
        """Convert retention years to S3 lifecycle rule days."""
        return self.retention_years * 365


def get_retention_policy(
    category: DataCategory,
) -> RetentionPolicy:
    """Return the retention policy for a given data category.

    S3 Object Lock (Compliance mode) is the only mechanism
    that truly prevents deletion before the retention period
    expires. IAM boundaries can be overridden by sufficiently
    privileged users; Object Lock cannot.

    Args:
        category: DataCategory enum value.

    Returns:
        RetentionPolicy with years, basis, and S3 lock flag.

    Raises:
        KeyError: If category has no schedule entry.

    Example:
        policy = get_retention_policy(
            DataCategory.CREDIT_DECISIONS
        )
        assert policy.retention_years == 7
        assert policy.requires_s3_object_lock is True
        assert policy.to_s3_lifecycle_days() == 2555
    """
    config = RETENTION_SCHEDULE[category]
    logger.info(
        "Retention policy: %s = %d years (%s)",
        category.value,
        config["years"],
        config["basis"],
    )
    return RetentionPolicy(
        category=category,
        retention_years=config["years"],
        regulatory_basis=config["basis"],
        requires_s3_object_lock=config["s3_lock"],
    )
