"""Exercise 10.2 — Model Registry and Deployment Gate.

AWB Chapter 10 | Difficulty: ★★☆☆☆ | ~25 minutes

TASK
----
Complete the four steps below to simulate the AWB model
registration and deployment gate lifecycle:

  Step 1: Register a new model MR-2026-099 (MEDIUM risk).
  Step 2: Attempt deployment before validation — gate must
          REJECT with a clear error message.
  Step 3: Record a PASS validation result for MR-2026-099.
  Step 4: Attempt deployment again — gate must APPROVE.

All four steps are covered by pytest assertions below.

Run your solution:
  cd chapter_10
  pytest exercises/ex_10_2_registry.py -v

Solution:
  github.com/lorvenio/ai-banking-risk-platform
  /chapter_10/solutions/sol_10_2_registry.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


# ── Enumerations (do not modify) ──────────────────────────────

class RiskRating(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class ModelStatus(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    VALIDATION  = "VALIDATION"
    APPROVED    = "APPROVED"
    ACTIVE      = "ACTIVE"
    RETIRED     = "RETIRED"


# ── Data classes (do not modify) ──────────────────────────────

@dataclass
class ModelRecord:
    mr_reference: str
    model_name: str
    ss1_23_risk: RiskRating
    status: ModelStatus
    owner: str
    registered_at: datetime = field(
        default_factory=datetime.utcnow
    )
    validated_at: Optional[datetime] = None
    validation_outcome: Optional[str] = None  # PASS / FAIL
    next_revalidation: Optional[datetime] = None


@dataclass
class DeploymentDecision:
    approved: bool
    mr_reference: str
    reason: str


# ── YOUR IMPLEMENTATION ────────────────────────────────────────

class ModelRegistry:
    """Simplified AWB PRA SS1/23 model registry.

    Complete each method marked TODO.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelRecord] = {}

    # ── TODO 1 ────────────────────────────────────────────────
    def register(self, model: ModelRecord) -> ModelRecord:
        """Register a new model.

        Rules:
          - Raise ValueError if mr_reference already exists.
          - Set next_revalidation based on risk rating:
              LOW    -> 24 months from now
              MEDIUM -> 12 months from now
              HIGH   -> 6 months from now
          - Store and return the model record.
        """
        # TODO: implement registration
        raise NotImplementedError

    # ── TODO 2 ────────────────────────────────────────────────
    def record_validation(
        self,
        mr_reference: str,
        outcome: str,       # "PASS" or "FAIL"
        validator_id: str,
    ) -> ModelRecord:
        """Record a validation result.

        Rules:
          - Raise KeyError if mr_reference not found.
          - Set validated_at to datetime.utcnow().
          - Set validation_outcome to outcome.
          - If outcome == "PASS": set status to ACTIVE.
          - If outcome == "FAIL": set status to VALIDATION.
          - Return the updated record.
        """
        # TODO: implement validation recording
        raise NotImplementedError

    # ── TODO 3 ────────────────────────────────────────────────
    def deployment_gate(
        self,
        mr_reference: str,
    ) -> DeploymentDecision:
        """Check whether a model may be deployed.

        Approval requires ALL of:
          1. mr_reference exists in the registry.
          2. model.status == ACTIVE.
          3. model.validation_outcome == "PASS".

        Return DeploymentDecision(approved=True, ...) if all
        checks pass, otherwise approved=False with a descriptive
        reason string explaining which check failed.
        """
        # TODO: implement deployment gate
        raise NotImplementedError

    def get(self, mr_reference: str) -> ModelRecord:
        """Retrieve a model record (raises KeyError if absent)."""
        return self._models[mr_reference]


# ── Tests ─────────────────────────────────────────────────────
# Run:  pytest exercises/ex_10_2_registry.py -v

MR_REF = "MR-2026-099"


def _make_model() -> ModelRecord:
    return ModelRecord(
        mr_reference=MR_REF,
        model_name="AWB Test Model",
        ss1_23_risk=RiskRating.MEDIUM,
        status=ModelStatus.DEVELOPMENT,
        owner="credit.team@awb.co.uk",
    )


# Step 1 ─────────────────────────────────────────────────────

def test_step1_registration_succeeds() -> None:
    registry = ModelRegistry()
    model = registry.register(_make_model())
    assert model.mr_reference == MR_REF
    assert registry.get(MR_REF).mr_reference == MR_REF


def test_step1_duplicate_raises() -> None:
    registry = ModelRegistry()
    registry.register(_make_model())
    try:
        registry.register(_make_model())
        raise AssertionError(
            "Expected ValueError for duplicate MR reference"
        )
    except ValueError:
        pass


def test_step1_revalidation_set() -> None:
    registry = ModelRegistry()
    model = registry.register(_make_model())
    # MEDIUM risk -> ~12 months
    assert model.next_revalidation is not None
    months_ahead = (
        model.next_revalidation - datetime.utcnow()
    ).days / 30
    assert 11 <= months_ahead <= 13, (
        f"MEDIUM risk revalidation should be ~12 months, "
        f"got {months_ahead:.1f}"
    )


# Step 2 ─────────────────────────────────────────────────────

def test_step2_gate_rejects_before_validation() -> None:
    registry = ModelRegistry()
    registry.register(_make_model())
    decision = registry.deployment_gate(MR_REF)
    assert not decision.approved, (
        "Gate must reject an unvalidated model"
    )
    assert decision.reason, "Rejection must include a reason"


# Step 3 ─────────────────────────────────────────────────────

def test_step3_validation_pass_sets_active() -> None:
    registry = ModelRegistry()
    registry.register(_make_model())
    record = registry.record_validation(
        MR_REF, "PASS", "model.risk@awb.co.uk"
    )
    assert record.status == ModelStatus.ACTIVE
    assert record.validation_outcome == "PASS"
    assert record.validated_at is not None


def test_step3_validation_fail_stays_validation() -> None:
    registry = ModelRegistry()
    registry.register(_make_model())
    record = registry.record_validation(
        MR_REF, "FAIL", "model.risk@awb.co.uk"
    )
    assert record.status == ModelStatus.VALIDATION
    decision = registry.deployment_gate(MR_REF)
    assert not decision.approved


# Step 4 ─────────────────────────────────────────────────────

def test_step4_gate_approves_after_pass() -> None:
    registry = ModelRegistry()
    registry.register(_make_model())
    registry.record_validation(
        MR_REF, "PASS", "model.risk@awb.co.uk"
    )
    decision = registry.deployment_gate(MR_REF)
    assert decision.approved, (
        f"Gate must approve after PASS validation. "
        f"Reason: {decision.reason}"
    )
    assert decision.mr_reference == MR_REF


def test_unknown_mr_reference_rejected() -> None:
    registry = ModelRegistry()
    decision = registry.deployment_gate("MR-2026-UNKNOWN")
    assert not decision.approved
