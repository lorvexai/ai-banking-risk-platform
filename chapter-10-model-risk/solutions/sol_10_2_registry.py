"""Solution — Exercise 10.2: Model Registry and Deployment Gate.

AWB Chapter 10 | Reference solution (do not peek until done!)
"""
from __future__ import annotations
from datetime import datetime, timedelta

from exercises.ex_10_2_registry import (
    ModelRecord,
    ModelRegistry,
    ModelStatus,
    RiskRating,
    DeploymentDecision,
)

REVALIDATION_MONTHS: dict[RiskRating, int] = {
    RiskRating.LOW:    24,
    RiskRating.MEDIUM: 12,
    RiskRating.HIGH:    6,
}


class SolModelRegistry(ModelRegistry):
    """Reference implementation of all three methods."""

    def register(self, model: ModelRecord) -> ModelRecord:
        if model.mr_reference in self._models:
            raise ValueError(
                f"{model.mr_reference} already registered"
            )
        months = REVALIDATION_MONTHS[model.ss1_23_risk]
        model.next_revalidation = (
            datetime.utcnow()
            + timedelta(days=months * 30)
        )
        self._models[model.mr_reference] = model
        return model

    def record_validation(
        self,
        mr_reference: str,
        outcome: str,
        validator_id: str,
    ) -> ModelRecord:
        record = self._models[mr_reference]  # KeyError if absent
        record.validated_at = datetime.utcnow()
        record.validation_outcome = outcome
        if outcome == "PASS":
            record.status = ModelStatus.ACTIVE
        else:
            record.status = ModelStatus.VALIDATION
        return record

    def deployment_gate(
        self,
        mr_reference: str,
    ) -> DeploymentDecision:
        if mr_reference not in self._models:
            return DeploymentDecision(
                approved=False,
                mr_reference=mr_reference,
                reason=(
                    f"{mr_reference} not found in registry. "
                    "Register before deploying."
                ),
            )
        record = self._models[mr_reference]
        if record.status != ModelStatus.ACTIVE:
            return DeploymentDecision(
                approved=False,
                mr_reference=mr_reference,
                reason=(
                    f"Model status is {record.status.value}. "
                    "Must be ACTIVE (requires PASS validation)."
                ),
            )
        if record.validation_outcome != "PASS":
            return DeploymentDecision(
                approved=False,
                mr_reference=mr_reference,
                reason=(
                    "No PASS validation recorded. "
                    "Complete independent validation first."
                ),
            )
        return DeploymentDecision(
            approved=True,
            mr_reference=mr_reference,
            reason="All SS1/23 deployment gate checks passed.",
        )
