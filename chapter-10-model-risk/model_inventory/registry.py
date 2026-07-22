"""
model_inventory/registry.py — AWB Enterprise Model Registry.
PRA SS1/23 compliant model inventory for AWB-AI-2025.
Avon & Wessex Bank plc (AWB), Bristol, UK.

All AI/ML models used in regulated functions must be registered
before deployment. This module manages the lifecycle of model
records from development through retirement.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import Optional
from awb_commons.models import (
    ModelRecord, ValidationResult,
    RiskRating, ModelStatus,
)

logger = logging.getLogger(__name__)

# PRA SS1/23: revalidation intervals by risk rating
REVALIDATION_MONTHS = {
    RiskRating.HIGH:   12,
    RiskRating.MEDIUM: 18,
    RiskRating.LOW:    24,
}


class ModelRegistry:
    """
    AWB Enterprise Model Registry (PRA SS1/23 compliant).

    Maintains the authoritative record of all AI/ML models used
    in AWB's regulated functions. The registry is the primary
    audit evidence for PRA supervisory reviews.

    SS1/23 requirements met:
    - Complete model inventory with risk ratings
    - Validation history and schedules
    - Status lifecycle management
    - Automatic revalidation trigger identification
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelRecord] = {}
        self._validations: dict[str, list[ValidationResult]] = {}
        self._preload_awb_models()
        logger.info(
            "ModelRegistry initialised: %d models loaded",
            len(self._models),
        )

    def register(self, model: ModelRecord) -> ModelRecord:
        """
        Register a new model in the inventory.

        Args:
            model: ModelRecord to register.

        Returns:
            The registered ModelRecord.

        Raises:
            ValueError: If MR reference already exists.
        """
        if model.mr_reference in self._models:
            raise ValueError(
                f"Model {model.mr_reference} already registered"
            )
        if model.next_revalidation is None:
            months = REVALIDATION_MONTHS[model.ss1_23_risk]
            model.next_revalidation = (
                datetime.utcnow()
                + timedelta(days=months * 30)
            )
        self._models[model.mr_reference] = model
        self._validations[model.mr_reference] = []
        logger.info(
            "Registered model: %s risk=%s next_val=%s",
            model.mr_reference,
            model.ss1_23_risk,
            model.next_revalidation.date()
            if model.next_revalidation else "None",
        )
        return model

    def get(self, mr_reference: str) -> Optional[ModelRecord]:
        """Retrieve a model record by MR reference."""
        return self._models.get(mr_reference)

    def update_status(
        self,
        mr_reference: str,
        new_status: ModelStatus,
    ) -> ModelRecord:
        """
        Transition model to a new lifecycle status.

        Args:
            mr_reference: Model identifier.
            new_status: Target status.

        Returns:
            Updated ModelRecord.

        Raises:
            KeyError: If model not found in registry.
            ValueError: If status transition is invalid.
        """
        model = self._require_model(mr_reference)
        self._validate_transition(model.status, new_status)
        model.status = new_status
        logger.info(
            "Status updated: %s → %s",
            mr_reference,
            new_status,
        )
        return model

    def record_validation(
        self, result: ValidationResult
    ) -> None:
        """Record a completed validation exercise."""
        mr = result.mr_reference
        model = self._require_model(mr)
        self._validations[mr].append(result)
        if result.outcome in ("PASS", "CONDITIONAL_PASS"):
            model.validated_at = result.validated_at
            months = REVALIDATION_MONTHS[model.ss1_23_risk]
            model.next_revalidation = (
                result.validated_at
                + timedelta(days=months * 30)
            )
        logger.info(
            "Validation recorded: %s outcome=%s",
            mr, result.outcome,
        )

    def due_for_revalidation(
        self, as_of: datetime | None = None
    ) -> list[ModelRecord]:
        """
        Return models whose revalidation is overdue or due
        within the next 30 days.
        """
        as_of = as_of or datetime.utcnow()
        window = as_of + timedelta(days=30)
        return [
            m for m in self._models.values()
            if m.next_revalidation
            and m.next_revalidation <= window
            and m.status == ModelStatus.ACTIVE
        ]

    def all_models(self) -> list[ModelRecord]:
        return list(self._models.values())

    def validation_history(
        self, mr_reference: str
    ) -> list[ValidationResult]:
        return self._validations.get(mr_reference, [])

    # ── Private helpers ───────────────────────────────────────────

    def _require_model(self, mr: str) -> ModelRecord:
        model = self._models.get(mr)
        if not model:
            raise KeyError(
                f"Model {mr} not found in registry"
            )
        return model

    def _validate_transition(
        self,
        current: ModelStatus,
        new: ModelStatus,
    ) -> None:
        allowed: dict[ModelStatus, list[ModelStatus]] = {
            ModelStatus.DEVELOPMENT: [
                ModelStatus.VALIDATION,
                ModelStatus.RETIRED,
            ],
            ModelStatus.VALIDATION: [
                ModelStatus.APPROVED,
                ModelStatus.DEVELOPMENT,
            ],
            ModelStatus.APPROVED: [
                ModelStatus.ACTIVE,
                ModelStatus.DEVELOPMENT,
            ],
            ModelStatus.ACTIVE: [
                ModelStatus.UNDER_REVIEW,
                ModelStatus.RETIRED,
            ],
            ModelStatus.UNDER_REVIEW: [
                ModelStatus.ACTIVE,
                ModelStatus.RETIRED,
            ],
            ModelStatus.RETIRED: [],
        }
        if new not in allowed.get(current, []):
            raise ValueError(
                f"Invalid transition: {current} → {new}"
            )

    def _preload_awb_models(self) -> None:
        """Pre-load the AWB-AI-2025 model registry."""
        from awb_commons.models import EUAIActClass
        models = [
            ModelRecord(
                mr_reference="MR-2026-035",
                model_name="AWB Credit Document Analyser",
                chapter=2,
                ss1_23_risk=RiskRating.MEDIUM,
                eu_ai_act=EUAIActClass.HIGH_RISK,
                status=ModelStatus.ACTIVE,
                owner="Credit Risk",
                validator="Model Risk Team",
                validated_at=datetime(2025, 2, 15),
            ),
            ModelRecord(
                mr_reference="MR-2026-037",
                model_name="AWB Credit Decision Agent",
                chapter=3,
                ss1_23_risk=RiskRating.HIGH,
                eu_ai_act=EUAIActClass.HIGH_RISK,
                status=ModelStatus.ACTIVE,
                owner="Credit Risk",
                validator="Model Risk Team",
                validated_at=datetime(2025, 3, 1),
            ),
            ModelRecord(
                mr_reference="MR-2026-038",
                model_name="AWB Regulatory Knowledge Assistant",
                chapter=4,
                ss1_23_risk=RiskRating.LOW,
                eu_ai_act=EUAIActClass.LIMITED,
                status=ModelStatus.ACTIVE,
                owner="Compliance",
                validator="Model Risk Team",
                validated_at=datetime(2025, 3, 15),
            ),
            ModelRecord(
                mr_reference="MR-2026-049",
                model_name="Payment Fraud Detector",
                chapter=8,
                ss1_23_risk=RiskRating.MEDIUM,
                eu_ai_act=EUAIActClass.HIGH_RISK,
                status=ModelStatus.ACTIVE,
                owner="Fraud Operations",
                validated_at=datetime(2025, 2, 1),
            ),
            ModelRecord(
                mr_reference="MR-2026-052",
                model_name="Cash Flow Forecaster",
                chapter=9,
                ss1_23_risk=RiskRating.MEDIUM,
                eu_ai_act=EUAIActClass.LIMITED,
                status=ModelStatus.ACTIVE,
                owner="Treasury",
                validated_at=datetime(2025, 4, 1),
            ),
        ]
        for m in models:
            self._models[m.mr_reference] = m
            self._validations[m.mr_reference] = []
