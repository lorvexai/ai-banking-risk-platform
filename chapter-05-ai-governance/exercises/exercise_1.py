"""
exercise_1.py
Chapter 5: AI Governance, Model Risk, and Regulatory Framework
Exercise 5.1: Register a New Model and Write Its Model Card

Difficulty: ★★★☆☆ | Estimated time: 30 minutes

Task:
    Using the AWB governance platform, register a new model
    for a hypothetical AWB Customer Churn Predictor and
    produce a PRA SS1/23-compliant model card. Then retrieve
    the model and verify that the inventory reports it as
    overdue for validation.

Requirements:
    1. Create a ModelCard with model_id "MR-2026-050" and
       risk_rating MEDIUM.
    2. Register it in the AWB inventory.
    3. Set next_validation_due to yesterday's date so the
       overdue check fires.
    4. Verify the model appears in get_overdue_validations().

British English throughout. GBP primary currency.

Solution:
    github.com/lorvenio/ai-banking-risk-platform
    /chapter_05/solutions/

Regulatory context:
    PRA SS1/23 s2.1  — Model inventory completeness
    PRA SS1/23 s4.2  — Validation frequency by risk tier
"""

from __future__ import annotations

import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_governance.model_inventory import (
    ModelCard,
    ModelInventory,
    ModelStatus,
    RiskRating,
    EUAIActClassification,
    ValidationStatus,
    build_awb_model_inventory,
)


def build_churn_model_card() -> ModelCard:
    """
    Build a model card for a customer churn predictor.

    Model ID: MR-2026-050
    Risk rating: MEDIUM (customer retention impact,
    no direct credit or regulatory decision).

    Returns:
        A populated ModelCard instance.
    """
    yesterday = (
        datetime.date.today() - datetime.timedelta(days=1)
    )

    # TODO: Replace each None with the correct value.
    return ModelCard(
        model_id="MR-2026-050",
        model_name="AWB Customer Churn Predictor",
        version="1.0.0",
        purpose=(
            # TODO: Write a purpose statement.
            "TODO — describe model purpose."
        ),
        model_type=(
            # TODO: Hint: XGBoost binary classifier.
            "TODO — model type"
        ),
        inputs=[
            # TODO: List at least 3 inputs.
            # Hint: transaction recency, product count,
            # digital engagement score, tenure months.
            "TODO — input 1",
        ],
        outputs=[
            # TODO: List at least 2 outputs.
            "TODO — output 1",
        ],
        limitations=[
            # TODO: List at least 2 limitations.
            "TODO — limitation 1",
        ],
        risk_rating=RiskRating.MEDIUM,
        eu_ai_act_classification=(
            EUAIActClassification.LIMITED
        ),
        owner="Head of Retail Banking",
        developer="AWB Data Science Team",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.PENDING,
        last_validated=None,
        # Set overdue: yesterday's date
        next_validation_due=yesterday,
        status=ModelStatus.DRAFT,
        pra_ss1_23_compliant=False,
        monitoring_plan=(
            # TODO: Write a monitoring plan.
            "TODO — monitoring plan."
        ),
    )


def main() -> None:
    """Run Exercise 5.1 workflow."""
    print("AWB Governance Platform — Exercise 5.1")
    inventory = build_awb_model_inventory()
    print(
        f"Starting inventory size: {len(inventory)}"
    )

    # TODO: Build and register the churn model card.
    # YOUR CODE HERE

    # TODO: Verify the model is registered.
    # Hint: use "MR-2026-050" in inventory
    # YOUR CODE HERE

    # TODO: Verify it appears in overdue validations.
    # Hint: inventory.get_overdue_validations()
    # YOUR CODE HERE

    print("Exercise 5.1 complete.")


if __name__ == "__main__":
    main()
