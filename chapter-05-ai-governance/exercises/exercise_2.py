"""
exercise_2.py
Chapter 5: AI Governance, Model Risk, and Regulatory Framework
Exercise 5.2: Build the Full Governance Lifecycle for MR-2026-037

Difficulty: ★★★★☆ | Estimated time: 45 minutes

Task:
    Register the AWB Credit Decision Agent (MR-2026-037) in the
    governance platform, write its PRA SS1/23 model card, assign the
    correct Impact × Complexity risk rating, and run the approval
    workflow state machine from DRAFT through to ERCC_APPROVED.

Requirements:
    1. Populate all mandatory model card fields including the EU AI Act
       Annex III §5b classification (HIGH_RISK).
    2. Correctly trigger the HIGH-risk validation pathway — not the
       LOW-risk fast-track.
    3. Generate a formatted ERCC summary report via the governance API.
    4. All 71 tests in the suite must continue to pass after your
       additions (run: python -m pytest tests/ -v).

British English throughout. GBP primary currency.
Max 60 characters per line (Kindle constraint).

Solution:
    github.com/lorvenio/ai-banking-risk-platform
    /chapter_05/solutions/

Regulatory context:
    PRA SS1/23 s2.1  — Model inventory completeness
    PRA SS1/23 s4.1  — Independent validation
    EU AI Act Art.14 — Human oversight for high-risk AI
    EU AI Act Ann.III §5b — Credit-scoring AI classification
    DORA Art.30      — Third-party ICT vendor governance
"""

from __future__ import annotations

import datetime
import sys
import os

# Add parent directory to path so imports work
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


# ---------------------------------------------------------------------------
# STEP 1 — Create the ModelCard for MR-2026-037
# ---------------------------------------------------------------------------
# Complete all fields below.
# MR-2026-037 = AWB Credit Decision Agent (Chapter 3)
# Risk rating: HIGH (PRA SS1/23 — direct credit decision influence)
# EU AI Act: HIGH_RISK (Annex III §5b — credit scoring)
# Human oversight: mandatory for facilities ≥ £500,000 (Art. 14)
# ---------------------------------------------------------------------------

def build_credit_agent_model_card() -> ModelCard:
    """
    Build the PRA SS1/23 model card for the AWB Credit
    Decision Agent (MR-2026-037).

    Returns a fully populated ModelCard ready for
    registration in the AWB governance platform.
    """
    # TODO: Replace each None / placeholder below
    # with the correct value.
    return ModelCard(
        model_id="MR-2026-037",
        model_name="AWB Credit Decision Agent",
        version="1.0.0",
        purpose=(
            # TODO: Write a 1–2 sentence purpose statement
            # covering what the agent does and its role
            # in the AWB credit workflow.
            "TODO — describe the agent purpose here."
        ),
        model_type=(
            # TODO: Specify the LLM + agent architecture.
            # Hint: ReAct agent, Gemini 3.5 Flash/Pro.
            "TODO — model type"
        ),
        inputs=[
            # TODO: List at least 3 inputs.
            # Hint: credit application documents, T24
            # exposure data, financial statements.
            "TODO — input 1",
        ],
        outputs=[
            # TODO: List at least 3 outputs.
            # Hint: credit policy assessment, risk
            # rating, credit memorandum, recommendation.
            "TODO — output 1",
        ],
        limitations=[
            # TODO: List at least 3 limitations.
            # Required by PRA SS1/23 s3.2 and
            # EU AI Act Art.13 (transparency).
            # Hint: human oversight threshold, T24
            # data dependency, qualitative factors.
            "TODO — limitation 1",
        ],
        # TODO: Set the correct PRA SS1/23 risk rating.
        # Hint: this model makes credit recommendations
        # — what tier does that imply?
        risk_rating=None,  # Replace with RiskRating.???

        # TODO: Set the correct EU AI Act classification.
        # Hint: credit-scoring AI → Annex III §5b.
        eu_ai_act_classification=None,  # EUAIActClassification.???

        owner="Head of Credit Risk",
        developer="AWB AI Platform Team",
        model_risk_contact="Model Risk — Bristol",

        # TODO: Set validation status.
        # New model entering the workflow starts here.
        validation_status=None,  # ValidationStatus.???

        # TODO: Set the dates.
        last_validated=None,  # datetime.date(2026, ?, ?)
        next_validation_due=None,  # datetime.date(?, ?, ?)

        # TODO: Set the model status.
        # The agent is in ERCC approval — what status?
        status=None,  # ModelStatus.???

        # TODO: Deployed date
        deployed_date=None,  # datetime.date(2026, ?, ?)

        pra_ss1_23_compliant=True,

        monitoring_plan=(
            # TODO: Write a monitoring plan.
            # Must reference: monthly accuracy review,
            # human override rate, EU AI Act Art.14
            # compliance log, quarterly Credit Committee.
            "TODO — monitoring plan"
        ),
    )


# ---------------------------------------------------------------------------
# STEP 2 — Register in the AWB governance inventory
# ---------------------------------------------------------------------------

def register_credit_agent(
    inventory: ModelInventory,
) -> ModelCard:
    """
    Register MR-2026-037 in the AWB model inventory.

    Args:
        inventory: The AWB model inventory instance.

    Returns:
        The registered ModelCard.

    Raises:
        KeyError: If model_id already registered.
    """
    card = build_credit_agent_model_card()

    # TODO: Register the card in the inventory.
    # Hint: inventory.register(card)
    # YOUR CODE HERE

    return card


# ---------------------------------------------------------------------------
# STEP 3 — Validate: HIGH-risk pathway check
# ---------------------------------------------------------------------------

def assert_high_risk_pathway(card: ModelCard) -> None:
    """
    Assert that MR-2026-037 triggers the HIGH-risk
    validation pathway (not the LOW-risk fast-track).

    A HIGH-risk model requires:
    - Independent 2nd-line validation (not self-cert)
    - ERCC approval (not Model Risk Manager sign-off)
    - EU AI Act conformity assessment

    Raises:
        AssertionError: If pathway requirements not met.
    """
    # TODO: Add assertions to confirm HIGH-risk pathway.
    # Hints:
    #   card.risk_rating == RiskRating.HIGH
    #   card.eu_ai_act_classification ==
    #       EUAIActClassification.HIGH_RISK
    #   card.requires_eu_ai_act_conformity_assessment()
    # YOUR CODE HERE
    pass


# ---------------------------------------------------------------------------
# STEP 4 — Generate ERCC summary report
# ---------------------------------------------------------------------------

def generate_ercc_summary(
    inventory: ModelInventory,
) -> str:
    """
    Generate a formatted ERCC summary report covering
    all models in the inventory.

    The report must include for each model:
    - Model ID and name
    - Risk rating and EU AI Act classification
    - Validation status and next review date
    - PRA SS1/23 compliance flag

    Args:
        inventory: The populated AWB inventory.

    Returns:
        A formatted multi-line string suitable for
        pasting into the monthly ERCC pack.
    """
    # TODO: Build and return the ERCC summary string.
    # Hint: use inventory.export_to_dict() to get all
    # models, then format each into a report row.
    # YOUR CODE HERE
    lines = [
        "AWB AI Governance Platform",
        "ERCC Model Risk Summary — June 2026",
        "=" * 50,
        "",
    ]

    # TODO: Add one line per model showing:
    # MR-2026-NNN | Model Name | RISK | STATUS
    for model_dict in inventory.export_to_dict():
        # YOUR CODE HERE
        lines.append(
            f"TODO: format {model_dict['model_id']}"
        )

    lines.append("")
    lines.append(
        f"Total models registered: {len(inventory)}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STEP 5 — Run the full lifecycle (entry point)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the complete governance lifecycle for
    MR-2026-037 and print the ERCC summary.

    Expected output:
    - Confirmation of registration
    - HIGH-risk pathway assertions passing
    - Full ERCC report to stdout
    """
    print("AWB Governance Platform — Exercise 5.2")
    print("Building inventory...")
    # NOTE: The base inventory already contains
    # MR-2026-035, -036, -038, -039 from Chs 1-5.
    # MR-2026-037 is deliberately excluded so you
    # can register it as the exercise task.
    inventory = ModelInventory()
    base = build_awb_model_inventory()
    for m in base.export_to_dict():
        if m["model_id"] != "MR-2026-037":
            inventory.register(
                base.get(m["model_id"])
            )
    print(f"  Base inventory: {len(inventory)} models")
    print("  (MR-2026-037 not yet registered)")

    print("Registering MR-2026-037...")
    # TODO: Call register_credit_agent(inventory)
    # and capture the returned card.
    # YOUR CODE HERE
    card = None  # Replace with your call

    if card is not None:
        print(
            f"  Registered: {card.model_id} "
            f"({card.model_name})"
        )
        print(
            f"  Risk rating: {card.risk_rating}"
        )
        print(
            f"  EU AI Act: "
            f"{card.eu_ai_act_classification}"
        )

    print("Validating HIGH-risk pathway...")
    # TODO: Call assert_high_risk_pathway(card)
    # YOUR CODE HERE
    print("  Pathway checks: TODO")

    print("\nGenerating ERCC summary report...")
    report = generate_ercc_summary(inventory)
    print(report)

    print("\nExercise 5.2 complete.")
    print(
        "Run: python -m pytest tests/ -v "
        "to confirm 71 tests still pass."
    )


if __name__ == "__main__":
    main()
