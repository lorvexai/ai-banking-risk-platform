"""
exercise_2_solution.py
Chapter 5 — Exercise 5.2 Reference Solution

AWB Credit Decision Agent (MR-2026-037) full governance
lifecycle: registration, HIGH-risk pathway validation,
ERCC summary report generation.

Regulatory context:
    PRA SS1/23 s2.1  — Model inventory completeness
    PRA SS1/23 s4.1  — Independent validation
    EU AI Act Art.14 — Human oversight requirement
    EU AI Act Ann.III §5b — HIGH-RISK classification
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


def build_credit_agent_model_card() -> ModelCard:
    """
    PRA SS1/23 model card for AWB Credit Decision Agent
    (MR-2026-037).

    HIGH-RISK under PRA SS1/23 (direct credit influence)
    and EU AI Act Annex III §5b (credit-scoring AI).
    Human oversight mandatory for facilities ≥ £500,000
    per EU AI Act Article 14.
    """
    return ModelCard(
        model_id="MR-2026-037",
        model_name="AWB Credit Decision Agent",
        version="1.0.0",
        purpose=(
            "Orchestrate multi-step credit assessment "
            "using a ReAct AI agent: document analysis, "
            "credit policy rule checking, covenant "
            "assessment, RWA forecasting (CRR3 Art.153),"
            " and credit memorandum drafting. Integrates"
            " outputs from MR-2026-035 and MR-2026-038."
        ),
        model_type=(
            "Gemini 3.5 Flash — ReAct agent "
            "(LangGraph state machine); "
            "Gemini 3.1 Pro for complex reasoning steps"
        ),
        inputs=[
            "Credit application documents (PDF)",
            "T24 exposure and limit data (REST API)",
            "Applicant financial statements",
            "MR-2026-035 extraction output (JSON)",
            "MR-2026-038 regulatory context (RAG)",
        ],
        outputs=[
            "Credit policy assessment (PASS/REFER/FAIL)",
            "PRA SS1/23 risk rating (1–10 scale)",
            "RWA forecast (CRR3 Article 153 IRB)",
            "Credit memorandum (Markdown + PDF)",
            "Recommendation: APPROVE/REFER/DECLINE",
        ],
        limitations=[
            "Human oversight mandatory for facilities "
            "≥ £500,000 (EU AI Act Article 14)",
            "Cannot assess qualitative management "
            "quality or character risk",
            "RWA forecasting requires valid T24 "
            "exposure data — degrades if T24 feed fails",
            "Dependent on MR-2026-035 extraction "
            "accuracy; errors propagate downstream",
            "Prompt injection risk from ingested "
            "documents — sanitiser runs at ingest time",
        ],
        risk_rating=RiskRating.HIGH,
        eu_ai_act_classification=(
            EUAIActClassification.HIGH_RISK
        ),
        owner="Head of Credit Risk",
        developer="AWB AI Platform Team",
        model_risk_contact="Model Risk — Bristol",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 3, 10),
        next_validation_due=datetime.date(2026, 9, 10),
        validation_frequency_months=6,
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 3, 20),
        pra_ss1_23_compliant=True,
        regulatory_approval_ref="ERCC-2026-037",
        monitoring_plan=(
            "Monthly recommendation accuracy review "
            "vs human decisions on same cases; "
            "human override rate tracking (target <15%)"
            "; EU AI Act Art.14 compliance log; "
            "quarterly Credit Committee review; "
            "bi-annual full 2nd-line validation."
        ),
        change_log=[
            "2026-03-10 — Initial validation by "
            "Model Risk (2nd line). PASS.",
            "2026-03-18 — ERCC approval granted. "
            "Ref: ERCC-2026-037.",
            "2026-03-20 — Production deployment. "
            "Human oversight gate active.",
        ],
    )


def register_credit_agent(
    inventory: ModelInventory,
) -> ModelCard:
    """Register MR-2026-037 in the AWB inventory."""
    card = build_credit_agent_model_card()
    inventory.register(card)
    return card


def assert_high_risk_pathway(card: ModelCard) -> None:
    """
    Assert HIGH-risk validation pathway requirements.

    HIGH-risk models require independent 2nd-line
    validation, ERCC approval, and EU AI Act conformity
    assessment — never the LOW-risk fast-track.
    """
    assert card.risk_rating == RiskRating.HIGH, (
        f"Expected HIGH, got {card.risk_rating}. "
        "Credit decision agents must be HIGH risk "
        "under PRA SS1/23 due to direct credit impact."
    )

    assert (
        card.eu_ai_act_classification
        == EUAIActClassification.HIGH_RISK
    ), (
        "Credit-scoring AI requires HIGH_RISK "
        "classification under EU AI Act Annex III §5b."
    )

    assert card.requires_eu_ai_act_conformity_assessment(), (
        "HIGH_RISK models must complete conformity "
        "assessment before August 2026 (EU AI Act)."
    )

    assert card.pra_ss1_23_compliant, (
        "Model card must confirm PRA SS1/23 compliance"
        " before ERCC approval."
    )

    print("  ✅ Risk rating: HIGH (correct)")
    print("  ✅ EU AI Act: HIGH_RISK (Annex III §5b)")
    print("  ✅ Conformity assessment required")
    print("  ✅ PRA SS1/23 compliant")


def generate_ercc_summary(
    inventory: ModelInventory,
) -> str:
    """
    Generate a formatted ERCC model risk summary.

    Covers all registered models with risk rating,
    validation status, and compliance flags.
    """
    lines = [
        "AWB AI Governance Platform",
        "ERCC Model Risk Summary — June 2026",
        "Prepared by: Model Risk Function",
        "=" * 60,
        "",
        f"{'ID':<16} {'Model':<32} "
        f"{'Risk':<9} {'Status':<22} {'SS1/23'}",
        "-" * 60,
    ]

    for m in inventory.export_to_dict():
        compliant = "✅" if m.get(
            "pra_ss1_23_compliant", False
        ) else "⚠️"
        # Truncate model name for 60-char line limit
        name = m["model_name"][:30]
        lines.append(
            f"{m['model_id']:<16} {name:<32} "
            f"{m['risk_rating']:<9} "
            f"{m['status']:<22} {compliant}"
        )

    lines.append("-" * 60)
    lines.append("")

    high_risk = [
        m for m in inventory.export_to_dict()
        if m["risk_rating"] in ("HIGH", "CRITICAL")
    ]
    lines.append(
        f"Total models registered : {len(inventory)}"
    )
    lines.append(
        f"HIGH/CRITICAL risk      : {len(high_risk)}"
    )
    lines.append(
        "EU AI Act deadline      : August 2026"
    )
    lines.append(
        "Next MRC meeting        : Last Thu monthly"
    )

    return "\n".join(lines)


def main() -> None:
    """Run the full MR-2026-037 governance lifecycle."""
    print("AWB Governance Platform — Exercise 5.2")
    print("=" * 45)

    print("\n[1/4] Building base AWB inventory...")
    # Build base inventory (contains MR-2026-035, -036,
    # -038, -039 and others — but NOT -037 yet,
    # since registering it is the exercise task).
    # We create a fresh inventory and add only the
    # models from chapters 1-2 and 4-5 as the base.
    inventory = ModelInventory()
    base = build_awb_model_inventory()
    # Add all except MR-2026-037 (that is our task)
    for m in base.export_to_dict():
        if m["model_id"] != "MR-2026-037":
            inventory.register(base.get(m["model_id"]))
    print(f"      Base inventory: {len(inventory)} models")
    print("      (MR-2026-037 not yet registered)")

    print("\n[2/4] Registering MR-2026-037...")
    card = register_credit_agent(inventory)
    print(
        f"      Registered : {card.model_id}"
    )
    print(
        f"      Model name : {card.model_name}"
    )
    print(
        f"      Risk rating: {card.risk_rating}"
    )
    print(
        f"      EU AI Act  : "
        f"{card.eu_ai_act_classification}"
    )
    print(
        f"      Inventory  : {len(inventory)} models"
    )

    print("\n[3/4] Validating HIGH-risk pathway...")
    assert_high_risk_pathway(card)

    print("\n[4/4] Generating ERCC summary report...")
    report = generate_ercc_summary(inventory)
    print()
    print(report)

    print("\n✅ Exercise 5.2 complete.")
    print(
        "Run: python -m pytest tests/ -v "
        "to confirm 71/71 tests pass."
    )


if __name__ == "__main__":
    main()
