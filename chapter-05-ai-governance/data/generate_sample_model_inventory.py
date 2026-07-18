"""
data/generate_sample_model_inventory.py
Generate AWB model inventory with 8 models across risk ratings.
Chapter 5: Model Risk Management (PRA SS1/23)

Models:
  1. MR-2022-015 — SME Credit Scoring Model (CRITICAL)
  2. MR-2023-021 — Retail Mortgage PD Model (HIGH)
  3. MR-2024-008 — Fraud Detection Engine (HIGH)
  4. MR-2026-030 — Customer Service Chatbot (MEDIUM)
  5. MR-2026-035 — Credit Document Analyser (HIGH)
  6. MR-2026-036 — Automated Credit Decision Workflow (HIGH)
  7. MR-2026-039 — ICAAP Liquidity Forecasting Model (MEDIUM)
  8. MR-2026-040 — COREP Regulatory Reporting Automation (LOW)

Run with: python data/generate_sample_model_inventory.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_governance.model_inventory import (
    ModelCard, ModelInventory, RiskRating, EUAIActClassification,
    ValidationStatus, ModelStatus,
)


def build_extended_inventory() -> ModelInventory:
    """Build the full 8-model AWB inventory for Chapter 5."""
    inventory = ModelInventory()

    # 1. SME Credit Scoring — CRITICAL
    inventory.register(ModelCard(
        model_id="MR-2022-015",
        model_name="AWB SME Credit Scoring Model",
        version="3.1.0",
        purpose="Score SME loan applications using financial ratios and credit bureau data.",
        model_type="Logistic Regression + XGBoost ensemble",
        inputs=["Financial ratios", "Credit bureau score", "Industry sector", "Years trading"],
        outputs=["Credit score (0–1000)", "Risk band (A–E)", "Rejection flag"],
        limitations=["Trained on 2019–2022 data", "Does not include qualitative factors"],
        risk_rating=RiskRating.CRITICAL,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Credit Risk",
        developer="AWB Quantitative Analytics",
        validation_status=ValidationStatus.OVERDUE,
        last_validated=datetime.date(2025, 9, 1),
        next_validation_due=datetime.date(2026, 3, 1),
        validation_frequency_months=6,
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2022, 8, 15),
        pra_ss1_23_compliant=True,
        monitoring_plan="Monthly GINI/PSI; semi-annual validation; annual IRB review.",
    ))

    # 2. Retail Mortgage PD Model — HIGH
    inventory.register(ModelCard(
        model_id="MR-2023-021",
        model_name="AWB Retail Mortgage PD Model",
        version="2.0.0",
        purpose="Estimate probability of default for retail mortgage applications (IRB Advanced).",
        model_type="Logistic Regression (IRB PD model)",
        inputs=["LTV ratio", "Income multiples", "Employment status", "Credit history"],
        outputs=["PD estimate (%)", "IRB risk weight", "Regulatory capital requirement"],
        limitations=[
            "Validated on 2010–2023 UK mortgage book",
            "Stress-tested for +300bps rate shock; extreme scenarios may exceed bounds",
        ],
        risk_rating=RiskRating.HIGH,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Retail Credit Risk",
        developer="AWB Quantitative Analytics",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2025, 11, 1),
        next_validation_due=datetime.date(2026, 5, 1),
        validation_frequency_months=6,
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2023, 4, 1),
        pra_ss1_23_compliant=True,
        regulatory_approval_ref="CC-2023-067",
        monitoring_plan="Monthly PSI; quarterly GINI; annual PRA IRB submission.",
    ))

    # 3. Fraud Detection Engine — HIGH
    inventory.register(ModelCard(
        model_id="MR-2024-008",
        model_name="AWB Real-Time Fraud Detection Engine",
        version="1.5.0",
        purpose="Detect fraudulent transactions in real-time across AWB payment channels.",
        model_type="Gradient Boosting + Anomaly Detection (Isolation Forest)",
        inputs=["Transaction amount", "Merchant category", "Device fingerprint", "Geolocation"],
        outputs=["Fraud probability score", "Decision (ALLOW/BLOCK/REVIEW)", "Alert reason codes"],
        limitations=[
            "False positive rate ~0.3% may cause customer friction",
            "Novel fraud patterns may evade detection until retraining",
        ],
        risk_rating=RiskRating.HIGH,
        eu_ai_act_classification=EUAIActClassification.MINIMAL,
        owner="Head of Financial Crime",
        developer="AWB Financial Crime Technology",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 1, 10),
        next_validation_due=datetime.date(2026, 7, 10),
        validation_frequency_months=6,
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2024, 3, 1),
        pra_ss1_23_compliant=True,
        monitoring_plan="Daily false positive/negative rates; weekly precision/recall; monthly retraining review.",
    ))

    # 4. Customer Service Chatbot — MEDIUM / LIMITED
    inventory.register(ModelCard(
        model_id="MR-2026-030",
        model_name="AWB Customer Service Intent Classifier",
        version="1.2.0",
        purpose="Classify inbound customer queries to route to appropriate service channel.",
        model_type="Gemini 3.5 Flash (LLM — intent classification)",
        inputs=["Customer message text", "Session context", "Channel identifier"],
        outputs=["Intent label", "Confidence score", "Routing action"],
        limitations=["May misclassify complex multi-intent queries"],
        risk_rating=RiskRating.MEDIUM,
        eu_ai_act_classification=EUAIActClassification.LIMITED,
        owner="Head of Digital Banking",
        developer="AWB Digital Products Team",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 1, 15),
        next_validation_due=datetime.date(2027, 1, 15),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2025, 6, 1),
        pra_ss1_23_compliant=True,
        monitoring_plan="Monthly accuracy; weekly volume/latency; quarterly FCA outcome review.",
    ))

    # 5. Credit Document Analyser — HIGH
    inventory.register(ModelCard(
        model_id="MR-2026-035",
        model_name="AWB Credit Document Analyser",
        version="1.0.0",
        purpose="Extract structured financial data from credit documents using LLM reasoning.",
        model_type="Gemini 3.1 Pro (LLM — document extraction)",
        inputs=["PDF documents", "Scanned financial statements", "Credit applications"],
        outputs=["Extracted financial metrics", "Entity names", "Validation flags"],
        limitations=["Requires human review for credit decisions ≥ £500,000"],
        risk_rating=RiskRating.HIGH,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Credit Risk",
        developer="AWB AI Platform Team",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 2, 1),
        next_validation_due=datetime.date(2026, 8, 1),
        validation_frequency_months=6,
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 2, 15),
        pra_ss1_23_compliant=True,
        monitoring_plan="Bi-annual validation; monthly extraction accuracy monitoring.",
    ))

    # 6. Automated Credit Decision Agent — HIGH
    inventory.register(ModelCard(
        model_id="MR-2026-036",
        model_name="AWB Credit Decision Agent",
        version="1.0.0",
        purpose="Orchestrate multi-step credit assessment with AI agent.",
        model_type="Gemini 3.1 Pro (ReAct agent) + Gemini 3.5 Flash (drafting)",
        inputs=["Credit application documents", "T24 exposure data", "Financial statements"],
        outputs=["Credit policy assessment", "Risk rating", "Credit memo", "APPROVE/REFER/DECLINE"],
        limitations=["Human oversight mandatory for facilities ≥ £500,000 (EU AI Act Article 14)"],
        risk_rating=RiskRating.HIGH,
        eu_ai_act_classification=EUAIActClassification.HIGH_RISK,
        owner="Head of Credit Risk",
        developer="AWB AI Platform Team",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 3, 1),
        next_validation_due=datetime.date(2026, 9, 1),
        validation_frequency_months=6,
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 3, 10),
        pra_ss1_23_compliant=True,
        regulatory_approval_ref="CC-2026-042",
        monitoring_plan="Monthly recommendation accuracy; quarterly Credit Committee review.",
    ))

    # 7. ICAAP Liquidity Forecasting Model — MEDIUM
    inventory.register(ModelCard(
        model_id="MR-2026-039",
        model_name="AWB ICAAP Liquidity Stress Forecasting",
        version="1.0.0",
        purpose="Generate 3-year liquidity forecasts for ICAAP submission under PRA stress scenarios.",
        model_type="Time-series ensemble (ARIMA + LSTM)",
        inputs=["Historical balance sheet data", "PRA stress scenarios", "Macroeconomic forecasts"],
        outputs=["Liquidity coverage ratio forecast", "Net stable funding ratio", "Stress scenario P&L"],
        limitations=["Forecasts beyond 18 months carry material uncertainty"],
        risk_rating=RiskRating.MEDIUM,
        eu_ai_act_classification=EUAIActClassification.MINIMAL,
        owner="Chief Risk Officer",
        developer="AWB Treasury Analytics",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 2, 28),
        next_validation_due=datetime.date(2027, 2, 28),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 3, 1),
        pra_ss1_23_compliant=True,
        monitoring_plan="Quarterly forecast vs actual; annual ICAAP validation.",
    ))

    # 8. COREP Regulatory Reporting Automation — LOW
    inventory.register(ModelCard(
        model_id="MR-2026-040",
        model_name="AWB COREP Regulatory Reporting Automation",
        version="1.0.0",
        purpose="Automate generation of COREP/FINREP EBA XBRL reports from T24 data.",
        model_type="Rules-based ETL pipeline (no ML component)",
        inputs=["T24 general ledger data", "EBA XBRL taxonomy", "Capital calculation inputs"],
        outputs=["COREP capital adequacy reports", "FINREP financial statements", "EBA XBRL packages"],
        limitations=["Dependent on T24 data quality; manual reconciliation required for exceptions"],
        risk_rating=RiskRating.LOW,
        eu_ai_act_classification=EUAIActClassification.MINIMAL,
        owner="Head of Regulatory Reporting",
        developer="AWB Finance Technology",
        validation_status=ValidationStatus.VALIDATED,
        last_validated=datetime.date(2026, 3, 5),
        next_validation_due=datetime.date(2027, 3, 5),
        status=ModelStatus.PRODUCTION,
        deployed_date=datetime.date(2026, 3, 15),
        pra_ss1_23_compliant=True,
        monitoring_plan="Annual reconciliation review; quarterly data quality checks.",
    ))

    return inventory


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate AWB model inventory.")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    inventory = build_extended_inventory()
    data = inventory.export_to_dict()
    summary = inventory.summary()

    output = {"summary": summary, "models": data}
    json_str = json.dumps(output, indent=2)

    if args.output:
        Path(args.output).write_text(json_str)
        print(f"✅ Model inventory written to {args.output} ({len(data)} models)")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
