# Chapter 14 — MLOps and LLMOps for Risk Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> End-to-end MLOps and LLMOps pipelines — Airflow DAGs, SS1/23 4-gate CI/CD, RAGAS monitoring, and prompt version control for all 23 AWB production AI models.

*Companion code for **"AI for Financial Risk, Compliance and Regulatory Reporting"** | AWB-AI-2025 Programme*

---

## Section 14.3A — Agentic AI Pipeline: MLOps Oversight

`agentic_mlops_llmops.py` implements a five-agent LangGraph StateGraph that
continuously monitors the AWB ML estate and enforces SS1/23 deployment gates
without human intervention for routine checks.

| Agent | LLM | Responsibility |
|---|---|---|
| ModelHealthAgent | Gemini 3.5 Flash | Scan 23 models — AUC-ROC, PSI, RAGAS scores |
| DriftDetectionAgent | Gemini 3.5 Flash | Compare reference distributions, raise PSI alerts |
| DeploymentGateAgent | Gemini 3.5 Flash | Block promotions failing SS1/23 4-gate check |
| PromptVersionAgent | Gemini 3.1 Pro | Diff prompt changes, enforce RAGAS ≥ 0.80 |
| MLOpsSummaryAgent | Claude Sonnet 4.6 | Board-ready narrative + HITL decision |

**HITL gate triggers:** AUC-ROC drop > 3 pp, PSI breach (> 0.20), or any
MAJOR prompt version change → HITLDecision.ESCALATE.

**Model registration:** MR-2026-058 (Prompt Registry), MR-2026-059 (MLOps Platform)

---

## Chapter 14 — MLOps and LLMOps for Risk Systems

**Book:** AI for Financial Risk, Compliance and Regulatory Reporting  
**Bank:** Avon & Wessex Bank plc (AWB) — entirely fictional  
**Programme:** AWB-AI-2025 | Namespace: `awb_commons`

---

### Overview

This package implements the AWB MLOps and LLMOps platform
for Chapter 14. It governs all 23 production AI systems
registered in the AWB model registry.

### Structure

```
chapter_14/
├── mlops/
│   ├── airflow_dags.py       # Airflow DAG definitions
│   └── ss1_23_validation.py  # 4-gate CI/CD validation
├── churn/
│   └── feature_engineer.py   # 28-feature engineering
├── llmops/
│   ├── prompt_registry.py    # MAJOR.MINOR.PATCH registry
│   └── ragas_monitor.py      # 5% sampling + auto-rollback
├── tests/
│   └── test_chapter_14.py    # 55+ pytest tests
├── exercises/
│   ├── scoring_dag.py        # Exercise 14.1 starter
│   └── exercise_2.py         # Exercise 14.2 starter
└── solutions/                # Reference implementations
```

### Quick Start

```bash
git clone https://github.com/lorvenio/ai-banking-risk-platform
cd ai-banking-risk-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests (no live APIs required)
pytest chapter_14/tests/ -v -k "not live"
```

### Key Systems

| System | Registry ID | SS1/23 Risk |
|--------|-------------|-------------|
| Customer Churn Predictor | MR-2026-053 | LOW |
| Credit PD Model (XGBoost) | MR-2026-043 | HIGH |
| MLOps Platform | MLO-2026-001 | — |
| Prompt Registry | MR-2026-058 | — |

### Validation Gates (Credit Models)

1. **Gate 1 — Performance**: AUC > champion + 0.02
2. **Gate 2 — Fairness**: FCA PS22/9 parity ±5pp
3. **Gate 3 — Governance**: SS1/23 card + MRC approval
4. **Gate 4 — Data Quality**: GE 24-rule suite pass

### LLMOps Versioning

- **MAJOR**: New output schema → MRC review + 2-wk A/B
- **MINOR**: Additive change → robustness suite + 1-wk A/B
- **PATCH**: Fix → robustness suite + 48hr monitoring

RAGAS thresholds (faithfulness ≥ 0.80 auto-rollback):
see `llmops/ragas_monitor.py`

### Regulatory References

- PRA SS1/23: Model risk management (primary)
- FCA Consumer Duty PS22/9: Gate 2 fairness
- CRR3 Art.92a: Output floor in Gate 1
- FCA COBS 9: 7-year audit retention

### Exchange Rate

USD costs: £1 = $1.27 (June 2026)

---

*AWB is entirely fictional. Nothing here constitutes
regulatory or legal advice.*

### Architecture Diagrams

```mermaid
flowchart TD
  T["chapter-14-mlops-llmops Architecture"]
  M1[""]
  T --> M1
  M2["churn"]
  T --> M2
  M3["churn.feature_engineer"]
  T --> M3
  M4["llmops"]
  T --> M4
  M5["llmops.prompt_registry"]
  T --> M5
  M6["llmops.ragas_monitor"]
  T --> M6
  M7["mlops"]
  T --> M7
  M8["mlops.airflow_dags"]
  T --> M8
  M9["mlops.ss1_23_validation"]
  T --> M9
```


