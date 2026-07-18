# Chapter 16 — Building an Integrated AI Risk Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Capstone chapter — unified API gateway, CRO dashboard, and audit log integrating all 23 AWB production AI systems. Includes the agentic platform health pipeline (Section 16.3A).

*Companion code for **"AI for Financial Risk, Compliance and Regulatory Reporting"** | AWB-AI-2025 Programme*

---

## Section 16.3A — Agentic AI Pipeline: Integrated Platform Oversight

`agentic_integrated_platform.py` (1,226 lines) is the capstone agentic
pipeline. It orchestrates five specialist agents over AWB's full 23-system
AI estate, enforcing CET1 capital thresholds and DORA concentration limits
in a single LangGraph StateGraph run.

| Agent | LLM | Responsibility |
|---|---|---|
| PlatformHealthAgent | Gemini 3.5 Flash | Scan all 23 models — uptime, AUC-ROC, PSI |
| CrossDomainRiskAgent | Gemini 3.5 Flash | Detect correlated exposures across risk domains |
| RegulatoryBreachAgent | Gemini 3.5 Flash | DORA Art.28 LLM concentration + PRA SS1/23 gates |
| CapitalAdequacyAgent | Gemini 3.1 Pro | CET1 ratio check — HITL if < 14.5% (Board resolution) |
| PlatformSummaryAgent | Claude Sonnet 4.6 | Board-level narrative + final HITL decision |

**Key constants:**
- `CET1_HITL_THRESHOLD_PCT = 14.5` — mandatory HITL below this
- `DORA_P1_RTO_MINUTES = 120` — P1 incident recovery target
- `DORA_P1_PRA_NOTIFY_HOURS = 4` — PRA notification deadline
- `AWB_LLM_CONCENTRATION = {"google_gemini": 68.0, "anthropic_claude": 17.0, "openai_gpt4o": 15.0}`
  — no provider exceeds the DORA Art.28 70% cap

**Model registration:** MR-2026-074-IP (Integrated Platform Agent)

---

## Chapter 16 — Building an Integrated AI Risk Platform

**AWB-AI-2025 Programme — Capstone Chapter**

### Overview

This code package delivers the integration layer
that unifies all 23 AI systems built across
Chapters 1–15 of *AI for Financial Risk, Compliance
and Regulatory Reporting*.

Bank: Avon & Wessex Bank plc (AWB) — fictional
Assets: £40B | Location: Bristol, UK
Regulator: PRA and FCA
Programme: AWB-AI-2025 (£3.2M, Jan 2025)

### Files

```
chapter-16-integrated-platform/
├── platform/
│   ├── api_gateway.py       # FastAPI gateway
│   │                        # JWT RS256, rate limit,
│   │                        # circuit breakers
│   ├── unified_audit_log.py # FCA COBS 9 audit log
│   │                        # 7-year retention
│   └── dashboard.py         # CRO/CFO dashboard
│                            # capital/liquidity/AML
├── exercises/
│   ├── deploy_service.sh    # Exercise 16.1 starter
│   └── integration_tests.py # Exercise 16.2 starter
├── tests/
│   └── test_platform.py     # 35 unit tests
├── solutions/               # See GitHub
└── requirements.txt
```

### Quick Start

```bash
# Create virtualenv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run unit tests (no live services needed)
pytest tests/test_platform.py -v

# Start API Gateway (dev mode)
uvicorn platform.api_gateway:app \
    --host 0.0.0.0 --port 8090 --reload

# Start CRO Dashboard (dev mode)
uvicorn platform.dashboard:app \
    --host 0.0.0.0 --port 8084 --reload
```

### Exercise 16.1

Deploy the Credit AI service stack to AWS ECS:

```bash
chmod +x exercises/deploy_service.sh
./exercises/deploy_service.sh dev
```

Prerequisites:
- AWS CLI v2 configured for eu-west-2
- ECR images built and pushed
- PostgreSQL audit DB reachable

### Exercise 16.2

Run the full integration test suite:

```bash
export AWB_API_GW_URL=http://localhost:8090
export AWB_DB_URL=postgresql://awb:awb@localhost/awb
export AWB_TEST_JWT=<your-test-token>

pytest exercises/integration_tests.py -v
```

All 47 tests must pass before the platform can
be declared production-ready.

### Model Registry

| Model ID    | System                      | Risk   |
|-------------|-----------------------------|--------|
| MR-2026-035 | Credit Document Analyser    | MEDIUM |
| MR-2026-036 | SME Financial Analyser      | MEDIUM |
| MR-2026-037 | Credit Decision Agent       | HIGH   |
| MR-2026-038 | Regulatory Knowledge Asst.  | LOW    |
| MR-2026-039 | AI Governance Platform      | LOW    |

### Regulatory References

- PRA SS1/23: Model risk management (primary)
- FCA Consumer Duty PS22/9
- CRR3 Art. 429: Leverage ratio
- DORA Art. 11: RTO/RPO targets
- DORA Art. 28: 70% single-provider LLM cap
- FCA COBS 9: 7-year audit log retention
- POCA 2002: AML/SAR obligations

### GitHub

Full solutions and additional code:
github.com/lorvenio/ai-banking-risk-platform

AWB = Avon & Wessex Bank plc — entirely fictional.

### Architecture Diagrams

#### Excalidraw-Style (Hand-Drawn)

![Architecture Diagram](architecture_handdrawn.svg)

#### Mermaid

```mermaid
flowchart TD
  T["chapter-16-integrated-platform Architecture"]
  M1[""]
  T --> M1
  M2["awb_platform"]
  T --> M2
  M3["awb_platform.api_gateway"]
  T --> M3
  M4["awb_platform.dashboard"]
  T --> M4
  M5["awb_platform.unified_audit_log"]
  T --> M5
```


