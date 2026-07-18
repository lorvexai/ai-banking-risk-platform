# AI for Financial Risk — Book + Code Project

## What This Project Is
- **Book**: "AI for Financial Risk, Compliance and Regulatory Reporting: The Enterprise Implementation Guide" by Sree Kotha
- **Companion repo**: `ai-banking-risk-platform` — 16 chapters, 23 production AI systems, 65,768+ lines of Python 3.11+
- **Current book version**: AI_For_Financial_Risk_v24.docx (in parent folder `C:\Users\sree_ai\Final\Final\`)
- **GitHub**: https://github.com/lorvenio/ai-banking-risk-platform

## Critical Rules (read before every task)
1. Every code or architecture change must be LLM/AI-based — no traditional direct code
2. When adding/updating use cases or case studies, update BOTH the book (.docx) and the code
3. Always use the docx skill when editing the book — never edit .docx with raw text tools
4. Never renumber chapters without explicit instruction — all 16 chapters are intentional
5. Code blocks >20 lines belong on GitHub, not in the book — use a GitHub reference box instead

## Agentic Pipeline Pattern (Chapters 3–16)
Every chapter has one `agentic_*.py` — a LangGraph StateGraph:
```
START -> Agent1 -> Agent2 -> Agent3 -> Agent4 -> Agent5 -> HITL -> END
```
- Agents 1-3: `google/gemini-3.5-flash`
- Agent 4:    `google/gemini-3.1-pro`
- Agent 5 + HITL: `anthropic/claude-sonnet-4-6`
- TOKEN_BUDGET_PER_RUN = 50,000 | COST_BUDGET_GBP_PER_RUN = 2.50
- DORA Art.28: Gemini 68% | Anthropic 17% | OpenAI 15% (no provider > 70%)

## Repository Structure
```
ai-banking-risk-platform/
├── CLAUDE.md                          ← you are here
├── README.md
├── chapter-01-ai-transformation/
├── chapter-02-genai-llms/
├── chapter-03-ai-agents/
│   └── credit_agent/
│       ├── agentic_ai_patterns.py     # Section 3.9A
│       └── mcp_servers.py             # Section 3.9B
├── chapter-04-rag-systems/
├── chapter-05-ai-governance/
│   └── model_governance/
│       └── operational_mrm.py         # Section 5.8A — SS1/23 gate, drift, prompts
├── chapter-06-credit-risk/
│   └── agentic_cim.py             # Sections 6.7A, 6.8A
├── chapter-07-market-risk/agentic_market_risk.py    # 7.3A
├── chapter-08-operational-risk/agentic_op_risk.py   # 8.3A
├── chapter-09-liquidity-risk/agentic_liquidity_risk.py  # 9.3A
├── chapter-10-model-risk/agentic_model_risk.py      # 10.3A
├── chapter-11-regulatory-compliance/agentic_regulatory_compliance.py  # 11.3A
├── chapter-12-aml-kyc/agentic_aml_kyc.py            # 12.3A
├── chapter-13-enterprise-architecture/agentic_enterprise_architecture.py  # 13.3A
├── chapter-14-mlops-llmops/agentic_mlops_llmops.py  # 14.3A
├── chapter-15-data-infrastructure/agentic_data_infrastructure.py  # 15.3A
└── chapter-16-integrated-platform/agentic_integrated_platform.py # 16.3A
```

## AWB Model Registry (23 systems)
| MR Reference   | System                      | Ch | SS1/23  | EU AI Act  |
|----------------|-----------------------------|----|---------|------------|
| MR-2026-035    | Credit Document Analyser    |  2 | MEDIUM  | HIGH-RISK  |
| MR-2026-036    | SME Financial Analyser      |  2 | MEDIUM  | HIGH-RISK  |
| MR-2026-037    | Credit Decision Agent       |  3 | HIGH    | HIGH-RISK  |
| MR-2026-038    | Treasury Operations Agent   |  3 | HIGH    | HIGH-RISK  |
| MR-2026-039    | Regulatory Knowledge Asst   |  4 | LOW     | Limited    |
| MR-2026-040    | AI Governance Platform      |  5 | LOW     | Not scope  |
| MR-2026-041    | Payment Fraud Detector      |  8 | MEDIUM  | HIGH-RISK  |
| MR-2026-042    | Op Loss Event NLP           |  8 | LOW     | Limited    |
| MR-2026-043    | Credit App Fraud Scorer     |  8 | MEDIUM  | HIGH-RISK  |
| MR-2026-044    | Cash Flow Forecaster        |  9 | MEDIUM  | Limited    |
| MR-2026-045    | LCR Stress Tester           |  9 | HIGH    | HIGH-RISK  |
| MR-2026-046    | Credit PD Model (XGBoost)   |  6 | HIGH    | HIGH-RISK  |
| MR-2026-047    | Market VaR Engine           |  7 | HIGH    | HIGH-RISK  |
| MR-2026-048    | FRTB IMA Calculator         |  7 | HIGH    | HIGH-RISK  |
| MR-2026-049    | AML Transaction Monitor     | 12 | HIGH    | HIGH-RISK  |
| MR-2026-050    | KYC Risk Scorer             | 12 | HIGH    | HIGH-RISK  |
| MR-2026-051    | Regulatory Horizon Scanner  | 11 | LOW     | Limited    |
| MR-2026-052    | ICAAP Stress Model          | 13 | HIGH    | HIGH-RISK  |
| MR-2026-053    | Customer Churn Predictor    | 14 | LOW     | Limited    |
| MR-2026-054    | Feature Store PIT Engine    | 15 | MEDIUM  | Limited    |
| MR-2026-055    | BCBS 239 Quality Monitor    | 15 | LOW     | Not scope  |
| MR-2026-058    | Prompt Registry             | 14 | LOW     | Not scope  |
| MR-2026-074-IP | Integrated Platform Agent   | 16 | HIGH    | HIGH-RISK  |

## Regulatory Framework (key constraints for all code)
- PRA SS1/23: 4-gate deployment (AUC-ROC≥0.70, PSI≤0.20, governance docs, output floor≥55%)
- DORA Art.28: LLM concentration cap — no single provider > 70%
- DORA Art.11: RTO ≤ 120 min (P1), PRA notification ≤ 4 hours
- CRR3 Art.72e: Output floor ≥ 55% (Gate 4)
- FCA PS22/9: Consumer Duty — fairness parity ±5pp
- BCBS 239: Data quality ≥ 9.2/10
- FCA COBS 9: 7-year audit log retention
- POCA 2002 s.333A: SAR history — MLRO access only

## Book Version History
- v22.docx — base (all 16 chapters, Sections 7.3A–16.3A inserted)
- v23.docx — added missing Sections 3.9A, 3.9B (Ch3), 5.8A (Ch5)
- v24.docx — CURRENT: long code blocks replaced with GitHub references (~608 pages)

## GitHub Reference Format (use this when replacing code in book)
```
— Key concept (abbreviated) —
[2-4 illustrative lines]

→  Full implementation on GitHub: chapter-XX-name/path/to/file.py
   MR-2026-XXX  |  ~NNN lines  |  https://github.com/lorvenio/ai-banking-risk-platform
   Clone: git clone https://github.com/lorvenio/ai-banking-risk-platform
```

## Common Tasks
- **Edit book chapter**: Use docx skill → open v24.docx → find chapter → edit → save as v25.docx
- **Add new code**: Create file in correct chapter folder, update chapter README, add GitHub ref to book
- **Add agentic section**: Follow pattern in agentic_credit_risk.py — 5 agents + HITL, same LLM allocation
- **Run tests**: `pytest chapter-*/*/tests/ Chapter-*/*/tests/ -v -k "not live"`
