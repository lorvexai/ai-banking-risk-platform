# AI Banking Risk Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Production-ready AI/ML implementations for banking risk, compliance, 
> and regulatory reporting**

Companion code repository for the book **"AI for Financial Risk, Compliance 
and Regulatory Reporting: The Enterprise Implementation Guide"**

## 🎯 What's Included

- ✅ **16 Complete Chapters** - From foundations to production deployment
- ✅ **50+ Production Systems** - Real, deployable implementations
- ✅ **40,000+ Lines of Code** - Tested Python code
- ✅ **5 Risk Domains** - Credit, Market, Operational, Liquidity, Model Risk
- ✅ **Compliance & Regulatory** - AML/KYC, Basel III, GDPR
- ✅ **Enterprise Architecture** - Microservices, MLOps, Data Infrastructure

## Chapter 7 — Market Risk Automation Platform

**AI for Financial Risk, Compliance and Regulatory Reporting**  
*Avon & Wessex Bank plc (AWB) — AWB-AI-2025 Programme*

---

### Overview

This codebase implements the Chapter 7 market risk automation stack for AWB:

| System | Model ID | PRA SS1/23 Risk | EU AI Act |
|--------|----------|-----------------|-----------|
| Real-Time Monte Carlo VaR Engine | MR-2026-046 | HIGH | HIGH-RISK §5b |
| Algorithmic Trading Backtester | MR-2026-047 | MEDIUM | HIGH-RISK §5b |
| CVA Calculator | MR-2026-048 | HIGH | HIGH-RISK §5b |
| VaR Attribution Reporter | MR-2026-046 (sub) | HIGH | EU AI Act Art. 14 |
| SA-FRTB Capital Calculator | MR-2026-046 (sub) | HIGH | Not in scope |

AWB trading book: £800M (equities £200M + rates £400M + FX/derivatives £200M)  
SA-FRTB capital: £42M (equity £12M + rates £18M + FX £8M + credit £4M)  
VaR (June 2026): £3.8M 1-day 99% | Back-test: 3 exceptions/250 days (Green)

**Annual saving:** £926K combined  
**Payback:** avg. 5 months  
**Monthly running cost:** £820 total across all systems

---

### Architecture

```
Market Data + Positions
        │
        ▼
MonteCarloVaREngine (MR-2026-046)
        │
        ├── VaRResult (VaR 95/99, ES, Component VaR)
        │           │
        │           ▼
        │   VaRBackTester ──── Green/Amber/Red
        │           │
        │     Exception? ──── VaRAttributionReporter
        │                          │
        │                    Gemini 3.5 Flash
        │                          │
        │                    AI-ASSISTED report
        │                    (Head of Market Risk
        │                     attestation required)
        │
        ├── SaFrtbCalculator (frtb/frtb_capital.py)
        │   SbM + DRC + RRAO = £42M SA-FRTB capital
        │
Strategy Signals + Prices
        │
        ▼
BacktestEngine (MR-2026-047)
        │
        ├── BacktestResult (Sharpe, VaR, CVaR)
        └── MARComplianceChecker (wash trades, spoofing)

Exposure Profile + PD Term Structure (MR-2026-040)
        │
        ▼
CVACalculator (MR-2026-048)
        │
        └── CVAResult + SA-CVA Capital
```

---

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests (no API key required)
cd chapter-07-market-risk
pytest tests/ -v -k "not live"

# 3. Run all tests including attribution reporter
export ANTHROPIC_API_KEY="your_key_here"
pytest tests/ -v

# 4. Interactive VaR demo
python -c "
import numpy as np
from var_engine.mc_var_engine import MonteCarloVaREngine

corr = np.array([[1.0, 0.3], [0.3, 1.0]])
vols = np.array([0.01, 0.008])
engine = MonteCarloVaREngine(
    corr, vols, ['GIRR_10Y', 'FX_GBPUSD']
)
result = engine.compute_full_var(
    np.array([[100_000, 0], [0, 50_000]])
)
print(f'VaR 99: GBP {result.var_99:,.0f}')
print(f'ES  99: GBP {result.expected_shortfall_99:,.0f}')
"

# 5. SA-FRTB capital demo (AWB June 2026 book)
python -c "
from frtb.frtb_capital import SaFrtbCalculator
calc = SaFrtbCalculator()
result = calc.calculate_total(
    girr_dv01=180_000_000,
    equity_delta=60_000_000,
    fx_delta=53_333_333,
    credit_delta=400_000_000,
    gross_jtd=200_000_000,
    net_jtd=150_000_000,
    exotic_notional=400_000_000,
)
print(f'SA-FRTB TOTAL: GBP {result.total_sa_frtb_gbp:,.0f}')
print(f'  GIRR: GBP {result.girr_capital_gbp:,.0f}')
print(f'  Equity: GBP {result.equity_capital_gbp:,.0f}')
print(f'  FX: GBP {result.fx_capital_gbp:,.0f}')
print(f'  Credit: GBP {result.credit_spread_capital_gbp:,.0f}')
print(f'  DRC: GBP {result.drc_gbp:,.0f}')
print(f'  RRAO: GBP {result.rrao_gbp:,.0f}')
"
```

---

### File Structure

```
chapter-07-market-risk/
├── backtesting/
│   └── backtest_engine.py    # MR-2026-047: backtest + MAR compliance
├── var_engine/
│   └── mc_var_engine.py      # MR-2026-046: Monte Carlo VaR + FRTB back-test
├── var_monitoring/
│   └── var_attribution_reporter.py  # LLM P&L attribution (Gemini 3.5 Flash)
├── cva/
│   └── cva_calculator.py     # MR-2026-048: SA-CVA calculator
├── frtb/
│   └── frtb_capital.py       # SA-FRTB: SbM + DRC + RRAO (CRR3)
├── exercises/
│   ├── var_exercise.py       # Exercise 7.1: 5-asset MC VaR
│   └── exercise_2.py         # Exercise 7.2: CVA + MR-2026-040 integration
├── tests/
│   └── test_chapter_07.py    # 40+ pytest tests (all passing)
├── requirements.txt
└── README.md
```

---

### Regulatory Compliance

| Obligation | Implementation |
|------------|----------------|
| PRA SS1/23 | Model IDs MR-2026-046 (HIGH), MR-2026-047 (MEDIUM), MR-2026-048 (HIGH) |
| CRR3 Art. 325bf/325bg | 250-day FRTB back-testing and traffic light in `VaRBackTester` |
| CRR3 Art. 325a–325bh | SA-FRTB SbM + DRC + RRAO in `SaFrtbCalculator` |
| CRR3 Art. 383 | SA-CVA capital in `CVACalculator` |
| CRR3 Art. 274 | 40% unsecured recovery rate default |
| EU/UK MAR 596/2014 | Wash-trade and spoofing detection in `MARComplianceChecker` |
| EU AI Act Art. 14 | Attribution reports require Head of Market Risk attestation |
| DORA Art. 17 | ICT assets: VAR-2026-046, BT-2026-001, CVA-2026-048 |
| PRA PS17/23 | UK FRTB implementation, January 2025 |

---

### Cost Derivation (GBP, June 2026)

| System | Monthly Cost |
|--------|-------------|
| VaR Engine compute (AWS EC2 c5.2xlarge × 2) | £380 |
| CVA Calculator compute | £180 |
| Backtesting platform compute | £160 |
| LLM attribution reports (20/month × £0.50) | £10 |
| PostgreSQL audit log storage | £55 |
| Redis (VaR cache) | £35 |
| **Total** | **£820/month** |

Annual savings:
- Backtesting (MR-2026-047): £411K/year (analyst time + loss avoidance)
- VaR Engine (MR-2026-046): £277K/year (quant time + capital multiplier risk)
- CVA Calculator (MR-2026-048): £62K/year (quarterly manual calculation replaced)
- **Combined: £750K+/year net of costs**

---

### LLM Selection

**Gemini 3.5 Flash** is AWB's primary model (68% of production calls).  
The VaR attribution reporter uses **Claude Haiku 4.5** for fast,
cost-effective regulatory narrative generation (~£0.50 per report).

All models from the approved June 2026 list only.  
Never use: GPT-4, Claude 3.5 Sonnet, Gemini 3 (deprecated).

---

*AWB = Avon & Wessex Bank plc — entirely fictional.*  
*GitHub: github.com/lorvenio/ai-banking-risk-platform*
