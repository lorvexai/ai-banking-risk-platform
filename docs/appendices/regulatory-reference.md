# Appendix B — UK/EU Regulatory Reference Guide

> Article-level summaries for the primary UK and EU regulations governing AI in financial services. For authoritative text, refer to the official FCA, PRA, EBA, and European Commission publications.

This appendix provides article-level reference summaries for the primary regulations governing AWB's AI systems. For each regulation, key articles relevant to AI/ML model deployment are listed with their practical implications for AWB. Readers should verify current requirements with the relevant authority; regulatory requirements change frequently.
B.1  PRA SS1/23 — Model Risk Management
Authority:  PRA
Effective date:  May 2023
Scope:  All models used in decision-making at PRA-regulated firms, including AI/ML models.
§3 — Model identification and classification , All models must be registered in a central inventory with risk rating (HIGH/MEDIUM/LOW) based on materiality and complexity. AWB uses a 4×4 Impact × Complexity matrix. Model IDs: MR-2026-035 through MR-2026-039 in Part I; additional IDs in Chs 6–16.
§4 — Model development , Development must follow documented methodology. Training data must be representative, complete, and subject to data quality checks. All assumptions must be documented. Code must be version-controlled.
§5 — Independent model validation , Validation must be conducted by a team independent of development. For HIGH-risk models, validation is mandatory before deployment. AWB's Model Risk team conducts validation; sign-off required before CI/CD pipeline proceeds to production.
§6 — Model use and ongoing monitoring , Models must be monitored against defined performance thresholds. AWB monitors AUC-ROC, PSI, and SHAP stability monthly. Drift above 3σ or PSI >0.20 triggers recalibration.
§7 — Audit and documentation , 7-year retention for all model documentation, validation reports, and decision audit trails. AWB stores in PostgreSQL (AWS RDS, eu-west-2).

B.2  EU AI Act 2024 — High-Risk AI Systems
Authority:  European Commission
Effective date:  August 2024 (phased)
Scope:  AI systems used in the EU. Annex III §5b: credit scoring and AML detection are HIGH-RISK.
Art. 10 — Data governance , Training, validation, and testing data must be relevant, representative, and free from errors. Bias testing required. AWB's Ch14 MLOps pipeline includes automated bias and drift checks at every training run.
Art. 14 — Human oversight , HIGH-RISK systems must enable human review and override. AWB implements mandatory human gates for credit decisions >£500K exposure and agent confidence <0.80. Non-blocking escalation to relationship manager queue.
Art. 17 — Quality management system , Providers of HIGH-RISK systems must establish a QMS covering design, testing, monitoring, and documentation. AWB's AI governance platform (MR-2026-039) serves as the QMS.
Art. 72 — Post-market monitoring , Continuous monitoring of HIGH-RISK systems in production. AWB samples 5% of live RAG queries for automated RAGAS evaluation; credit model performance reviewed monthly by Model Risk team.

B.3  DORA — Digital Operational Resilience Act
Authority:  EBA / European Commission
Effective date:  January 2025
Scope:  All EU/UK financial entities. ICT systems including AI/LLM deployments.
Art. 17 — ICT incident classification , Autonomous agent security incidents (prompt injection, tool hijacking) must be classified as ICT-related incidents. AWB classifies and reports per the DORA incident taxonomy.
Art. 28 — ICT third-party concentration risk , No single ICT provider may exceed 70% of critical functions. AWB's multi-LLM strategy: Gemini 3.5 Flash 68% | Claude Sonnet 4.6 17% | GPT-5.5 15%. All providers below the 70% cap.
Art. 30 — Contractual arrangements , Contracts with ICT providers (Google, Anthropic, OpenAI) must include audit rights, data location clauses, and exit provisions.

B.4  CRR3 / Basel IV — Capital Requirements
Authority:  EBA / PRA
Effective date:  January 2025
Scope:  EU/UK banks. Credit risk, market risk, operational risk, and leverage ratio capital.
Articles 112–191 — Credit risk RWA , Standardised and IRB approaches for credit risk. AWB uses IRB for its corporate credit portfolio. PD, LGD, EAD models require PRA approval and annual validation under SS1/23.
Article 153 — IRB RWA formula , RWA = f(PD, LGD, EAD, M) × 1.06 × correlation factor. AWB's RWAForecastAgent implements this formula for real-time capital estimation in the credit decision workflow.
Article 429 — Leverage ratio , Leverage ratio = Tier 1 Capital ÷ Total Leverage Exposure Measure. Minimum 3% (3.5% for G-SIBs). Quarterly COREP return C 47.00. AWB's reporting platform includes leverage ratio as the fourth pillar.
Articles 316–323 — Operational risk SMA , Standardised Measurement Approach for op risk capital. SMA Capital = BIC × ILM. AWB's AI-enhanced loss data feeds improve ILM accuracy through automated BCBS event categorisation.

B.5  POCA 2002 / JMLSG — AML/KYC
Authority:  HM Government / JMLSG
Effective date:  2002 (ongoing amendments)
Scope:  All UK financial institutions.
POCA s.330 — Failure to disclose , Offence for regulated persons who know or suspect money laundering and fail to disclose. AWB's AML system (MR-2026-061) triggers SAR drafting when ML alert confidence >0.85.
POCA s.333A — Tipping-off prohibition , Offence to disclose that a SAR has been filed. AWB's credit gate prevents credit decisions on SAR-subject borrowers, enforced architecturally before T24 facility creation.
JMLSG Part I — CDD and EDD , Standard CDD for all borrowers. Enhanced Due Diligence for PEPs, high-risk jurisdictions, and complex ownership structures. AWB's KYC system (MR-2026-063) automates CDD via Companies House API and PSC Register tracing to four layers.

B.6  FCA Consumer Duty — PS22/9
Authority:  FCA
Effective date:  July 2023
Scope:  All FCA-regulated firms selling to retail customers.
Four Outcome Areas , (1) Products and services: AI-generated product recommendations must match customer eligibility and risk profile. (2) Price and value: AI pricing must evidence fair value. (3) Consumer understanding: AI explanations must be clear and not misleading. (4) Consumer support: AI support channels must enable customers to act in their interests.
COBS 4 — Explanation requirements , Customers must receive clear explanations of AI-assisted decisions. AWB's credit memo generator and product suitability adviser include SHAP-based explanations in all customer-facing outputs.

B.7  BCBS 239 — Risk Data Aggregation
Authority:  BCBS
Effective date:  January 2013 (G-SIBs); guidance for all significant banks
Scope:  Risk data governance, aggregation, and reporting.
Principles 1–4: Governance and infrastructure , Strong data governance, single authoritative source per risk type, automated data quality checks. AWB's data architecture (Ch15) implements a single Enterprise Risk Data Warehouse as the canonical source for all regulatory reporting.
Principles 5–9: Risk data aggregation , Data must be accurate, complete, timely, and adaptable. AWB's nightly reconciliation against T24 balance sheet totals flags variances >£1M before morning COREP runs.
Principles 10–14: Risk reporting , Risk reports must be accurate, clear, complete, and distributed to appropriate levels. AWB's executive dashboard (Ch16) delivers real-time CRO/CFO-level risk views from the ERDW.

