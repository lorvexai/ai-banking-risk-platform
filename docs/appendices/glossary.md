# Appendix A — Glossary of Terms

> This glossary defines 108 key terms used across the 16 chapters of *AI for Financial Risk, Compliance and Regulatory Reporting*. Terms are drawn from AI/ML practice, UK/EU financial regulation, and the AWB reference implementation.

This glossary defines 108 key terms used across the 16 chapters of AI for Financial Risk, Compliance and Regulatory Reporting. Terms are listed alphabetically. Acronyms are listed under their expanded form where possible, and also under the acronym itself for quick lookup.


---

## A

**Adversarial input**  
An input deliberately crafted to mislead an AI model into producing incorrect outputs. Relevant to fraud detection and model security under DORA.

**Agent**  
An AI system that perceives its environment, reasons about it, and takes actions via tools to accomplish a goal. AWB agents follow the ReAct (Reason+Act) pattern.

**AgentRunBudget**  
An AWB dataclass enforcing hard limits (max_tokens, max_tool_calls, max_cost_gbp) on every agent execution. Introduced after the overnight loop war story cost £840 in API fees.

**Airflow**  
Apache Airflow. AWB's workflow orchestration platform for ML retraining pipelines, regulatory reporting jobs, and nightly data ingestion DAGs.

**AML**  
Anti-Money Laundering. Controls to detect, prevent, and report suspected money laundering. UK primary legislation: POCA 2002. AWB's AML system (MR-2026-061) reduced false positives by 80%.

**Approximate Nearest Neighbour (ANN)**  
A search algorithm that finds vectors approximately closest to a query vector. Used in ChromaDB with HNSW index for RAG retrieval. Trades perfect recall for query speed.

**AUC-ROC**  
Area Under the Receiver Operating Characteristic curve. Binary classification performance metric. PRA SS1/23 minimum for AWB credit PD models: ≥0.75. AWB production PD model: 0.834.

**Audit trail**  
A complete, tamper-evident record of all model inputs, outputs, decisions, and overrides. PRA SS1/23 and FCA COBS 9 require 7-year retention. AWB stores audit trails in PostgreSQL.

**Avon & Wessex Bank plc (AWB)**  
The fictional £40B UK bank used as the primary case study throughout this book. Bristol-headquartered, PRA/FCA regulated, Temenos T24 (2019). Entirely fictional.

**AWB-AI-2025**  
AWB's £3.2M AI transformation programme (January 2025). Target: 23 AI systems across credit risk, market risk, AML/KYC, compliance, MLOps, and data infrastructure.

**awb_commons**  
Shared Python library namespace across all AWB AI systems. Contains Pydantic schemas, LLM client factory, PostgreSQL base classes, and audit logger.


---

## B

**Basel IV**  
Informal name for the final Basel III reforms (BCBS, December 2017). Implemented in the EU/UK via CRR3 (January 2025). Key changes: output floor, revised standardised approaches, leverage ratio.

**BCBS 239**  
Basel Committee Principles for Effective Risk Data Aggregation and Risk Reporting (2013). 14 principles covering data governance, accuracy, completeness, timeliness, and adaptability.

**Bias**  
Systematic error in model predictions arising from flawed assumptions in the training process. PRA SS1/23 requires bias testing; EU AI Act Art. 10 mandates bias monitoring for HIGH-RISK systems.

**BM25**  
Best Match 25. Probabilistic term-frequency ranking function for information retrieval. Used in AWB's hybrid RAG alongside dense vector search, improving recall on exact-match regulatory queries by 23%.

**Bootstrapping**  
In statistics, resampling a dataset to estimate confidence intervals. In ML, a technique for initialising ensemble models. Also used in credit risk to generate synthetic default observations.


---

## C

**CBES**  
Climate Biennial Exploratory Scenario. Bank of England stress test for climate-related risks. UK equivalent of US DFAST climate scenarios. AWB Ch11 regulatory reporting references CBES.

**ChromaDB**  
Open-source vector database used by AWB for RAG semantic search. HNSW index with cosine similarity. Deployed in AWS eu-west-2 for UK data residency compliance.

**CI/CD**  
Continuous Integration / Continuous Deployment. AWB's PRA SS1/23-gated pipeline: code commit → unit tests → integration tests → SS1/23 validation gate → staging → production.
