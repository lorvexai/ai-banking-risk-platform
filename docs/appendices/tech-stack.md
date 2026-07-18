# Appendix C — Technology Stack Reference

> Complete technology stack used in the AWB (*Avon & Wessex Bank*) reference implementation across all 16 chapters.

This appendix documents the approved technology stack for the AWB-AI-2025 programme as of June 2026. All versions, costs, and configuration choices reflect AWB's production environment. Verify current pricing before procurement.
C.1  Approved LLM Models — June 2026
| Provider | Model | Context | Input £/1M | Output £/1M | AWB Use |
| --- | --- | --- | --- | --- | --- |
| Google | Gemini 3.5 Flash | 1M tokens | £1.18 | £7.09 | ✅ Production primary (68%) |
| Google | Gemini 3.1 Pro | 1M tokens | £1.57 | £9.45 | ✅ Complex reasoning (15%) |
| OpenAI | GPT-5.5 | 1M tokens | £3.94 | £23.62 | ✅ Alternative (10%) |
| OpenAI | GPT-5 Mini | 1M tokens | £0.20 | £1.57 | ✅ Budget option |
| Anthropic | Claude Opus 4.8 | 1M tokens | £3.94 | £19.69 | ✅ Highest reasoning (17%) |
| Anthropic | Claude Sonnet 4.6 | 1M tokens | £2.36 | £11.81 | ✅ Balanced |
| Meta | Llama 4 Maverick (self-hosted) | 1M tokens | Infrastructure | — | ❌ PRA SS1/23 burden |

USD costs converted at £1 = $1.27 (June 2026). Model availability and pricing change frequently; the maintained list lives in this repository, with current vendor pricing at anthropic.com, openai.com, and ai.google.dev. AWB's multi-LLM split satisfies DORA Art. 28 70% concentration cap.
C.2  Core Python Libraries
| Library | Version | Purpose | Chapter |
| --- | --- | --- | --- |
| Python | 3.11+ | Primary language | All |
| FastAPI | 0.110+ | API framework | All services |
| Pydantic | 2.6+ | Data validation & settings | All |
| LangGraph | 0.1+ | Agent state machines | Ch03 |
| ChromaDB | 0.4+ | Vector database | Ch04, Ch12 |
| RAGAS | 0.1+ | RAG evaluation | Ch04 |
| XGBoost | 2.0+ | Credit & fraud models | Ch06, Ch08 |
| SHAP | 0.44+ | Model explainability | Ch06, Ch07, Ch08 |
| MLflow | 2.11+ | Experiment tracking | Ch14 |
| DVC | 3.40+ | Data versioning | Ch14 |
| Airflow | 2.9+ | Pipeline orchestration | Ch14, Ch15 |
| rank-bm25 | 0.2+ | BM25 sparse retrieval | Ch04 |
| adjustText | 1.0+ | Chart label placement | All figures |
| pytest | 7.4+ | Testing framework | All |

C.3  Cloud Services — AWS eu-west-2 (UK)
| Service | AWB Use | Data Residency | PRA/DORA Notes |
| --- | --- | --- | --- |
| ECS Fargate | All 23 AI service containers | eu-west-2 (UK) | ICT asset registration required |
| RDS PostgreSQL | Audit logs, model outputs, COREP data | eu-west-2 (UK) | 7-year retention policy |
| ElastiCache Redis | Agent memory, API caching | eu-west-2 (UK) | 90-day TTL for working memory |
| S3 | Data lake raw/curated zones, model artefacts | eu-west-2 (UK) | 7-year retention for raw zone |
| MSK (Kafka) | Real-time fraud feature streaming | eu-west-2 (UK) | DORA ICT asset registration |
| CloudWatch | Metrics, alarms, log aggregation | eu-west-2 (UK) | DORA incident detection |

C.4  Developer Cost Guide by Chapter Group
| Chapter Group | API Cost/Month | Notes |
| --- | --- | --- |
| Chapters 1–5 (Foundations) | £0 | Free tier: Gemini 3.1 Pro / Flash free tier covers all examples |
| Chapters 6–10 (Risk domains) | £5–£20 | LLM calls + modest PostgreSQL/Redis for local dev |
| Chapters 11–12 (Compliance) | £5–£15 | AML/KYC API calls, sanctions data feed (test) |
| Chapters 13–15 (Enterprise) | £20–£50 | AWS ECS + RDS + ElastiCache for integration tests |
| Chapter 16 (Integrated) | £30–£60 | Full platform stack, all 23 services running |

All code examples can be run locally with docker-compose and Gemini 3.5 Flash free tier. AWS services are optional for local development; mocked equivalents are provided in the test suite.
