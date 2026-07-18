# Companion Documentation

This folder contains reference material from *AI for Financial Risk, Compliance and Regulatory Reporting: The Enterprise Implementation Guide* by Sree Kotha.

These documents were included as appendices in early drafts and are published here so they can be kept current without requiring a reprint.

---

## Appendices

| File | Description |
| --- | --- |
| [appendices/glossary.md](appendices/glossary.md) | **Appendix A — Glossary of Terms** — Definitions for key AI/ML, regulatory, and AWB-specific terms used across all 16 chapters |
| [appendices/regulatory-reference.md](appendices/regulatory-reference.md) | **Appendix B — UK/EU Regulatory Reference Guide** — Article-level summaries of PRA SS1/23, EU AI Act, DORA, CRR3/Basel IV, POCA/JMLSG, and FCA Consumer Duty |
| [appendices/tech-stack.md](appendices/tech-stack.md) | **Appendix C — Technology Stack Reference** — Approved LLM models, Python library versions, AWS services, and per-chapter developer cost estimates for the AWB reference implementation |

---

## How to use these files

**Readers of the book** — these files extend the printed appendices with content that can be updated as regulations and technology evolve. Check back for corrections and additions between print editions.

**Practitioners** — the [Regulatory Reference Guide](appendices/regulatory-reference.md) provides article-level mappings between PRA/FCA/EU AI Act requirements and specific AWB implementation choices. Use it alongside the chapter code when designing your own compliance controls.

**Developers** — the [Tech Stack Reference](appendices/tech-stack.md) lists exact library versions and AWS services used in production. The cost guide at the bottom of that file estimates monthly API spend per chapter group so you can plan your local development budget.

---

## Keeping these files current

Regulatory guidance changes frequently. If you spot an outdated reference or an article number that has been renumbered in a final text, please open an issue or pull request. The glossary and tech stack are similarly versioned — the current snapshot reflects AWB's production environment as of **June 2026**.
