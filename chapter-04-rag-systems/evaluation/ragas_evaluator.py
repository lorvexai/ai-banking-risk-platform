"""
evaluation/ragas_evaluator.py
AWB RAGAS Evaluation Harness — PRA SS1/23 Model Validation
Chapter 4: Section 4.8 — RAG Evaluation with RAGAS

MR-2026-038 (AWB Regulatory Knowledge Assistant) is a registered model
under PRA SS1/23 (LOW risk rating). Validation uses RAGAS — Retrieval-
Augmented Generation Assessment — as the primary quantitative framework.

Validation thresholds (AWB, June 2026):
  faithfulness       >= 0.85   (hallucination guard — primary metric)
  answer_relevancy   >= 0.80   (answers the question asked)
  context_precision  >= 0.75   (retrieved chunks are relevant)
  context_recall     >= 0.70   (all relevant chunks retrieved)

Test set: 150 question-answer pairs
  50 credit questions  (CRR3 Art.153, IRB PD/LGD, covenant definitions)
  50 capital questions (RWA methodology, leverage ratio, FRTB)
  50 reporting questions (CoRep deadlines, EBA taxonomy fields)

Re-evaluation triggered by:
  - Corpus update > 10% new documents
  - Quarterly scheduled review
  - Faithfulness drop below 0.80 in production monitoring (5% sampling)

Regulatory context:
  PRA SS1/23: model output traceability + ongoing monitoring
  FCA PS22/9: RAG answers must be explainable and cite sources
  DORA Art. 9: ICT asset RKA-2026-001; validation evidence retained 7yr
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("awb.rag.evaluation")


# ── Validation thresholds ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RAGASThresholds:
    """
    AWB validation thresholds for MR-2026-038.

    All four metrics must be met for the model to pass PRA SS1/23
    periodic validation. A sustained faithfulness drop below 0.80
    warrants a formal model performance review.
    """
    faithfulness:      float = 0.85
    answer_relevancy:  float = 0.80
    context_precision: float = 0.75
    context_recall:    float = 0.70

    def validate(self, scores: "RAGASScores") -> bool:
        """Return True if all thresholds are met."""
        return (
            scores.faithfulness      >= self.faithfulness
            and scores.answer_relevancy  >= self.answer_relevancy
            and scores.context_precision >= self.context_precision
            and scores.context_recall    >= self.context_recall
        )


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class RAGASScores:
    """RAGAS metric scores for a validation run."""
    faithfulness:      float = 0.0
    answer_relevancy:  float = 0.0
    context_precision: float = 0.0
    context_recall:    float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }

    def worst_metric(self) -> str:
        """Return the metric name with the lowest score."""
        metrics = self.as_dict()
        return min(metrics, key=metrics.__getitem__)


@dataclass
class ValidationResult:
    """
    Full PRA SS1/23 validation result for MR-2026-038.

    Retained 7 years as primary validation evidence.
    """
    model_id:          str
    passed:            bool
    scores:            RAGASScores
    test_set_path:     str
    test_set_size:     int
    evaluated_at:      datetime = field(default_factory=datetime.utcnow)
    failure_reason:    Optional[str] = None
    by_category:       Dict[str, RAGASScores] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line summary for model card / audit log."""
        status = "PASS" if self.passed else "FAIL"
        s = self.scores
        return (
            f"{self.model_id} [{status}] "
            f"faithfulness={s.faithfulness:.3f} "
            f"relevancy={s.answer_relevancy:.3f} "
            f"precision={s.context_precision:.3f} "
            f"recall={s.context_recall:.3f} "
            f"({self.evaluated_at.strftime('%Y-%m-%d')})"
        )


@dataclass
class QAPair:
    """A single question-answer pair from the validation test set."""
    question:         str
    expected_answer:  str
    category:         str        # "credit" | "capital" | "reporting"
    question_id:      str = ""
    ground_truth_docs: List[str] = field(default_factory=list)


# ── Main evaluator class ──────────────────────────────────────────────────────

class AWBRagasEvaluator:
    """
    RAGAS evaluation harness for PRA SS1/23 model validation.

    Evaluates MR-2026-038 (Regulatory Knowledge Assistant) against
    AWB's 150-question test set and produces a structured
    ValidationResult for the model card.

    Usage (offline — no live API required for unit tests):
        evaluator = AWBRagasEvaluator(use_mock=True)
        result = evaluator.validate(
            model_id="MR-2026-038",
            test_set_path="evaluation/test_set.json",
            rag_engine=engine,
        )
        print(result.summary())
        # MR-2026-038 [PASS] faithfulness=0.91 relevancy=0.88 ...

    Production (5% sampling for ongoing monitoring):
        evaluator = AWBRagasEvaluator()
        sample = evaluator.sample_for_monitoring(
            test_set_path="evaluation/test_set.json",
            sample_rate=0.05,
        )
        result = evaluator.validate("MR-2026-038", sample, engine)
    """

    CATEGORIES = ("credit", "capital", "reporting")

    def __init__(
        self,
        thresholds: RAGASThresholds = RAGASThresholds(),
        use_mock: bool = False,
    ) -> None:
        """
        Args:
            thresholds: Validation thresholds. Defaults to AWB standard.
            use_mock: If True, use deterministic mock scores instead
                of calling the RAGAS library. For testing only.
        """
        self.thresholds = thresholds
        self.use_mock   = use_mock

    def validate(
        self,
        model_id: str,
        test_set_path: str,
        rag_engine=None,
    ) -> ValidationResult:
        """
        Run full RAGAS validation against the 150-question test set.

        Args:
            model_id: AWB model registry ID (e.g. "MR-2026-038").
            test_set_path: Path to JSON test set file.
            rag_engine: RAG engine to evaluate. Must implement
                .query(question: str) -> RegulatoryAnswer.
                If None and use_mock=False, raises RuntimeError.

        Returns:
            ValidationResult with pass/fail and per-category scores.
        """
        qa_pairs = self._load_test_set(test_set_path)
        if not qa_pairs:
            raise ValueError(
                f"Test set at {test_set_path} is empty or unreadable"
            )

        logger.info(
            "Starting RAGAS validation: model=%s, test_set=%s, n=%d",
            model_id, test_set_path, len(qa_pairs),
        )

        if self.use_mock:
            scores = self._mock_scores()
            by_cat = {
                c: self._mock_scores() for c in self.CATEGORIES
            }
        elif rag_engine is None:
            raise RuntimeError(
                "rag_engine is required when use_mock=False"
            )
        else:
            scores, by_cat = self._run_ragas(qa_pairs, rag_engine)

        passed = self.thresholds.validate(scores)
        failure_reason = None
        if not passed:
            worst = scores.worst_metric()
            actual = getattr(scores, worst)
            threshold = getattr(self.thresholds, worst)
            failure_reason = (
                f"{worst}={actual:.3f} below threshold {threshold:.3f}"
            )

        result = ValidationResult(
            model_id=model_id,
            passed=passed,
            scores=scores,
            test_set_path=test_set_path,
            test_set_size=len(qa_pairs),
            failure_reason=failure_reason,
            by_category=by_cat,
        )
        log_fn = logger.info if passed else logger.warning
        log_fn("Validation result: %s", result.summary())
        return result

    def sample_for_monitoring(
        self,
        test_set_path: str,
        sample_rate: float = 0.05,
    ) -> str:
        """
        Return path to a random 5% sample for production monitoring.

        AWB production: 5% of live queries are sampled weekly.
        Alert triggered if rolling faithfulness drops below 0.80.
        """
        import random, tempfile, os

        qa_pairs = self._load_test_set(test_set_path)
        k = max(1, round(len(qa_pairs) * sample_rate))
        sample = random.sample(qa_pairs, k)

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_sample.json", delete=False
        )
        json.dump(
            [
                {
                    "question":        q.question,
                    "expected_answer": q.expected_answer,
                    "category":        q.category,
                    "question_id":     q.question_id,
                }
                for q in sample
            ],
            tmp,
        )
        tmp.close()
        logger.info(
            "Generated monitoring sample: %d/%d questions -> %s",
            k, len(qa_pairs), tmp.name,
        )
        return tmp.name

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_test_set(self, path: str) -> List[QAPair]:
        """Load QA pairs from JSON file."""
        p = Path(path)
        if not p.exists():
            logger.warning("Test set not found: %s", path)
            return []
        with p.open() as f:
            raw = json.load(f)
        return [
            QAPair(
                question=item["question"],
                expected_answer=item["expected_answer"],
                category=item.get("category", "general"),
                question_id=item.get("question_id", ""),
                ground_truth_docs=item.get("ground_truth_docs", []),
            )
            for item in raw
        ]

    def _run_ragas(
        self,
        qa_pairs: List[QAPair],
        rag_engine,
    ) -> tuple[RAGASScores, Dict[str, RAGASScores]]:
        """
        Run RAGAS evaluation using the ragas library.

        Production implementation — requires:
          pip install ragas datasets
        and a configured LLM client for RAGAS's internal evaluation.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
        except ImportError as exc:
            raise ImportError(
                "RAGAS evaluation requires: "
                "pip install ragas datasets"
            ) from exc

        rows = []
        for qa in qa_pairs:
            answer_obj = rag_engine.query(qa.question)
            context = [c.excerpt for c in answer_obj.citations]
            rows.append({
                "question":  qa.question,
                "answer":    answer_obj.answer,
                "contexts":  context,
                "ground_truth": qa.expected_answer,
            })

        ds = Dataset.from_list(rows)
        result = evaluate(
            ds,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        all_scores = RAGASScores(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
        )

        # Per-category scores
        by_cat: Dict[str, RAGASScores] = {}
        for cat in self.CATEGORIES:
            cat_rows = [
                r for r, q in zip(rows, qa_pairs)
                if q.category == cat
            ]
            if not cat_rows:
                continue
            cat_ds = Dataset.from_list(cat_rows)
            cat_result = evaluate(
                cat_ds,
                metrics=[
                    faithfulness, answer_relevancy,
                    context_precision, context_recall,
                ],
            )
            by_cat[cat] = RAGASScores(
                faithfulness=cat_result["faithfulness"],
                answer_relevancy=cat_result["answer_relevancy"],
                context_precision=cat_result["context_precision"],
                context_recall=cat_result["context_recall"],
            )

        return all_scores, by_cat

    @staticmethod
    def _mock_scores() -> RAGASScores:
        """Mock scores for testing. All above AWB thresholds."""
        return RAGASScores(
            faithfulness=0.91,
            answer_relevancy=0.88,
            context_precision=0.82,
            context_recall=0.78,
        )
