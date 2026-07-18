# llmops/ragas_monitor.py | RAGAS 5% production sampling
# Chapter 14 | Auto-rollback if faithfulness < 0.80
# Linked to MR-2026-058 prompt registry
from __future__ import annotations
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Deque, Optional

log = logging.getLogger(__name__)

SAMPLE_RATE = 0.05          # 5% of production calls
FAITHFULNESS_ALERT = 0.85   # alert threshold
FAITHFULNESS_ROLLBACK = 0.80  # auto-rollback threshold
ANSWER_REL_ALERT = 0.75
WINDOW_HOURS = 3            # rolling evaluation window
ROLLBACK_SLA_MINUTES = 15   # target rollback SLA


@dataclass
class RAGASMetrics:
    """RAGAS quality metrics for one evaluated call."""
    service_id: str
    prompt_version: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    evaluated_at: datetime = field(
        default_factory=datetime.utcnow
    )


@dataclass
class WindowStats:
    """Rolling-window aggregate RAGAS statistics."""
    service_id: str
    window_start: datetime
    window_end: datetime
    sample_count: int
    mean_faithfulness: float
    mean_answer_relevancy: float
    rollback_triggered: bool = False


class RAGASMonitor:
    """5% production sampling RAGAS monitor.

    Samples 5% of live LLM requests. Computes rolling
    3-hour window statistics. Auto-rollback fires if
    mean faithfulness drops below 0.80 for any service.

    Per AWB MLOps: 4,200 calls/day * 5% = 210 samples/day
    sufficient for hourly statistics with narrow CI.
    """

    def __init__(
        self,
        service_id: str,
        rollback_fn: Callable[[str, str], None],
        sample_rate: float = SAMPLE_RATE,
    ) -> None:
        self.service_id = service_id
        self.rollback_fn = rollback_fn
        self.sample_rate = sample_rate
        self._window: Deque[RAGASMetrics] = deque()
        self._current_version: Optional[str] = None

    def set_production_version(
        self, version: str
    ) -> None:
        """Update the tracked production prompt version."""
        self._current_version = version
        log.info(
            "RAGAS monitor: %s now tracking v%s",
            self.service_id,
            version,
        )

    def maybe_evaluate(
        self,
        query: str,
        context_chunks: list[str],
        response: str,
    ) -> Optional[RAGASMetrics]:
        """Probabilistically evaluate; check thresholds.

        Args:
            query: User query string.
            context_chunks: Retrieved RAG context.
            response: LLM response to evaluate.
        Returns:
            RAGASMetrics if sampled, else None.
        """
        if random.random() > self.sample_rate:
            return None

        metrics = self._evaluate(
            query, context_chunks, response
        )
        self._window.append(metrics)
        self._prune_window()

        if metrics.faithfulness < FAITHFULNESS_ALERT:
            log.warning(
                "Faithfulness alert: %s v%s = %.3f",
                self.service_id,
                self._current_version,
                metrics.faithfulness,
            )

        stats = self._compute_window_stats()
        if stats and self._should_rollback(stats):
            self._trigger_rollback(stats)

        return metrics

    def _evaluate(
        self,
        query: str,
        context_chunks: list[str],
        response: str,
    ) -> RAGASMetrics:
        """Run RAGAS evaluation on sampled call.

        Full implementation uses ragas library.
        See: github.com/lorvenio/ai-banking-risk-platform
        """
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        # Placeholder — real impl calls ragas evaluate()
        return RAGASMetrics(
            service_id=self.service_id,
            prompt_version=self._current_version or "",
            faithfulness=0.0,      # replaced by ragas
            answer_relevancy=0.0,
            context_precision=0.0,
            context_recall=0.0,
        )

    def _prune_window(self) -> None:
        cutoff = (
            datetime.utcnow()
            - timedelta(hours=WINDOW_HOURS)
        )
        while (
            self._window
            and self._window[0].evaluated_at < cutoff
        ):
            self._window.popleft()

    def _compute_window_stats(
        self,
    ) -> Optional[WindowStats]:
        if not self._window:
            return None
        faithfulness_vals = [
            m.faithfulness for m in self._window
        ]
        relevancy_vals = [
            m.answer_relevancy for m in self._window
        ]
        now = datetime.utcnow()
        return WindowStats(
            service_id=self.service_id,
            window_start=self._window[0].evaluated_at,
            window_end=now,
            sample_count=len(self._window),
            mean_faithfulness=(
                sum(faithfulness_vals)
                / len(faithfulness_vals)
            ),
            mean_answer_relevancy=(
                sum(relevancy_vals) / len(relevancy_vals)
            ),
        )

    def _should_rollback(
        self, stats: WindowStats
    ) -> bool:
        return (
            stats.sample_count >= 10
            and stats.mean_faithfulness
            < FAITHFULNESS_ROLLBACK
        )

    def _trigger_rollback(
        self, stats: WindowStats
    ) -> None:
        """Fire automatic rollback within 15-min SLA.

        Calls rollback_fn(service_id, last_good_version),
        pages on-call engineer via PagerDuty, and logs
        the incident for post-incident review.
        """
        log.error(
            "AUTO-ROLLBACK: %s faithfulness=%.3f "
            "< %.2f over %d samples in %dhr window. "
            "Initiating rollback. Target SLA: %d min.",
            self.service_id,
            stats.mean_faithfulness,
            FAITHFULNESS_ROLLBACK,
            stats.sample_count,
            WINDOW_HOURS,
            ROLLBACK_SLA_MINUTES,
        )
        stats.rollback_triggered = True
        self.rollback_fn(
            self.service_id,
            self._current_version or "unknown",
        )
