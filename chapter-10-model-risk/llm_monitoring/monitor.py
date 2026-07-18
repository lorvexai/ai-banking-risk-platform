"""
llm_monitoring/monitor.py — AWB LLM Prompt Versioning and Monitoring.
PRA SS1/23: LLMs used in regulated functions require the same
governance as statistical models — version control, monitoring,
and periodic revalidation.
Avon & Wessex Bank plc (AWB) — AWB-AI-2025 programme.
"""
from __future__ import annotations
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from awb_commons.models import LLMMonitoringSnapshot

logger = logging.getLogger(__name__)

# PRA SS1/23: RAGAS monitoring thresholds (from model cards)
THRESHOLDS = {
    "faithfulness":      0.85,
    "answer_relevancy":  0.80,
    "context_precision": 0.75,
    "context_recall":    0.70,
}

# Alert triggers
HALLUCINATION_ALERT_PCT = 1.0   # >1% requires investigation
COST_ALERT_MULTIPLIER   = 1.30  # >30% cost increase vs baseline
LATENCY_SLA_MS          = 2000  # p95 must be <2000ms


@dataclass
class PromptVersion:
    """
    Versioned prompt template for PRA SS1/23 governance.
    Every prompt used in a regulated function is versioned,
    hashed, and stored in the prompt registry before use.
    """
    prompt_id: str
    mr_reference: str
    version: str
    system_prompt: str
    user_template: str
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    is_active: bool = False

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of system + user template."""
        content = self.system_prompt + self.user_template
        return hashlib.sha256(
            content.encode()
        ).hexdigest()[:16]


class PromptRegistry:
    """
    Version-controlled prompt registry for AWB LLM systems.

    Satisfies PRA SS1/23 requirement that changes to model
    inputs (including prompts) trigger a change management
    process equivalent to model redevelopment.

    Workflow:
    1. Developer creates new PromptVersion (draft)
    2. Model risk team reviews and approves
    3. Active prompt is updated in registry
    4. Previous version archived (never deleted)
    """

    def __init__(self) -> None:
        self._prompts: dict[str, list[PromptVersion]] = {}
        logger.info("PromptRegistry initialised")

    def register(self, prompt: PromptVersion) -> PromptVersion:
        """Register a new prompt version (initially inactive)."""
        mr = prompt.mr_reference
        if mr not in self._prompts:
            self._prompts[mr] = []
        existing_versions = [
            p.version for p in self._prompts[mr]
        ]
        if prompt.version in existing_versions:
            raise ValueError(
                f"Version {prompt.version} already exists "
                f"for {mr}"
            )
        self._prompts[mr].append(prompt)
        logger.info(
            "Prompt registered: %s v%s hash=%s",
            mr, prompt.version, prompt.content_hash,
        )
        return prompt

    def approve(
        self,
        mr_reference: str,
        version: str,
        approver_id: str,
    ) -> PromptVersion:
        """
        Approve a prompt version for production use.
        Deactivates the currently active version.
        """
        prompt = self._find(mr_reference, version)
        # Deactivate current active version
        for p in self._prompts.get(mr_reference, []):
            if p.is_active:
                p.is_active = False
                logger.info(
                    "Deactivated prompt: %s v%s",
                    mr_reference, p.version,
                )
        prompt.approved_by = approver_id
        prompt.approved_at = datetime.utcnow()
        prompt.is_active = True
        logger.info(
            "Prompt approved: %s v%s by %s",
            mr_reference, version, approver_id,
        )
        return prompt

    def active_prompt(
        self, mr_reference: str
    ) -> Optional[PromptVersion]:
        """Return the currently active prompt for a model."""
        for p in self._prompts.get(mr_reference, []):
            if p.is_active:
                return p
        return None

    def version_history(
        self, mr_reference: str
    ) -> list[PromptVersion]:
        """Return all prompt versions for a model."""
        return self._prompts.get(mr_reference, [])

    def _find(
        self, mr_reference: str, version: str
    ) -> PromptVersion:
        for p in self._prompts.get(mr_reference, []):
            if p.version == version:
                return p
        raise KeyError(
            f"Prompt {mr_reference} v{version} not found"
        )


class LLMMonitor:
    """
    Production monitoring for AWB LLM systems.

    Collects RAGAS metrics, cost, latency, and hallucination
    rate on a 5% sampling basis. Alerts are raised when
    metrics breach thresholds defined in PRA SS1/23 model cards.

    Monthly snapshots are stored for annual model review
    and PRA supervisory examination.
    """

    def __init__(
        self,
        mr_reference: str,
        baseline_cost_gbp: float = 0.0040,
    ) -> None:
        self.mr_reference = mr_reference
        self.baseline_cost_gbp = baseline_cost_gbp
        logger.info(
            "LLMMonitor initialised: %s", mr_reference
        )

    def assess_snapshot(
        self,
        snapshot: LLMMonitoringSnapshot,
    ) -> list[str]:
        """
        Evaluate a monitoring snapshot against thresholds.

        Args:
            snapshot: Monthly monitoring data.

        Returns:
            List of alert messages (empty if all clear).
        """
        alerts: list[str] = []
        ragas_map = {
            "faithfulness":      snapshot.faithfulness_score,
            "answer_relevancy":  snapshot.answer_relevancy,
            "context_precision": snapshot.context_precision,
            "context_recall":    snapshot.context_recall,
        }
        for metric, value in ragas_map.items():
            threshold = THRESHOLDS.get(metric, 0.0)
            if value < threshold:
                alerts.append(
                    f"RAGAS {metric} {value:.3f} below "
                    f"threshold {threshold}"
                )
        if snapshot.hallucination_rate_pct > HALLUCINATION_ALERT_PCT:
            alerts.append(
                f"Hallucination rate {snapshot.hallucination_rate_pct:.1f}% "
                f"exceeds threshold {HALLUCINATION_ALERT_PCT}%"
            )
        if snapshot.p95_latency_ms > LATENCY_SLA_MS:
            alerts.append(
                f"P95 latency {snapshot.p95_latency_ms}ms "
                f"exceeds SLA {LATENCY_SLA_MS}ms"
            )
        cost_ratio = (
            snapshot.avg_cost_per_query_gbp
            / max(self.baseline_cost_gbp, 1e-9)
        )
        if cost_ratio > COST_ALERT_MULTIPLIER:
            alerts.append(
                f"Cost per query £{snapshot.avg_cost_per_query_gbp:.4f} "
                f"is {cost_ratio:.0%} of baseline — unexpected increase"
            )
        if alerts:
            logger.warning(
                "LLM monitoring alerts for %s: %d issues",
                self.mr_reference, len(alerts),
            )
        else:
            logger.info(
                "LLM monitoring: %s all metrics within thresholds",
                self.mr_reference,
            )
        return alerts

    def trigger_revalidation(
        self,
        snapshot: LLMMonitoringSnapshot,
    ) -> bool:
        """
        Determine whether monitoring findings require formal
        revalidation per PRA SS1/23.
        Returns True if revalidation should be triggered.
        """
        alerts = self.assess_snapshot(snapshot)
        critical = [a for a in alerts if "below threshold" in a]
        return len(critical) >= 2
