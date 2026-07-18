# agent_evaluation.py | AWB Agent Evaluation Framework
# Chapter 3 | Section 3.9C | Non-deterministic agent evaluation
# Golden trajectories, LLM-as-judge, reliability SLOs, regression gating
# MR-2026-037 | EU AI Act Arts 9/14 | PRA SS1/23 paras 4 & 6
"""
AWB Agent Evaluation Framework — Section 3.9C

Section 3.9 verifies that an *individual* agent run behaves correctly.
Evaluation is a different discipline: it measures, statistically and
repeatedly, whether the agent behaves correctly across the *distribution*
of inputs it will meet in production. For a non-deterministic, tool-using
agent the unit of performance is the trajectory, not the prediction.

This module implements the three components described in Section 3.9C:

  3.9C.1  GoldenTrajectory / TrajectoryScorer
          recorded, human-validated execution paths scored on
          outcome match, trajectory adherence, tool-argument fidelity.
  3.9C.2  LLMJudge
          free-text artefact scoring with three guardrails
          (calibration, route-to-human, monthly spot-audit).
  3.9C.3  ReliabilitySLO / regression_gate
          SLO breach == model risk event (SS1/23 ongoing monitoring);
          a change is promoted only if the suite meets the baseline.

Regulatory basis:
  PRA SS1/23 para 4 — validation and deployment controls
  PRA SS1/23 para 6 — ongoing monitoring and performance tracking
  EU AI Act Art.14 — effective human oversight (judge never auto-approves)

The framework is stdlib-only and provider-agnostic; the LLM judge and the
agent runner are injected so the same harness runs in CI (replayed
fixtures) and against live production samples.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Protocol, Sequence


# ---------------------------------------------------------------------------
# 3.9C.1  Golden trajectories
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolCall:
    """A single step in an agent trajectory."""

    tool: str
    args: dict[str, object]


@dataclass(frozen=True)
class GoldenTrajectory:
    """A recorded, human-validated execution path for one scenario.

    Versioned alongside the prompts it exercises; ``prompt_version`` ties the
    expected behaviour to a specific prompt baseline so that a prompt change
    forces re-validation (see Section 5.8B.4 change management).
    """

    scenario_id: str
    description: str
    expected_tools: tuple[ToolCall, ...]
    expected_outcome: str
    prompt_version: str
    adversarial: bool = False


@dataclass(frozen=True)
class AgentRun:
    """The observed result of replaying one trajectory through the agent."""

    scenario_id: str
    tools_called: tuple[ToolCall, ...]
    outcome: str
    artefacts: dict[str, str] = field(default_factory=dict)  # name -> free text
    cost_gbp: float = 0.0
    latency_seconds: float = 0.0


class AgentRunner(Protocol):
    """Injected agent under test. The same protocol works for CI fixtures and
    live production replay."""

    def replay(self, trajectory: GoldenTrajectory) -> AgentRun: ...


def _ordered_match(expected: Sequence[ToolCall], actual: Sequence[ToolCall]) -> float:
    """Trajectory adherence allowing *benign reordering*: the fraction of
    expected tool calls present with materially correct arguments, regardless
    of order."""
    if not expected:
        return 1.0
    remaining = list(actual)
    hits = 0
    for exp in expected:
        for i, act in enumerate(remaining):
            if act.tool == exp.tool and _args_fidelity(exp.args, act.args) >= 0.99:
                hits += 1
                remaining.pop(i)
                break
    return hits / len(expected)


def _args_fidelity(expected: dict, actual: dict) -> float:
    """Tool-argument fidelity: fraction of expected keys whose values match."""
    if not expected:
        return 1.0
    ok = sum(1 for k, v in expected.items() if str(actual.get(k)) == str(v))
    return ok / len(expected)


@dataclass
class TrajectoryScore:
    scenario_id: str
    outcome_match: bool
    trajectory_adherence: float
    tool_argument_fidelity: float
    tool_selection_correct: bool


class TrajectoryScorer:
    """Scores a replayed run against its golden trajectory on the three
    dimensions of Section 3.9C.1."""

    def score(self, gold: GoldenTrajectory, run: AgentRun) -> TrajectoryScore:
        adherence = _ordered_match(gold.expected_tools, run.tools_called)
        fidelity = statistics.fmean(
            _args_fidelity(e.args, a.args)
            for e, a in zip(gold.expected_tools, run.tools_called)
        ) if gold.expected_tools and run.tools_called else (
            1.0 if not gold.expected_tools else 0.0
        )
        expected_names = [t.tool for t in gold.expected_tools]
        actual_names = [t.tool for t in run.tools_called]
        return TrajectoryScore(
            scenario_id=gold.scenario_id,
            outcome_match=(run.outcome == gold.expected_outcome),
            trajectory_adherence=adherence,
            tool_argument_fidelity=fidelity,
            tool_selection_correct=(set(expected_names) == set(actual_names)),
        )


# ---------------------------------------------------------------------------
# 3.9C.2  LLM-as-judge with guardrails
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JudgeVerdict:
    artefact: str
    faithfulness: float       # 0..1 grounded in the underlying data
    completeness: float       # 0..1 required sections present
    unsupported_claims: int   # count of claims not supported by source
    route_to_human: bool


# A judge function is injected (e.g. Claude Opus 4.8 — deliberately a
# different provider from the Gemini agents under test, to avoid
# self-preference bias). Signature: (artefact_text, source_context) -> dict
JudgeFn = Callable[[str, str], dict]


class LLMJudge:
    """Free-text scoring with the three guardrails of Section 3.9C.2.

    The judge is itself a fallible model, so it (1) is calibrated against a
    held-out set with known-correct gradings, (2) never auto-rejects — it
    routes below-threshold artefacts to human review (EU AI Act Art.14), and
    (3) exposes gradings for the monthly spot-audit.
    """

    def __init__(self, judge_fn: JudgeFn, threshold: float = 0.85):
        self._judge = judge_fn
        self.threshold = threshold
        self._audit_log: list[JudgeVerdict] = []
        self._calibration_error: float | None = None

    def calibrate(self, labelled: list[tuple[str, str, float]]) -> float:
        """labelled: (artefact, source, known_faithfulness). Returns mean
        absolute calibration error; a high value blocks use of the judge."""
        errors = []
        for artefact, source, known in labelled:
            v = self._judge(artefact, source)
            errors.append(abs(float(v.get("faithfulness", 0.0)) - known))
        self._calibration_error = statistics.fmean(errors) if errors else 0.0
        return self._calibration_error

    def grade(self, artefact_name: str, artefact: str, source: str) -> JudgeVerdict:
        raw = self._judge(artefact, source)
        faith = float(raw.get("faithfulness", 0.0))
        comp = float(raw.get("completeness", 0.0))
        unsupported = int(raw.get("unsupported_claims", 0))
        verdict = JudgeVerdict(
            artefact=artefact_name,
            faithfulness=faith,
            completeness=comp,
            unsupported_claims=unsupported,
            # never auto-reject: below-threshold => human review
            route_to_human=(faith < self.threshold or unsupported > 0),
        )
        self._audit_log.append(verdict)
        return verdict

    @property
    def audit_log(self) -> list[JudgeVerdict]:
        """Monthly spot-audit feed (guardrail 3)."""
        return list(self._audit_log)


# ---------------------------------------------------------------------------
# 3.9C.3  Reliability SLOs and regression gating
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReliabilitySLO:
    """Explicit thresholds. A breach is a model risk event under the SS1/23
    monitoring regime (Section 5.6 incident process), not a software bug."""

    min_outcome_match: float = 0.98
    min_tool_selection_accuracy: float = 0.995
    max_cost_variance: float = 0.20          # +/- vs baseline cost-per-run
    baseline_cost_gbp: float = 6.0           # midpoint of the GBP 4-8 baseline


@dataclass
class EvalReport:
    timestamp: str
    n: int
    outcome_match: float
    tool_selection_accuracy: float
    mean_trajectory_adherence: float
    mean_cost_gbp: float
    cost_variance: float
    judge_route_rate: float
    breaches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.breaches


class AgentEvaluator:
    """Runs the full suite and produces a gating report (Section 3.9C.3)."""

    def __init__(self, runner: AgentRunner, slo: ReliabilitySLO | None = None,
                 judge: LLMJudge | None = None):
        self.runner = runner
        self.slo = slo or ReliabilitySLO()
        self.judge = judge
        self.scorer = TrajectoryScorer()

    def evaluate(self, trajectories: list[GoldenTrajectory]) -> EvalReport:
        scores: list[TrajectoryScore] = []
        costs: list[float] = []
        judge_routes = 0
        judged = 0
        for gold in trajectories:
            run = self.runner.replay(gold)
            scores.append(self.scorer.score(gold, run))
            costs.append(run.cost_gbp)
            if self.judge:
                for name, text in run.artefacts.items():
                    judged += 1
                    if self.judge.grade(name, text, gold.description).route_to_human:
                        judge_routes += 1

        n = len(scores) or 1
        outcome = sum(s.outcome_match for s in scores) / n
        tool_acc = sum(s.tool_selection_correct for s in scores) / n
        adherence = statistics.fmean(s.trajectory_adherence for s in scores) if scores else 0.0
        mean_cost = statistics.fmean(costs) if costs else 0.0
        variance = (abs(mean_cost - self.slo.baseline_cost_gbp) / self.slo.baseline_cost_gbp
                    if self.slo.baseline_cost_gbp else 0.0)

        breaches: list[str] = []
        if outcome < self.slo.min_outcome_match:
            breaches.append(f"outcome_match {outcome:.3f} < {self.slo.min_outcome_match}")
        if tool_acc < self.slo.min_tool_selection_accuracy:
            breaches.append(
                f"tool_selection_accuracy {tool_acc:.3f} < {self.slo.min_tool_selection_accuracy}")
        if variance > self.slo.max_cost_variance:
            breaches.append(f"cost_variance {variance:.2f} > {self.slo.max_cost_variance}")

        return EvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n=len(scores),
            outcome_match=outcome,
            tool_selection_accuracy=tool_acc,
            mean_trajectory_adherence=adherence,
            mean_cost_gbp=mean_cost,
            cost_variance=variance,
            judge_route_rate=(judge_routes / judged) if judged else 0.0,
            breaches=breaches,
        )


def regression_gate(candidate: EvalReport, baseline: EvalReport) -> bool:
    """A prompt or model change is promoted only if the suite passes at or
    above the current baseline (the agentic equivalent of the SS1/23
    four-gate deployment control, Section 5.8A). The thresholds are a
    governance decision; this function only enforces them."""
    if not candidate.passed:
        return False
    return (
        candidate.outcome_match >= baseline.outcome_match
        and candidate.tool_selection_accuracy >= baseline.tool_selection_accuracy
        and candidate.cost_variance <= max(baseline.cost_variance, 0.20)
    )


__all__ = [
    "ToolCall", "GoldenTrajectory", "AgentRun", "AgentRunner",
    "TrajectoryScore", "TrajectoryScorer",
    "JudgeVerdict", "LLMJudge",
    "ReliabilitySLO", "EvalReport", "AgentEvaluator", "regression_gate",
]
