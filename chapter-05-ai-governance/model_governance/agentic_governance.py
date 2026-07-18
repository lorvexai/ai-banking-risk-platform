"""
agentic_governance.py
=====================
Chapter 5 — AI Governance and Model Risk Management
Section 5.8B — Governing Agentic Systems

Avon & Wessex Bank plc (AWB) | AWB-AI-2025 Programme
Regulation: PRA SS1/23, SR 11-7, EU AI Act Art.14, DORA Arts 17 & 28

Sections 5.2-5.8A apply model risk management as PRA SS1/23 and SR 11-7
conceived it: to models that take an input and return a prediction. Most of
AWB's agentic systems do not fit that shape — they plan, call tools that take
real-world action, carry memory between steps, and behave non-deterministically.
This module extends governance to the four properties that make agents
different:

  5.8B.1  AutonomyLevel / OversightGate
          five-level autonomy scale; autonomy (not accuracy) sets the
          governance burden; human gate under EU AI Act Art.14.
  5.8B.2  TrajectoryTrace / ContinuousAssuranceMonitor
          point-in-time validation supplemented by trajectory-based
          validation and daily continuous assurance.
  5.8B.3  KillSwitchControl
          circuit breaker / kill-switch as a first-class governed control
          in the DORA Art.17 incident taxonomy.
  5.8B.4  AgentChangeManagement
          any change to the agent surface (prompt, tool, budget, provider)
          is a model change and re-enters the regression gate (Section 3.9C).

Stdlib-only; the evaluation harness and incident manager are injected so this
sits on top of agent_evaluation.py (3.9C) and incident_management.py (5.6).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, Enum
from typing import Callable, Optional


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 5.8B.1  Autonomy levels and risk appetite
# ---------------------------------------------------------------------------
class AutonomyLevel(IntEnum):
    """Five-level autonomy scale. Autonomy, not model accuracy, sets the
    governance burden. Raising a system's level is a board-reserved decision
    recorded in the model inventory, not an engineering configuration change."""

    L0_DRAFT_ONLY = 0       # agent drafts; a human performs every action
    L1_SUGGEST = 1          # agent proposes actions; human approves each
    L2_ACT_WITH_GATES = 2   # agent acts, but defined decisions are gated
    L3_ACT_WITH_AUDIT = 3   # agent acts; human reviews after the fact
    L4_AUTONOMOUS = 4        # agent acts without review within set limits


class Disposition(Enum):
    AUTO = "auto"
    HUMAN = "human"


@dataclass(frozen=True)
class Decision:
    """A material decision proposed by an agent (e.g. a credit recommendation)."""

    value_gbp: float
    confidence: float
    description: str = ""


@dataclass(frozen=True)
class OversightPolicy:
    """EU AI Act Art.14 effective-oversight thresholds for an L2 system.

    For the Credit Decision Agent: every decision above GBP 5M or below the
    confidence floor is gated to a human; the Treasury Operations Agent's
    market-order tools are capped at L1 (every action gated)."""

    value_gate_gbp: float = 5_000_000.0
    confidence_floor: float = 0.80


class OversightGate:
    """Routes a decision to a human or to automatic execution, given the
    system's autonomy level and oversight policy (Section 5.8B.1)."""

    def __init__(self, level: AutonomyLevel, policy: OversightPolicy | None = None):
        self.level = level
        self.policy = policy or OversightPolicy()

    def disposition(self, decision: Decision) -> Disposition:
        if self.level <= AutonomyLevel.L1_SUGGEST:
            return Disposition.HUMAN
        if self.level >= AutonomyLevel.L4_AUTONOMOUS:
            return Disposition.AUTO
        # L2 / L3: gate on value and confidence
        if (decision.value_gbp > self.policy.value_gate_gbp
                or decision.confidence < self.policy.confidence_floor):
            return Disposition.HUMAN
        return Disposition.AUTO


# ---------------------------------------------------------------------------
# 5.8B.2  Trajectory-based validation and continuous assurance
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryTrace:
    """A signed, replayable record of one multi-step run. The signature makes
    the trajectory tamper-evident for audit; provenance tags on each step let
    the DecisionAgent's inputs be validated against source documents before
    they are consumed (the control added after the Section 5.8B war story)."""

    run_id: str
    system_id: str
    steps: list[dict] = field(default_factory=list)
    created: str = field(default_factory=_utc)
    signature: str = ""

    def add_step(self, agent: str, action: str, output_ref: str,
                 source_validated: bool) -> None:
        self.steps.append({
            "agent": agent, "action": action,
            "output_ref": output_ref, "source_validated": source_validated,
        })

    def sign(self, secret: str) -> str:
        payload = json.dumps(
            {"run_id": self.run_id, "system_id": self.system_id,
             "steps": self.steps, "created": self.created},
            sort_keys=True,
        )
        self.signature = hashlib.sha256((secret + payload).encode()).hexdigest()
        return self.signature

    @property
    def unvalidated_handoffs(self) -> int:
        """Steps whose output was consumed downstream without source
        validation — the failure mode behind the 5.8B war story."""
        return sum(1 for s in self.steps if not s["source_validated"])


class ContinuousAssuranceMonitor:
    """Agents drift with every prompt/tool/model change, none of which appear
    in a traditional model-version record. AWB therefore treats agents as
    continuously assured rather than periodically revalidated: a daily
    production sample is re-scored, and a sustained SLO breach raises the same
    incident as a drift breach in a predictive model (Section 5.6)."""

    def __init__(self, evaluate_sample: Callable[[], bool],
                 raise_incident: Callable[[str, str], None],
                 breach_streak_to_escalate: int = 2):
        self._evaluate_sample = evaluate_sample      # returns True if within SLO
        self._raise_incident = raise_incident        # (system_id, detail)
        self._threshold = breach_streak_to_escalate
        self._streak = 0

    def daily_check(self, system_id: str) -> bool:
        within_slo = self._evaluate_sample()
        if within_slo:
            self._streak = 0
            return True
        self._streak += 1
        if self._streak >= self._threshold:
            self._raise_incident(
                system_id,
                f"Continuous-assurance SLO breach on {self._streak} consecutive days",
            )
        return False


# ---------------------------------------------------------------------------
# 5.8B.3  Kill-switches and circuit breakers as governed controls
# ---------------------------------------------------------------------------
@dataclass
class KillSwitchControl:
    """The kill-switch is a first-class governed control, not an
    implementation detail. Each has a named owner, a documented trigger and
    reset procedure, a tested runbook, and a place in the DORA Art.17 incident
    taxonomy. The token/cost budget (Section 3.8) is the routine expression of
    the same principle; the kill-switch is its escalation."""

    system_id: str
    owner: str
    runbook_url: str
    dora_incident_class: str = "ICT-OPERATIONAL"
    engaged: bool = False
    last_tested: Optional[str] = None
    history: list[dict] = field(default_factory=list)

    def engage(self, actor: str, reason: str) -> None:
        self.engaged = True
        self.history.append({"ts": _utc(), "actor": actor,
                             "action": "ENGAGE", "reason": reason})

    def reset(self, actor: str, approval_ref: str) -> None:
        # reset requires an approval reference — change-managed (5.8B.4)
        self.engaged = False
        self.history.append({"ts": _utc(), "actor": actor,
                             "action": "RESET", "approval_ref": approval_ref})

    def record_test(self) -> None:
        self.last_tested = _utc()


# ---------------------------------------------------------------------------
# 5.8B.4  Change management for non-deterministic systems
# ---------------------------------------------------------------------------
class ChangeSurface(Enum):
    PROMPT = "prompt"
    TOOL = "tool"
    BUDGET = "budget"
    MODEL_PROVIDER = "model_provider"
    OTHER = "other"


@dataclass(frozen=True)
class ChangeRequest:
    system_id: str
    surface: ChangeSurface
    description: str


@dataclass(frozen=True)
class ChangeDecision:
    requires_security_review: bool
    requires_regression_gate: bool
    requires_board_signoff: bool
    rationale: str


class AgentChangeManagement:
    """If a change can alter what the agent does, it is a model change and is
    governed as one. Every change re-enters the regression gate of Section
    3.9C before promotion."""

    def classify(self, req: ChangeRequest, raises_autonomy: bool = False) -> ChangeDecision:
        sec = req.surface in (ChangeSurface.TOOL,)        # threat model (3.6)
        board = raises_autonomy                            # board-reserved (5.8B.1)
        # every surface change re-enters the regression gate
        return ChangeDecision(
            requires_security_review=sec,
            requires_regression_gate=True,
            requires_board_signoff=board,
            rationale=(
                f"{req.surface.value} change to {req.system_id}: "
                f"regression gate always required; "
                f"security review={'yes' if sec else 'no'}; "
                f"board sign-off={'yes' if board else 'no'}."
            ),
        )


# ---------------------------------------------------------------------------
# Registry — ties the controls to each agentic system in the model inventory
# ---------------------------------------------------------------------------
@dataclass
class AgenticSystemGovernance:
    """The governance record held against each agentic system in the AWB model
    inventory (Section 5.8A)."""

    system_id: str            # e.g. MR-2026-037
    name: str
    autonomy: AutonomyLevel
    oversight: OversightGate
    kill_switch: KillSwitchControl
    golden_trajectory_pass_rate: float = 0.0   # gating metric (3.9C)

    def to_inventory_row(self) -> dict:
        return {
            "system_id": self.system_id,
            "name": self.name,
            "autonomy_level": self.autonomy.name,
            "value_gate_gbp": self.oversight.policy.value_gate_gbp,
            "kill_switch_owner": self.kill_switch.owner,
            "kill_switch_last_tested": self.kill_switch.last_tested,
            "golden_trajectory_pass_rate": self.golden_trajectory_pass_rate,
        }


__all__ = [
    "AutonomyLevel", "Disposition", "Decision", "OversightPolicy", "OversightGate",
    "TrajectoryTrace", "ContinuousAssuranceMonitor",
    "KillSwitchControl",
    "ChangeSurface", "ChangeRequest", "ChangeDecision", "AgentChangeManagement",
    "AgenticSystemGovernance",
]
