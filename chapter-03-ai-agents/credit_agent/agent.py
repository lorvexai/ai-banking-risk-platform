"""
credit_agent/agent.py
AWB Automated Credit Decision Workflow — ReAct Agent Loop
Chapter 3: Agentic AI for Financial Risk

Implements a ReAct (Reason + Act) agent loop for automated credit assessment.

Architecture:
    Thought → Action → Observation → Thought (repeating cycle)
    Maximum 10 iterations to prevent infinite loops (operational resilience).

Regulatory context:
- PRA SS1/23 Section 3.4: AI agents are model outputs; MR-2026-037 registration.
- EU AI Act 2024 Article 14: Human-in-the-loop mandatory for credit decisions ≥ £500,000.
- DORA Article 6: Operational resilience — agent must not fail catastrophically
  on individual tool failures (graceful degradation).
- FCA PS22/9: Decision outcomes must be auditable and explainable.

LLM routing:
- Gemini 3.1 Pro: reasoning / tool selection steps (high-complexity)
- Gemini 3.5 Flash: memo drafting (high-throughput, cost-efficient)
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from credit_agent.tools import TOOL_REGISTRY, get_tool_schemas
from credit_agent.credit_memo_generator import (
    CreditMemo,
    build_credit_memo_from_agent_output,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("awb.credit_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 10           # Prevent infinite loops (operational resilience)
HITL_THRESHOLD_GBP = 500_000  # EU AI Act Article 14 human oversight threshold
MODEL_REASONING = "gemini-3.1-pro"     # High-complexity reasoning
MODEL_DRAFTING = "gemini-3.5-flash"    # Memo drafting (cost-efficient)
MODEL_REGISTRATION = "MR-2026-037"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class AgentStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    AWAITING_HUMAN_REVIEW = "AWAITING_HUMAN_REVIEW"
    FAILED = "FAILED"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"


@dataclass
class ToolCallLog:
    """
    Immutable log entry for a single tool call.
    Retained as part of the model audit trail (PRA SS1/23 Section 5.3).
    """
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    tool_name: str = ""
    tool_inputs: Dict[str, Any] = field(default_factory=dict)
    tool_outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    success: bool = True
    model_registration: str = MODEL_REGISTRATION

    def to_dict(self) -> dict:
        return {
            "log_id": self.log_id,
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "tool_inputs": self.tool_inputs,
            "tool_outputs": self.tool_outputs,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
            "model_registration": self.model_registration,
        }


@dataclass
class AgentStep:
    """One complete ReAct cycle: Thought → Action → Observation."""
    step_number: int
    thought: str
    action_tool: Optional[str]
    action_inputs: Optional[Dict[str, Any]]
    observation: Optional[str]
    tool_call_log: Optional[ToolCallLog]
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


@dataclass
class AgentRunResult:
    """
    Complete result of a single agent run.

    Contains: final credit memo, full step history, all tool call logs,
    and the complete audit trail required for PRA SS1/23 compliance.
    """
    run_id: str
    status: AgentStatus
    credit_memo: Optional[CreditMemo]
    steps: List[AgentStep]
    tool_call_logs: List[ToolCallLog]
    total_iterations: int
    total_latency_ms: float
    model_registration: str = MODEL_REGISTRATION
    completed_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    error_message: Optional[str] = None

    def get_audit_log(self) -> Dict[str, Any]:
        """
        Return the complete audit log entry for this agent run.
        Stored in AWB's audit database (7-year retention).
        """
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "total_iterations": self.total_iterations,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "tool_calls": [log.to_dict() for log in self.tool_call_logs],
            "model_registration": self.model_registration,
            "completed_at": self.completed_at,
            "memo_id": self.credit_memo.memo_id if self.credit_memo else None,
            "recommendation": (
                self.credit_memo.recommendation.value if self.credit_memo else None
            ),
            "human_review_required": (
                self.credit_memo.human_review_required if self.credit_memo else None
            ),
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# LLM client (abstraction over Gemini)
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Minimal LLM abstraction for Gemini 3.1 Pro/Flash.

    In production: replace the _call_gemini_api method with actual
    google-generativeai SDK calls. The mock implementation returns
    structured tool-call decisions for testing.

    DORA compliance: if Gemini is unavailable, the client raises
    LLMProviderError so the agent can apply graceful degradation.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key  # From environment in production

    def reason_and_select_tool(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        tool_schemas: List[Dict[str, Any]],
        remaining_tools: List[str],
    ) -> Dict[str, Any]:
        """
        Ask the LLM to reason about the next action and select a tool.

        In production this calls the Gemini 3.1 Pro API with function
        calling enabled. The mock implementation returns a deterministic
        sequence of tool calls for testing.

        Returns:
            dict with keys:
                thought (str): LLM reasoning
                tool_name (str or None): Tool to call
                tool_inputs (dict): Inputs for the tool
                is_final (bool): True if agent should terminate
        """
        # Production implementation (commented out for mock):
        # import google.generativeai as genai
        # genai.configure(api_key=self.api_key)
        # model = genai.GenerativeModel(self.model)
        # response = model.generate_content(
        #     [system_prompt] + [m["content"] for m in conversation_history],
        #     tools=tool_schemas,
        # )
        # return self._parse_gemini_response(response)

        # --- Mock implementation for testing ---
        # Returns the first tool not yet called
        if not remaining_tools:
            return {
                "thought": "All required tools have been called. Ready to finalise.",
                "tool_name": None,
                "tool_inputs": {},
                "is_final": True,
            }

        next_tool = remaining_tools[0]

        # Provide minimal valid inputs per tool
        mock_inputs = self._get_mock_inputs(next_tool)
        return {
            "thought": f"I need to call {next_tool} to gather credit assessment data.",
            "tool_name": next_tool,
            "tool_inputs": mock_inputs,
            "is_final": False,
        }

    def draft_memo_summary(
        self,
        findings: Dict[str, Any],
        application: Dict[str, Any],
    ) -> str:
        """
        Use Gemini 3.5 Flash to generate a plain-English rationale.
        Mock returns a template string in production tests.
        """
        applicant = application.get("applicant_name", "the applicant")
        amount = application.get("facility_amount_gbp", 0)
        recommendation = findings.get("check_credit_policy", {}).get("recommendation", "REFER")
        breaches = findings.get("check_credit_policy", {}).get("breach_count", 0)

        return (
            f"Following automated analysis of the credit application submitted by "
            f"{applicant} for a £{amount:,.0f} facility, the AWB Credit Decision "
            f"Workflow (Model MR-2026-037) recommends {recommendation}. "
            f"The assessment identified {breaches} policy consideration(s). "
            f"This recommendation is based on financial ratio analysis, AWB credit "
            f"policy evaluation, covenant assessment, and peer benchmarking. "
            f"A full rationale is provided in the attached credit memorandum."
        )

    def _get_mock_inputs(self, tool_name: str) -> Dict[str, Any]:
        """Return minimal valid inputs for each tool (used in mock mode)."""
        mock_map = {
            "fetch_t24_exposure": {"customer_id": "AWB-CUST-001234"},
            "calculate_ratios": {
                "revenue_gbp": 25_000_000.0,
                "ebitda_gbp": 4_500_000.0,
                "ebit_gbp": 3_800_000.0,
                "net_profit_gbp": 2_100_000.0,
                "total_assets_gbp": 32_000_000.0,
                "total_liabilities_gbp": 20_000_000.0,
                "current_assets_gbp": 8_000_000.0,
                "current_liabilities_gbp": 5_500_000.0,
                "net_debt_gbp": 18_000_000.0,
                "interest_expense_gbp": 950_000.0,
                "capital_expenditure_gbp": 600_000.0,
            },
            "check_credit_policy": {
                "net_debt_gbp": 18_000_000.0,
                "ebitda_gbp": 4_500_000.0,
                "ebit_gbp": 3_800_000.0,
                "interest_expense_gbp": 950_000.0,
                "total_exposure_gbp": 23_000_000.0,
                "tangible_equity_gbp": 12_000_000.0,
                "total_assets_gbp": 32_000_000.0,
                "facility_type": "TERM_LOAN",
            },
            "assess_covenants": {
                "facility_amount_gbp": 5_000_000.0,
                "facility_type": "TERM_LOAN",
                "leverage_ratio": 4.0,
                "interest_cover_ratio": 4.0,
            },
            "fetch_comparable_portfolio": {
                "industry_code": "4120",
                "facility_type": "TERM_LOAN",
                "facility_size_band": "MEDIUM",
            },
            "draft_credit_memo": {
                "applicant_name": "Fenland Construction Ltd",
                "facility_amount_gbp": 5_000_000.0,
                "facility_type": "TERM_LOAN",
                "policy_assessment": {},
                "financial_ratios": {},
                "covenant_assessment": {},
                "existing_exposure": {},
                "portfolio_benchmarks": {},
                "agent_recommendation": "REFER",
                "risk_rating": 6,
            },
        }
        return mock_map.get(tool_name, {})


# ---------------------------------------------------------------------------
# Human oversight checkpoint
# ---------------------------------------------------------------------------

class HumanOversightCheckpoint:
    """
    EU AI Act 2024 Article 14 — Human Oversight for Credit Decisions.

    For facilities of £500,000 or more, the agent MUST pause and await
    explicit approval from a Senior Credit Officer before finalising the
    recommendation.

    In production this integrates with AWB's workflow management system
    (ServiceNow) to create an approval task and await a callback.
    """

    EU_AI_ACT_THRESHOLD_GBP = HITL_THRESHOLD_GBP

    @staticmethod
    def is_required(facility_amount_gbp: float) -> bool:
        """Returns True if human review is mandatory under EU AI Act."""
        return facility_amount_gbp >= HumanOversightCheckpoint.EU_AI_ACT_THRESHOLD_GBP

    @staticmethod
    def request_review(
        memo: CreditMemo,
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Create a human review request.

        In production: POST to ServiceNow API to create approval task.
        Assigns to the duty Senior Credit Officer.

        Returns:
            dict with task_id, assignee, deadline, and instructions.
        """
        if not HumanOversightCheckpoint.is_required(memo.facility_amount_gbp):
            return {"review_required": False}

        deadline = datetime.datetime.utcnow() + datetime.timedelta(hours=24)

        logger.info(
            "EU AI ACT HITL: Human review requested for run_id=%s, memo_id=%s, "
            "facility=£%s, recommendation=%s",
            run_id,
            memo.memo_id,
            f"{memo.facility_amount_gbp:,.0f}",
            memo.recommendation.value,
        )

        return {
            "review_required": True,
            "task_id": f"SCO-{uuid.uuid4().hex[:8].upper()}",
            "memo_id": memo.memo_id,
            "run_id": run_id,
            "facility_amount_gbp": memo.facility_amount_gbp,
            "agent_recommendation": memo.recommendation.value,
            "assignee": "duty_senior_credit_officer@awb.co.uk",
            "deadline": deadline.isoformat(),
            "regulatory_basis": "EU AI Act 2024 Article 14",
            "instructions": (
                "Please review the attached credit memorandum and confirm or override "
                "the AI agent's recommendation. Your decision will be recorded in the "
                "PRA SS1/23 audit trail (MR-2026-037)."
            ),
            "task_url": f"https://awb.service-now.com/credit_review?task_id=SCO-{uuid.uuid4().hex[:8].upper()}",
        }

    @staticmethod
    def simulate_human_approval(memo: CreditMemo, reviewer_id: str = "SCO-001") -> CreditMemo:
        """
        Simulate human approval for testing purposes.

        In production this is called via a webhook when the Senior Credit
        Officer submits their decision in ServiceNow.
        """
        return memo.complete_human_review(reviewer_id=reviewer_id)


# ---------------------------------------------------------------------------
# Core agent class
# ---------------------------------------------------------------------------

class CreditDecisionAgent:
    """
    AWB Automated Credit Decision Workflow — ReAct Agent.

    Orchestrates multi-step credit assessment using a Thought → Action →
    Observation loop, with a mandatory human oversight checkpoint for
    material facilities.

    Usage:
        agent = CreditDecisionAgent()
        result = agent.run(credit_application)

    The agent calls tools in a logical sequence:
        1. fetch_t24_exposure      — understand existing indebtedness
        2. calculate_ratios        — compute financial ratios
        3. check_credit_policy     — evaluate against AWB policy rules
        4. assess_covenants        — recommend covenant structure
        5. fetch_comparable_portfolio — peer benchmarking
        6. draft_credit_memo       — synthesise findings into memo

    After tool execution, if facility ≥ £500,000 the agent pauses for
    human review before returning a final recommendation.
    """

    # Default tool execution order
    DEFAULT_TOOL_SEQUENCE = [
        "fetch_t24_exposure",
        "calculate_ratios",
        "check_credit_policy",
        "assess_covenants",
        "fetch_comparable_portfolio",
        "draft_credit_memo",
    ]

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_iterations: int = MAX_ITERATIONS,
        tool_registry: Optional[Dict] = None,
    ):
        self.llm_client = llm_client or LLMClient(model=MODEL_REASONING)
        self.max_iterations = max_iterations
        self.tool_registry = tool_registry or TOOL_REGISTRY
        self.tool_schemas = get_tool_schemas()

    def _build_system_prompt(self, application: Dict[str, Any]) -> str:
        """Build the system prompt for the LLM reasoning model."""
        return f"""You are AWB's Automated Credit Decision Agent (Model MR-2026-037).

Your task: Conduct a comprehensive credit assessment for the following application
and produce a structured recommendation for the Credit Committee.

Credit Application:
{json.dumps(application, indent=2)}

You must:
1. Call tools in logical order to gather all necessary information
2. Reason about each tool's output before proceeding to the next
3. Identify policy breaches and explain their significance
4. Recommend appropriate covenants based on risk profile
5. Draft a complete credit memorandum with APPROVE / REFER / DECLINE recommendation

Regulatory requirements:
- PRA SS1/23: All decisions must be logged with full traceability (MR-2026-037)
- EU AI Act 2024 Article 14: For facilities ≥ £{HITL_THRESHOLD_GBP:,}, human
  oversight is MANDATORY — do not finalise recommendation without human review
- FCA Consumer Duty PS22/9: Rationale must be explainable in plain English

Always complete all 6 tool calls before submitting your final recommendation.
"""

    def _execute_tool(
        self,
        tool_name: str,
        tool_inputs: Dict[str, Any],
        application: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], ToolCallLog]:
        """
        Execute a single tool with logging and error handling.

        Implements DORA Article 6 graceful degradation: if a tool fails,
        the agent continues with the remaining tools rather than aborting.

        Args:
            tool_name: Name of the tool to call.
            tool_inputs: Arguments to pass to the tool.
            application: Original credit application (for context injection).

        Returns:
            Tuple of (tool_output_or_None, ToolCallLog).
        """
        log = ToolCallLog(
            tool_name=tool_name,
            tool_inputs=tool_inputs,
        )
        start_time = time.perf_counter()

        try:
            if tool_name not in self.tool_registry:
                raise KeyError(f"Tool '{tool_name}' is not registered.")

            tool_fn = self.tool_registry[tool_name]

            # Inject application context for draft_credit_memo
            if tool_name == "draft_credit_memo" and not tool_inputs.get("applicant_name"):
                tool_inputs["applicant_name"] = application.get(
                    "applicant_name", "Unknown Applicant"
                )

            output = tool_fn(**tool_inputs)
            log.tool_outputs = output
            log.success = True

            logger.info(
                "Tool call SUCCESS | tool=%s | latency=%.1fms",
                tool_name,
                (time.perf_counter() - start_time) * 1000,
            )

        except Exception as exc:
            log.error = str(exc)
            log.success = False
            output = None

            logger.warning(
                "Tool call FAILURE | tool=%s | error=%s | "
                "Applying graceful degradation (DORA Article 6).",
                tool_name,
                str(exc),
            )

        finally:
            log.latency_ms = (time.perf_counter() - start_time) * 1000

        return output, log

    def run(
        self,
        credit_application: Dict[str, Any],
        auto_approve_human_review: bool = False,
        human_reviewer_id: str = "SCO-001",
    ) -> AgentRunResult:
        """
        Execute the full credit decision workflow.

        Args:
            credit_application: Dict containing applicant details and financials.
                Required keys: applicant_name, customer_id, facility_amount_gbp,
                facility_type, financials.
            auto_approve_human_review: If True, simulate immediate human approval
                (for testing only; never True in production).
            human_reviewer_id: Reviewer ID for simulated approval (testing only).

        Returns:
            AgentRunResult containing the final credit memo, step history,
            and complete audit trail.
        """
        run_id = f"RUN-{uuid.uuid4().hex[:12].upper()}"
        start_time = time.perf_counter()

        logger.info(
            "Agent run START | run_id=%s | applicant=%s | facility=£%s",
            run_id,
            credit_application.get("applicant_name", "Unknown"),
            f"{credit_application.get('facility_amount_gbp', 0):,.0f}",
        )

        steps: List[AgentStep] = []
        tool_call_logs: List[ToolCallLog] = []
        agent_findings: Dict[str, Any] = {}
        remaining_tools = list(self.DEFAULT_TOOL_SEQUENCE)
        conversation_history: List[Dict[str, str]] = []

        system_prompt = self._build_system_prompt(credit_application)
        status = AgentStatus.RUNNING
        credit_memo: Optional[CreditMemo] = None
        error_message: Optional[str] = None

        # --- ReAct loop ---
        for iteration in range(self.max_iterations):
            logger.info("Agent iteration %d/%d", iteration + 1, self.max_iterations)

            # Step 1: THOUGHT — ask LLM to reason and select next action
            try:
                llm_decision = self.llm_client.reason_and_select_tool(
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    tool_schemas=self.tool_schemas,
                    remaining_tools=remaining_tools,
                )
            except Exception as exc:
                logger.error("LLM reasoning failed: %s", exc)
                # Graceful degradation: use next tool in default sequence
                if remaining_tools:
                    llm_decision = {
                        "thought": f"LLM unavailable; using default sequence. Next: {remaining_tools[0]}",
                        "tool_name": remaining_tools[0],
                        "tool_inputs": self.llm_client._get_mock_inputs(remaining_tools[0]),
                        "is_final": False,
                    }
                else:
                    llm_decision = {"thought": "All tools complete.", "tool_name": None, "is_final": True}

            thought = llm_decision.get("thought", "")
            tool_name = llm_decision.get("tool_name")
            tool_inputs = llm_decision.get("tool_inputs", {})
            is_final = llm_decision.get("is_final", False)

            # Step 2: ACTION — execute tool if selected
            observation = None
            tool_log = None

            if tool_name and tool_name in remaining_tools:
                output, tool_log = self._execute_tool(tool_name, tool_inputs, credit_application)
                tool_call_logs.append(tool_log)

                if output is not None:
                    agent_findings[tool_name] = output
                    observation = f"Tool {tool_name} succeeded. Output keys: {list(output.keys())}"
                    remaining_tools.remove(tool_name)
                else:
                    # Tool failed — skip and continue (graceful degradation)
                    observation = (
                        f"Tool {tool_name} failed with error: {tool_log.error}. "
                        f"Continuing with remaining tools."
                    )
                    remaining_tools.remove(tool_name)  # Don't retry failed tools

                conversation_history.append({
                    "role": "assistant",
                    "content": thought,
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

            # Record step
            steps.append(AgentStep(
                step_number=iteration + 1,
                thought=thought,
                action_tool=tool_name,
                action_inputs=tool_inputs,
                observation=observation,
                tool_call_log=tool_log,
            ))

            # Check termination conditions
            if is_final or not remaining_tools:
                logger.info("Agent loop complete after %d iterations.", iteration + 1)
                break

        else:
            # Loop exhausted without break — max iterations reached
            logger.warning(
                "MAX ITERATIONS REACHED (%d) for run_id=%s. "
                "Proceeding with partial findings.",
                self.max_iterations,
                run_id,
            )
            status = AgentStatus.MAX_ITERATIONS_REACHED

        # --- Build credit memo from findings ---
        try:
            credit_memo = build_credit_memo_from_agent_output(
                agent_findings=agent_findings,
                credit_application=credit_application,
            )
            logger.info(
                "Credit memo built | memo_id=%s | recommendation=%s",
                credit_memo.memo_id,
                credit_memo.recommendation.value,
            )
        except Exception as exc:
            logger.error("Credit memo construction failed: %s", exc)
            error_message = str(exc)
            status = AgentStatus.FAILED

        # --- Human oversight checkpoint (EU AI Act Article 14) ---
        if credit_memo and HumanOversightCheckpoint.is_required(credit_application.get("facility_amount_gbp", 0)):
            logger.info(
                "EU AI ACT HITL: Facility ≥ £%s requires human review.",
                f"{HITL_THRESHOLD_GBP:,}",
            )
            review_request = HumanOversightCheckpoint.request_review(credit_memo, run_id)

            if auto_approve_human_review:
                # For testing only — simulate immediate approval
                credit_memo = HumanOversightCheckpoint.simulate_human_approval(
                    credit_memo, reviewer_id=human_reviewer_id
                )
                status = AgentStatus.COMPLETED
                logger.info("Simulated human approval by %s", human_reviewer_id)
            else:
                # Production: pause and wait for webhook callback
                status = AgentStatus.AWAITING_HUMAN_REVIEW
                logger.info(
                    "Agent PAUSED awaiting human review | task_id=%s | deadline=%s",
                    review_request.get("task_id"),
                    review_request.get("deadline"),
                )
        elif credit_memo:
            status = AgentStatus.COMPLETED

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        result = AgentRunResult(
            run_id=run_id,
            status=status,
            credit_memo=credit_memo,
            steps=steps,
            tool_call_logs=tool_call_logs,
            total_iterations=len(steps),
            total_latency_ms=total_latency_ms,
            error_message=error_message,
        )

        logger.info(
            "Agent run COMPLETE | run_id=%s | status=%s | iterations=%d | latency=%.0fms",
            run_id,
            status.value,
            len(steps),
            total_latency_ms,
        )

        return result

    def complete_human_review(
        self,
        result: AgentRunResult,
        reviewer_id: str,
    ) -> AgentRunResult:
        """
        Record completion of human review for a paused agent run.

        Called by the webhook handler when a Senior Credit Officer submits
        their decision in ServiceNow.

        Args:
            result: The AgentRunResult in AWAITING_HUMAN_REVIEW status.
            reviewer_id: Employee ID of the reviewing officer.

        Returns:
            Updated AgentRunResult with COMPLETED status.
        """
        if result.status != AgentStatus.AWAITING_HUMAN_REVIEW:
            raise ValueError(
                f"Cannot complete human review: run is in status {result.status.value}."
            )

        if result.credit_memo is None:
            raise ValueError("No credit memo found; cannot complete human review.")

        result.credit_memo = HumanOversightCheckpoint.simulate_human_approval(
            result.credit_memo, reviewer_id=reviewer_id
        )
        result.status = AgentStatus.COMPLETED

        logger.info(
            "Human review completed | run_id=%s | reviewer=%s | memo_id=%s",
            result.run_id,
            reviewer_id,
            result.credit_memo.memo_id,
        )

        return result
