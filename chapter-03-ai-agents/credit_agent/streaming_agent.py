"""
credit_agent/streaming_agent.py
AWB Credit Agent — Async Streaming ReAct Loop
Chapter 3: Agentic AI for Financial Risk

Implements a streaming version of the credit decision agent that yields
real-time events as the agent reasons and acts. Enables front-end
applications to show live agent reasoning — each thought, tool call,
and observation streams to the user as it happens rather than appearing
only when the full run completes.

This pattern is essential for two AWB use cases:
  1. Credit Officer Dashboard: Shows the agent's live reasoning chain
     so the reviewing officer understands WHY each tool was called.
  2. Treasury Operations: Streaming report delivery (see treasury_agent.py)
     must begin before all data is collected, to meet the 07:00 window.

Architecture contrast with agent.py:
  agent.py            — synchronous, returns AgentRunResult on completion
  streaming_agent.py  — async generator, yields AgentEvent objects per step

Both use the same tools (tools.py) and policy rules (policy_rules.py).
The streaming version adds a real-time event layer over the same loop.

Event taxonomy:
  THOUGHT      — LLM reasoning step (shown to credit officer)
  TOOL_CALL    — Tool being executed (name + inputs)
  OBSERVATION  — Tool result summary (not full output — privacy)
  MEMO_CHUNK   — Token-by-token streaming of the credit memo narrative
  HITL_PAUSE   — Pipeline paused for human review (EU AI Act Art. 14)
  COMPLETE     — Run finished; full AgentRunResult attached
  ERROR        — Recoverable error; run continues (DORA Art. 6)

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 3 — Agentic AI for Financial Risk
Version: 1.0.0 (June 2026)
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from credit_agent.agent import (
    AgentRunResult,
    AgentStatus,
    AgentStep,
    HumanOversightCheckpoint,
    LLMClient,
    MAX_ITERATIONS,
    MODEL_DRAFTING,
    MODEL_REASONING,
    MODEL_REGISTRATION,
    HITL_THRESHOLD_GBP,
    ToolCallLog,
)
from credit_agent.tools import TOOL_REGISTRY, get_tool_schemas
from credit_agent.credit_memo_generator import build_credit_memo_from_agent_output

logger = logging.getLogger("awb.streaming_agent")


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class AgentEventType(str, Enum):
    THOUGHT     = "THOUGHT"
    TOOL_CALL   = "TOOL_CALL"
    OBSERVATION = "OBSERVATION"
    MEMO_CHUNK  = "MEMO_CHUNK"
    HITL_PAUSE  = "HITL_PAUSE"
    COMPLETE    = "COMPLETE"
    ERROR       = "ERROR"


@dataclass
class AgentEvent:
    """
    A single streaming event from the credit agent.

    Clients subscribe to the async generator and receive these events
    in real time, enabling live display of the agent's reasoning.

    Fields:
        event_type:    One of AgentEventType values.
        run_id:        Unique run identifier for correlation.
        iteration:     ReAct loop iteration number (1-indexed).
        content:       Human-readable event text.
        data:          Structured data (tool name, inputs, etc.).
        timestamp:     ISO-8601 UTC timestamp.
        is_final:      True for COMPLETE and HITL_PAUSE events.
    """
    event_type: AgentEventType
    run_id: str
    iteration: int
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    is_final: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    def to_json(self) -> str:
        """JSON-encode for WebSocket / SSE transport."""
        return json.dumps(self.to_dict(), default=str)

    def to_sse(self) -> str:
        """Format as Server-Sent Events message."""
        return f"event: {self.event_type.value}\ndata: {self.to_json()}\n\n"


# ---------------------------------------------------------------------------
# Streaming LLM client
# ---------------------------------------------------------------------------

class StreamingLLMClient:
    """
    LLM client that streams token-by-token for memo generation.

    For reasoning steps (THOUGHT), the full response is returned at once
    (Gemini Pro function calling does not support streaming mid-call).
    For memo narrative generation (MEMO_CHUNK), streaming is used so the
    credit officer sees the memo being written in real time.

    In production, the streaming memo call uses:
        model = genai.GenerativeModel("gemini-3.5-flash")
        response = model.generate_content(prompt, stream=True)
        async for chunk in response:
            yield AgentEvent(MEMO_CHUNK, chunk.text)
    """

    def __init__(self, reasoning_model: str = MODEL_REASONING):
        self.reasoning_model = reasoning_model
        self._sync_client = LLMClient(model=reasoning_model)

    def reason_and_select_tool(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        tool_schemas: List[Dict[str, Any]],
        remaining_tools: List[str],
    ) -> Dict[str, Any]:
        """Non-streaming reasoning step (delegates to sync LLMClient)."""
        return self._sync_client.reason_and_select_tool(
            system_prompt, conversation_history, tool_schemas, remaining_tools
        )

    async def stream_memo_narrative(
        self,
        findings: Dict[str, Any],
        application: Dict[str, Any],
    ) -> AsyncIterator[str]:
        """
        Async generator that streams the memo narrative token-by-token.

        In production:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(MODEL_DRAFTING)
            response = model.generate_content(
                build_memo_prompt(findings, application),
                stream=True,
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        Mock: yields the narrative in word-sized chunks with a small delay.
        """
        full_narrative = self._sync_client.draft_memo_summary(findings, application)
        words = full_narrative.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield chunk
            await asyncio.sleep(0.005)   # Simulate inter-token latency ~5ms


# ---------------------------------------------------------------------------
# Streaming agent
# ---------------------------------------------------------------------------

class StreamingCreditAgent:
    """
    Async streaming version of the AWB credit decision agent.

    Yields AgentEvent objects as it reasons through the ReAct loop,
    allowing callers to display live progress in the credit officer
    dashboard or pipe events to a WebSocket / SSE stream.

    Usage:
        agent = StreamingCreditAgent()
        async for event in agent.stream(credit_application):
            print(event.to_json())
            if event.is_final:
                break

    The generator terminates on:
      - COMPLETE event: all tools called, memo drafted, run finished.
      - HITL_PAUSE event: facility ≥ £500k, awaiting human review.
      - ERROR event with is_final=True: unrecoverable failure.
    """

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
        llm_client: Optional[StreamingLLMClient] = None,
        max_iterations: int = MAX_ITERATIONS,
    ):
        self.llm_client = llm_client or StreamingLLMClient()
        self.max_iterations = max_iterations
        self.tool_registry = TOOL_REGISTRY
        self.tool_schemas = get_tool_schemas()

    def _build_system_prompt(self, application: Dict[str, Any]) -> str:
        return (
            f"You are AWB's Automated Credit Decision Agent (Model {MODEL_REGISTRATION}).\n"
            f"Assess the credit application for {application.get('applicant_name', 'Unknown')} "
            f"and produce a structured recommendation.\n"
            f"Facility: £{application.get('facility_amount_gbp', 0):,.0f} "
            f"({application.get('facility_type', 'TERM_LOAN')}).\n"
            f"Call all 6 tools in sequence. Show your reasoning at each step."
        )

    async def stream(
        self,
        credit_application: Dict[str, Any],
        include_memo_streaming: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        """
        Async generator yielding real-time agent events.

        Args:
            credit_application: Dict with applicant details and financials.
            include_memo_streaming: If True, stream the memo narrative
                token-by-token (MEMO_CHUNK events). If False, return
                the full memo as a single COMPLETE event.

        Yields:
            AgentEvent objects in chronological order.
        """
        run_id = f"STR-{uuid.uuid4().hex[:12].upper()}"
        start_time = time.perf_counter()

        steps: List[AgentStep] = []
        tool_call_logs: List[ToolCallLog] = []
        agent_findings: Dict[str, Any] = {}
        remaining_tools = list(self.DEFAULT_TOOL_SEQUENCE)
        conversation_history: List[Dict[str, str]] = []
        system_prompt = self._build_system_prompt(credit_application)

        logger.info(
            "StreamingCreditAgent: START | run_id=%s | applicant=%s",
            run_id,
            credit_application.get("applicant_name"),
        )

        # ── ReAct streaming loop ───────────────────────────────────────────
        for iteration in range(self.max_iterations):

            # Step 1: THOUGHT — LLM reasons and selects next tool
            try:
                llm_decision = self.llm_client.reason_and_select_tool(
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                    tool_schemas=self.tool_schemas,
                    remaining_tools=remaining_tools,
                )
            except Exception as exc:
                # DORA graceful degradation: use default sequence on LLM failure
                llm_decision = {
                    "thought": f"LLM unavailable; applying default sequence. Error: {exc}",
                    "tool_name": remaining_tools[0] if remaining_tools else None,
                    "tool_inputs": {},
                    "is_final": not remaining_tools,
                }
                yield AgentEvent(
                    event_type=AgentEventType.ERROR,
                    run_id=run_id,
                    iteration=iteration + 1,
                    content=f"LLM reasoning failed; using fallback sequence. Error: {exc}",
                    data={"error": str(exc), "recovery": "default_sequence"},
                )

            thought = llm_decision.get("thought", "")
            tool_name = llm_decision.get("tool_name")
            tool_inputs = llm_decision.get("tool_inputs", {})
            is_final = llm_decision.get("is_final", False)

            # Yield THOUGHT event
            yield AgentEvent(
                event_type=AgentEventType.THOUGHT,
                run_id=run_id,
                iteration=iteration + 1,
                content=thought,
                data={"remaining_tools": remaining_tools},
            )

            # Small yield point so event is flushed before tool execution
            await asyncio.sleep(0)

            if is_final or not remaining_tools:
                break

            # Step 2: ACTION — execute tool and yield TOOL_CALL + OBSERVATION
            if tool_name and tool_name in remaining_tools:
                # Yield TOOL_CALL event
                yield AgentEvent(
                    event_type=AgentEventType.TOOL_CALL,
                    run_id=run_id,
                    iteration=iteration + 1,
                    content=f"Calling {tool_name}",
                    data={
                        "tool_name": tool_name,
                        "tool_inputs_summary": {
                            k: str(v)[:80] for k, v in tool_inputs.items()
                        },
                    },
                )
                await asyncio.sleep(0)

                # Execute tool (wrapped in async to not block event loop)
                log = ToolCallLog(tool_name=tool_name, tool_inputs=tool_inputs)
                t0 = time.perf_counter()

                try:
                    tool_fn = self.tool_registry[tool_name]
                    if tool_name == "draft_credit_memo" and not tool_inputs.get("applicant_name"):
                        tool_inputs["applicant_name"] = credit_application.get(
                            "applicant_name", "Unknown"
                        )
                    # Run synchronous tool in executor to avoid blocking
                    output = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool_fn(**tool_inputs)
                    )
                    log.tool_outputs = output
                    log.success = True
                    agent_findings[tool_name] = output
                    remaining_tools.remove(tool_name)

                    observation = (
                        f"{tool_name} succeeded. "
                        f"Output keys: {list(output.keys())[:5]}"
                    )

                except Exception as exc:
                    log.error = str(exc)
                    log.success = False
                    output = None
                    observation = f"{tool_name} failed: {exc}. Continuing."
                    remaining_tools.remove(tool_name)

                    yield AgentEvent(
                        event_type=AgentEventType.ERROR,
                        run_id=run_id,
                        iteration=iteration + 1,
                        content=f"Tool {tool_name} failed; applying DORA graceful degradation.",
                        data={"tool_name": tool_name, "error": str(exc)},
                    )

                finally:
                    log.latency_ms = (time.perf_counter() - t0) * 1000

                tool_call_logs.append(log)

                # Yield OBSERVATION event
                yield AgentEvent(
                    event_type=AgentEventType.OBSERVATION,
                    run_id=run_id,
                    iteration=iteration + 1,
                    content=observation,
                    data={
                        "tool_name": tool_name,
                        "success": log.success,
                        "latency_ms": round(log.latency_ms, 1),
                    },
                )

                conversation_history.extend([
                    {"role": "assistant", "content": thought},
                    {"role": "user", "content": f"Observation: {observation}"},
                ])

            steps.append(AgentStep(
                step_number=iteration + 1,
                thought=thought,
                action_tool=tool_name,
                action_inputs=tool_inputs,
                observation=observation if tool_name else None,
                tool_call_log=log if tool_name else None,
            ))

            await asyncio.sleep(0)  # yield control between iterations

        # ── Build credit memo ──────────────────────────────────────────────
        try:
            credit_memo = build_credit_memo_from_agent_output(
                agent_findings=agent_findings,
                credit_application=credit_application,
            )
        except Exception as exc:
            yield AgentEvent(
                event_type=AgentEventType.ERROR,
                run_id=run_id,
                iteration=0,
                content=f"Credit memo construction failed: {exc}",
                is_final=True,
            )
            return

        # ── Stream memo narrative (MEMO_CHUNK events) ─────────────────────
        if include_memo_streaming:
            async for chunk in self.llm_client.stream_memo_narrative(
                findings=agent_findings,
                application=credit_application,
            ):
                yield AgentEvent(
                    event_type=AgentEventType.MEMO_CHUNK,
                    run_id=run_id,
                    iteration=0,
                    content=chunk,
                    data={"memo_id": credit_memo.memo_id},
                )

        # ── Human oversight check ──────────────────────────────────────────
        facility_amount = credit_application.get("facility_amount_gbp", 0)
        if HumanOversightCheckpoint.is_required(facility_amount):
            review_request = HumanOversightCheckpoint.request_review(credit_memo, run_id)
            yield AgentEvent(
                event_type=AgentEventType.HITL_PAUSE,
                run_id=run_id,
                iteration=0,
                content=(
                    f"Facility £{facility_amount:,.0f} requires human review "
                    f"(EU AI Act Article 14). Task: {review_request.get('task_id')}"
                ),
                data=review_request,
                is_final=True,
            )
            return

        # ── Final COMPLETE event ───────────────────────────────────────────
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        result = AgentRunResult(
            run_id=run_id,
            status=AgentStatus.COMPLETED,
            credit_memo=credit_memo,
            steps=steps,
            tool_call_logs=tool_call_logs,
            total_iterations=len(steps),
            total_latency_ms=total_latency_ms,
        )

        yield AgentEvent(
            event_type=AgentEventType.COMPLETE,
            run_id=run_id,
            iteration=len(steps),
            content=(
                f"Credit assessment complete. "
                f"Recommendation: {credit_memo.recommendation.value}. "
                f"Memo: {credit_memo.memo_id}. "
                f"Latency: {total_latency_ms:.0f}ms."
            ),
            data=result.get_audit_log(),
            is_final=True,
        )

        logger.info(
            "StreamingCreditAgent: COMPLETE | run_id=%s | recommendation=%s | "
            "iterations=%d | latency=%.0fms",
            run_id,
            credit_memo.recommendation.value,
            len(steps),
            total_latency_ms,
        )


# ---------------------------------------------------------------------------
# WebSocket / SSE server helpers
# ---------------------------------------------------------------------------

async def stream_to_websocket(
    websocket_send,
    credit_application: Dict[str, Any],
) -> None:
    """
    Stream agent events to a WebSocket connection.

    Example integration with FastAPI WebSocket:

        @app.websocket("/ws/credit-agent")
        async def credit_agent_ws(websocket: WebSocket):
            await websocket.accept()
            application = await websocket.receive_json()
            await stream_to_websocket(websocket.send_text, application)

    Args:
        websocket_send: Callable that sends a string message (e.g. ws.send_text).
        credit_application: Dict with applicant details.
    """
    agent = StreamingCreditAgent()
    async for event in agent.stream(credit_application):
        await websocket_send(event.to_json())
        if event.is_final:
            break


async def stream_to_sse(
    credit_application: Dict[str, Any],
) -> AsyncIterator[str]:
    """
    Stream agent events as Server-Sent Events (SSE).

    Example integration with FastAPI:

        @app.get("/sse/credit-agent")
        async def credit_agent_sse(application_id: str):
            application = load_application(application_id)
            return EventSourceResponse(stream_to_sse(application))

    Yields:
        SSE-formatted strings for each agent event.
    """
    agent = StreamingCreditAgent()
    async for event in agent.stream(credit_application):
        yield event.to_sse()
        if event.is_final:
            break


def run_streaming_agent_sync(
    credit_application: Dict[str, Any],
    print_events: bool = True,
) -> List[AgentEvent]:
    """
    Synchronous wrapper that collects all streaming events.

    Useful for testing and for non-async callers (e.g. Jupyter notebooks).

    Args:
        credit_application: Dict with applicant details.
        print_events: If True, print each event as it is yielded.

    Returns:
        List of all AgentEvent objects from the run.
    """
    async def _collect():
        events = []
        agent = StreamingCreditAgent()
        async for event in agent.stream(credit_application):
            if print_events:
                ts = event.timestamp[11:19]  # HH:MM:SS
                print(f"[{ts}] {event.event_type.value:12s} iter={event.iteration} | {event.content[:100]}")
            events.append(event)
            if event.is_final:
                break
        return events

    return asyncio.run(_collect())
