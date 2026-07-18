"""
AWB Automated Credit Decision Workflow
Chapter 3: Agentic AI for Financial Risk

Avon & Wessex Bank plc — Agentic AI Platform
Model registrations:
  MR-2026-037 — Credit Decision Agent (PRA SS1/23)
  MR-2026-038 — Treasury Operations Agent (PRA SS1/23)

New in v22 (Sections 3.9A and 3.9B):
  agentic_ai_patterns.py — multi-agent topology patterns, guardrails, ReAct loop (Section 3.9A)
  mcp_servers.py         — MCP server catalogue: FCA Handbook, Bloomberg, Model Inventory (Section 3.9B)

Module inventory:
  agent.py           — ReAct loop (sequential credit decision)
  langgraph_agent.py — LangGraph 4-node stateful pipeline
  treasury_agent.py  — Parallel treasury operations (async fan-out)
  streaming_agent.py — Streaming ReAct loop (token-by-token events)
  memory.py          — Dual-store memory (Redis + PostgreSQL + pgvector)
  tools.py           — Tool registry (6 credit assessment tools)
  policy_rules.py    — AWB credit policy rule set
  credit_memo_generator.py — Credit memo dataclasses and builder
"""

# ── Core ReAct agent ──────────────────────────────────────────────────────
from credit_agent.agent import (
    CreditDecisionAgent,
    AgentRunResult,
    AgentStatus,
    AgentStep,
    ToolCallLog,
    HumanOversightCheckpoint,
    LLMClient,
)

# ── LangGraph stateful pipeline ───────────────────────────────────────────
from credit_agent.langgraph_agent import (
    CreditState,
    build_credit_graph,
    run_credit_pipeline,
    resume_human_review,
    node_document_ingestor,
    node_financial_analyser,
    node_policy_checker,
    node_memo_drafter,
    node_human_review,
)

# ── Treasury Operations Agent (parallel) ─────────────────────────────────
from credit_agent.treasury_agent import (
    TreasuryState,
    run_treasury_pipeline,
    run_treasury_pipeline_sync,
    stream_treasury_report,
    node_cash_position_agent,
    node_fx_exposure_agent,
    node_settlement_risk_agent,
    node_treasury_report,
)

# ── Streaming agent ───────────────────────────────────────────────────────
from credit_agent.streaming_agent import (
    StreamingCreditAgent,
    AgentEvent,
    AgentEventType,
    stream_to_websocket,
    stream_to_sse,
    run_streaming_agent_sync,
)

# ── Dual-store memory ─────────────────────────────────────────────────────
from credit_agent.memory import (
    AgentMemory,
    RedisWorkingMemory,
    PostgresMemory,
    AuditRecord,
    MemoEmbeddingRecord,
    get_memory,
)

# ── Domain models ─────────────────────────────────────────────────────────
from credit_agent.credit_memo_generator import CreditMemo, CreditDecision
from credit_agent.policy_rules import (
    AWBCreditPolicyRuleSet,
    PolicyBreach,
    Severity,
    DEFAULT_POLICY,
)
from credit_agent.tools import TOOL_REGISTRY

# ── Section 3.9A — Agentic AI Architecture patterns ──────────────────────
from credit_agent.agentic_ai_patterns import (
    AgentTopology,
    RiskZone,
    ReActLoop,
    ReActStep,
    GuardrailTier,
    Guardrail,
    GuardrailRegistry,
    AgentRunBudget,
    BudgetExceededError,
    SupervisorWorkerOrchestrator,
    validate_agent_output,
    AWB_AGENT_ARCHITECTURE_MAP,
    AWB_LLM_AGENTS_1_3,
    AWB_LLM_AGENT_4,
    AWB_LLM_AGENT_5,
    TOKEN_BUDGET_PER_RUN,
    COST_BUDGET_GBP_PER_RUN,
)

# ── Section 3.9B — MCP Server catalogue ──────────────────────────────────
from credit_agent.mcp_servers import (
    AWBMCPServer,
    AWBMCPServerRegistry,
    MCPToolDefinition,
    MCPToolCall,
    MCPCallAuditRecord,
    MCPFCAHandbookServer,
    MCPBloombergServer,
    MCPModelInventoryServer,
)

__all__ = [
    # ReAct agent
    "CreditDecisionAgent",
    "AgentRunResult",
    "AgentStatus",
    "AgentStep",
    "ToolCallLog",
    "HumanOversightCheckpoint",
    "LLMClient",
    # LangGraph pipeline
    "CreditState",
    "build_credit_graph",
    "run_credit_pipeline",
    "resume_human_review",
    "node_document_ingestor",
    "node_financial_analyser",
    "node_policy_checker",
    "node_memo_drafter",
    "node_human_review",
    # Treasury agent
    "TreasuryState",
    "run_treasury_pipeline",
    "run_treasury_pipeline_sync",
    "stream_treasury_report",
    "node_cash_position_agent",
    "node_fx_exposure_agent",
    "node_settlement_risk_agent",
    "node_treasury_report",
    # Streaming agent
    "StreamingCreditAgent",
    "AgentEvent",
    "AgentEventType",
    "stream_to_websocket",
    "stream_to_sse",
    "run_streaming_agent_sync",
    # Memory
    "AgentMemory",
    "RedisWorkingMemory",
    "PostgresMemory",
    "AuditRecord",
    "MemoEmbeddingRecord",
    "get_memory",
    # Domain models
    "CreditMemo",
    "CreditDecision",
    "AWBCreditPolicyRuleSet",
    "PolicyBreach",
    "Severity",
    "DEFAULT_POLICY",
    "TOOL_REGISTRY",
    # Section 3.9A — Agentic AI Architecture
    "AgentTopology",
    "RiskZone",
    "ReActLoop",
    "ReActStep",
    "GuardrailTier",
    "Guardrail",
    "GuardrailRegistry",
    "AgentRunBudget",
    "BudgetExceededError",
    "SupervisorWorkerOrchestrator",
    "validate_agent_output",
    "AWB_AGENT_ARCHITECTURE_MAP",
    "AWB_LLM_AGENTS_1_3",
    "AWB_LLM_AGENT_4",
    "AWB_LLM_AGENT_5",
    "TOKEN_BUDGET_PER_RUN",
    "COST_BUDGET_GBP_PER_RUN",
    # Section 3.9B — MCP Servers
    "AWBMCPServer",
    "AWBMCPServerRegistry",
    "MCPToolDefinition",
    "MCPToolCall",
    "MCPCallAuditRecord",
    "MCPFCAHandbookServer",
    "MCPBloombergServer",
    "MCPModelInventoryServer",
]

__version__ = "3.9.0"   # bumped for Sections 3.9A + 3.9B additions (book v22)
__model_registrations__ = ["MR-2026-037", "MR-2026-038"]
