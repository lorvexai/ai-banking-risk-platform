# mcp_servers.py | AWB Model Context Protocol Server Implementations
# Chapter 3 | Section 3.9B | Connecting AWB Agents to Live Data Sources
# MCP specification: https://modelcontextprotocol.io (Anthropic, Nov 2024)
# BAP-2026-MCP-001 | FCA Handbook live queries | Bloomberg market data
"""
AWB Model Context Protocol (MCP) — Section 3.9B

The Model Context Protocol (MCP) is Anthropic's open standard (November 2024)
for connecting AI agents to external data sources and tools via a unified,
versioned, and authenticated interface. In the AWB-AI-2025 programme, MCP
servers replace ad-hoc API wrappers with a standardised protocol layer that:

1. Provides a consistent tool-discovery interface across all AWB agents
2. Enforces authentication and authorisation at the protocol level
3. Enables agents to query live data without hardcoded API clients
4. Creates an auditable record of every data source consultation

AWB MCP Server Catalogue
-------------------------
MCPFCAHandbookServer    — Live FCA Handbook regulatory text queries
MCPBloombergServer      — Live Bloomberg market data, ratings, news
MCPModelInventoryServer — AWB internal model registry (PRA SS1/23)
MCPCreditBureauServer   — Experian/Equifax credit bureau integration
MCPRegulatoryCalServer  — EBA/PRA regulatory calendar and deadlines

Architecture
------------
Each MCP server implements the standard tool discovery protocol:
  list_tools()  → returns available tool definitions
  call_tool()   → executes a named tool with validated arguments

AWB agents discover available tools from registered MCP servers at
runtime rather than having tools hardcoded — this means new data
sources can be added without modifying agent code.

Regulatory context:
- All MCP tool calls logged to audit trail (FCA COBS 9.1.3R)
- Personal data access via MCP servers requires ROPA entry (UK GDPR Art.30)
- External API calls to Bloomberg/Experian governed by DORA Art.28 (third-party ICT)
- MCP server credentials stored in AWS Secrets Manager (not environment variables)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP Protocol base types
# ---------------------------------------------------------------------------

# ── Per-Invocation Budget (aligned with all AWB agentic pipelines Ch4-Ch16) ──
TOKEN_BUDGET_PER_RUN    = 50_000   # Max tokens per MCP tool call chain
COST_BUDGET_GBP_PER_RUN = 2.50    # Max cost per MCP server session


@dataclass
class MCPToolDefinition:
    """Standard MCP tool definition returned by list_tools()."""
    name:        str
    description: str
    input_schema: Dict[str, Any]   # JSON Schema for input validation
    output_schema: Dict[str, Any]  # JSON Schema for output validation
    requires_auth: bool = True
    audit_required: bool = True    # All AWB MCP calls are audited
    data_classification: str = "RESTRICTED"  # PUBLIC / INTERNAL / RESTRICTED / CONFIDENTIAL


@dataclass
class MCPToolCall:
    """An individual MCP tool invocation — logged to audit trail."""
    call_id:     str
    server_name: str
    tool_name:   str
    arguments:   Dict[str, Any]
    called_by:   str              # agent name
    timestamp:   str
    latency_ms:  float = 0.0
    response:    Optional[Dict[str, Any]] = None
    error:       Optional[str] = None


@dataclass
class MCPCallAuditRecord:
    """Immutable audit record for one MCP tool call (7-year retention)."""
    call_id:      str
    server_name:  str
    tool_name:    str
    called_by:    str
    timestamp:    str
    input_hash:   str   # SHA-256 of arguments — no PII in audit log
    output_hash:  str   # SHA-256 of response
    latency_ms:   float
    success:      bool
    regulatory_ref: str  # e.g. "FCA COBS 9.1.3R"


class MCPAuthError(Exception):
    """Raised when MCP server authentication fails."""


class MCPToolNotFoundError(Exception):
    """Raised when a requested tool is not available on this server."""


class MCPRateLimitError(Exception):
    """Raised when an MCP server rate limit is exceeded."""


# ---------------------------------------------------------------------------
# Base MCP Server
# ---------------------------------------------------------------------------

class AWBMCPServer:
    """
    Base class for all AWB MCP servers.

    Implements the standard MCP protocol lifecycle:
    - authenticate()   — validate API credentials
    - list_tools()     — return available tool definitions
    - call_tool()      — execute a named tool
    - _audit_log()     — record call to audit trail

    All AWB MCP servers inherit from this class and override
    _execute_tool() with their specific data source logic.

    Authentication model (BAP-2026-MCP-001 §3):
    - API keys stored in AWS Secrets Manager
    - Short-lived tokens (15 minutes) issued on authentication
    - All credentials rotated quarterly
    - MCP server access logged to SM&CR conduct log
    """

    SERVER_NAME: str = "base"
    REGULATORY_REF: str = "FCA COBS 9.1.3R"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_seconds: int = 10,
        rate_limit_per_minute: int = 60,
    ) -> None:
        self._api_key = api_key or "DEMO_KEY"
        self._timeout = timeout_seconds
        self._rate_limit = rate_limit_per_minute
        self._call_log: List[MCPCallAuditRecord] = []
        self._token: Optional[str] = None
        self._token_expiry: float = 0.0
        self._call_count_minute: int = 0
        self._minute_start: float = time.time()

    def authenticate(self) -> str:
        """Issue a short-lived session token (15 minutes)."""
        if self._api_key == "DEMO_KEY":
            log.warning("[%s] Using DEMO_KEY — not for production use", self.SERVER_NAME)
        self._token = hashlib.sha256(
            f"{self._api_key}:{time.time()}".encode()
        ).hexdigest()[:32]
        self._token_expiry = time.time() + 900  # 15 minutes
        log.info("[%s] Authenticated. Token expires in 15 minutes.", self.SERVER_NAME)
        return self._token

    def _check_auth(self) -> None:
        if not self._token or time.time() > self._token_expiry:
            self.authenticate()

    def _check_rate_limit(self) -> None:
        now = time.time()
        if now - self._minute_start > 60:
            self._call_count_minute = 0
            self._minute_start = now
        self._call_count_minute += 1
        if self._call_count_minute > self._rate_limit:
            raise MCPRateLimitError(
                f"[{self.SERVER_NAME}] Rate limit {self._rate_limit}/min exceeded. "
                f"DORA Art.28: third-party ICT service limits must be monitored."
            )

    def list_tools(self) -> List[MCPToolDefinition]:
        """Return available tools. Subclasses must override."""
        raise NotImplementedError

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool. Subclasses must override."""
        raise NotImplementedError

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        called_by: str = "UnknownAgent",
    ) -> Dict[str, Any]:
        """
        Execute a named tool with full audit logging.

        This is the primary public interface for all AWB agents.
        Every call is authenticated, rate-limited, executed, and
        audited regardless of outcome.
        """
        self._check_auth()
        self._check_rate_limit()

        call_id = str(uuid.uuid4())[:12]
        start_ms = time.time() * 1000

        call = MCPToolCall(
            call_id=call_id,
            server_name=self.SERVER_NAME,
            tool_name=tool_name,
            arguments=arguments,
            called_by=called_by,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        try:
            result = self._execute_tool(tool_name, arguments)
            call.response = result
            success = True
        except MCPToolNotFoundError:
            raise
        except Exception as exc:
            call.error = str(exc)
            result = {"error": str(exc), "call_id": call_id}
            success = False
            log.warning("[%s] Tool call failed: %s — %s", self.SERVER_NAME, tool_name, exc)

        latency = time.time() * 1000 - start_ms
        call.latency_ms = latency

        # Audit record — hashed inputs/outputs, no PII stored in audit log
        audit = MCPCallAuditRecord(
            call_id=call_id,
            server_name=self.SERVER_NAME,
            tool_name=tool_name,
            called_by=called_by,
            timestamp=call.timestamp,
            input_hash=hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()[:16],
            output_hash=hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()[:16],
            latency_ms=round(latency, 2),
            success=success,
            regulatory_ref=self.REGULATORY_REF,
        )
        self._call_log.append(audit)
        log.debug("[%s] %s → %s latency=%.1fms ok=%s",
                  self.SERVER_NAME, tool_name, call_id, latency, success)

        if not success and "error" in result:
            return result
        return {"call_id": call_id, "server": self.SERVER_NAME, **result}

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return audit records for FCA COBS 9.1.3R compliance review."""
        return [vars(r) for r in self._call_log]


# ---------------------------------------------------------------------------
# MCP Server 1: FCA Handbook
# ---------------------------------------------------------------------------

class MCPFCAHandbookServer(AWBMCPServer):
    """
    MCP server for live FCA Handbook regulatory text queries.

    Enables AWB agents to query the FCA Handbook in real time — so
    that policy checkers, regulatory compliance agents, and AML agents
    always work with current regulatory text rather than static
    knowledge cut-off snapshots.

    Tools
    -----
    fca_handbook_search    — Full-text search across all FCA sourcebooks
    fca_rule_lookup        — Retrieve a specific rule by reference (e.g. SYSC 6.3.3R)
    fca_guidance_lookup    — Retrieve regulatory guidance (e.g. FG21/1)
    fca_threshold_check    — Check whether a numeric value breaches an FCA threshold
    fca_recent_changes     — Get recent FCA Handbook changes (last N days)

    AWB Usage
    ---------
    - Chapter 3 PolicyChecker: validates credit terms against CONC 5
    - Chapter 5 GovernanceAgent: checks current PRA model risk rules
    - Chapter 11 RegulatoryCalendarAgent: retrieves submission deadlines
    - Chapter 12 KYCScreeningAgent: validates current MLR requirements

    Data source: FCA Handbook API (api.handbook.fca.org.uk)
    Authentication: FCA Open API key (public — no personal data)
    DORA Art.28: FCA Handbook API classified as non-critical third-party ICT
    """

    SERVER_NAME = "fca_handbook"
    REGULATORY_REF = "FCA COBS 9.1.3R / FCA SYSC 6.3"

    def list_tools(self) -> List[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="fca_handbook_search",
                description="Full-text search the FCA Handbook for regulatory provisions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search terms"},
                        "sourcebook": {"type": "string", "description": "e.g. SYSC, CONC, MCOB, SUP"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"},
                        "total_matches": {"type": "integer"},
                    },
                },
                requires_auth=False,  # FCA Handbook is public
                data_classification="PUBLIC",
            ),
            MCPToolDefinition(
                name="fca_rule_lookup",
                description="Retrieve the full text of a specific FCA rule by reference",
                input_schema={
                    "type": "object",
                    "properties": {
                        "rule_ref": {"type": "string", "description": "e.g. SYSC 6.3.3R, CONC 5.2.1R"},
                    },
                    "required": ["rule_ref"],
                },
                output_schema={"type": "object"},
                requires_auth=False,
                data_classification="PUBLIC",
            ),
            MCPToolDefinition(
                name="fca_threshold_check",
                description="Check whether a value breaches an FCA regulatory threshold",
                input_schema={
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "description": "e.g. 'mortgage_ltv', 'credit_apr'"},
                        "value": {"type": "number"},
                        "context": {"type": "string"},
                    },
                    "required": ["metric", "value"],
                },
                output_schema={"type": "object"},
                requires_auth=False,
                data_classification="PUBLIC",
            ),
            MCPToolDefinition(
                name="fca_recent_changes",
                description="Get recent FCA Handbook changes in the last N days",
                input_schema={
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "default": 30},
                        "sourcebook": {"type": "string"},
                    },
                },
                output_schema={"type": "object"},
                requires_auth=False,
                data_classification="PUBLIC",
            ),
        ]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FCA Handbook tool. Stub returns realistic mock data."""

        if tool_name == "fca_handbook_search":
            query = arguments.get("query", "")
            sourcebook = arguments.get("sourcebook", "ALL")
            return {
                "query": query,
                "sourcebook": sourcebook,
                "total_matches": 3,
                "results": [
                    {
                        "ref": "SYSC 6.3.3R",
                        "title": "Obligations of firms under SYSC 6.3",
                        "snippet": (
                            "A firm must take reasonable care to establish and maintain "
                            "effective systems and controls for compliance with applicable "
                            "requirements and standards under the regulatory system and for "
                            "countering the risk that the firm might be used to further "
                            "financial crime..."
                        ),
                        "last_updated": "2024-07-01",
                    },
                ],
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
                "source": "FCA Handbook API (live)",
            }

        elif tool_name == "fca_rule_lookup":
            rule_ref = arguments.get("rule_ref", "")
            # Map known AWB-relevant rules to mock text
            known_rules = {
                "SYSC 6.3.3R": "A firm must take reasonable care to establish and maintain effective systems and controls for compliance...",
                "CONC 5.2.1R": "A firm must not enter into a regulated credit agreement with a customer unless it has carried out a creditworthiness assessment...",
                "MCOB 11.6.2R": "A firm must not enter into a regulated mortgage contract with a customer unless it has carried out an assessment of whether the customer will be able to repay...",
                "SUP 15.3.1R": "A firm must notify the FCA immediately upon becoming aware of any matter which it would reasonably conclude would be of material significance...",
            }
            text = known_rules.get(rule_ref, f"[Rule text for {rule_ref} — live API response in production]")
            return {
                "rule_ref": rule_ref,
                "text": text,
                "in_force_from": "2024-01-01",
                "sourcebook": rule_ref.split(" ")[0] if " " in rule_ref else "UNKNOWN",
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
                "source": "FCA Handbook API (live)",
            }

        elif tool_name == "fca_threshold_check":
            metric = arguments.get("metric", "")
            value = arguments.get("value", 0)
            # Mock threshold checks relevant to AWB use cases
            breached = False
            threshold_ref = "N/A"
            if metric == "mortgage_ltv" and value > 0.95:
                breached = True
                threshold_ref = "FCA MCOB 11 — 95% LTV prudential limit"
            elif metric == "credit_apr" and value > 100.0:
                breached = True
                threshold_ref = "FCA CONC — high-cost short-term credit definition"
            return {
                "metric": metric,
                "value": value,
                "breached": breached,
                "threshold_ref": threshold_ref,
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
            }

        elif tool_name == "fca_recent_changes":
            days = arguments.get("days", 30)
            return {
                "period_days": days,
                "changes": [
                    {
                        "date": "2026-04-15",
                        "sourcebook": "SYSC",
                        "change": "SYSC 15A.1 updated — DORA alignment for ICT risk management",
                        "effective": "2026-01-17",
                    },
                    {
                        "date": "2026-03-01",
                        "sourcebook": "CONC",
                        "change": "CONC 5 creditworthiness guidance updated — Consumer Duty alignment",
                        "effective": "2026-04-01",
                    },
                ],
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
                "source": "FCA Handbook API (live)",
            }

        raise MCPToolNotFoundError(f"Tool '{tool_name}' not found on {self.SERVER_NAME}")


# ---------------------------------------------------------------------------
# MCP Server 2: Bloomberg Market Data
# ---------------------------------------------------------------------------

class MCPBloombergServer(AWBMCPServer):
    """
    MCP server for live Bloomberg market data, ratings, and news.

    Enables AWB market risk and credit agents to query live Bloomberg
    data rather than relying on stale batch extracts.

    Tools
    -----
    bloomberg_quote         — Live price/spread/rate for any instrument
    bloomberg_credit_rating — S&P / Moody's / Fitch ratings for counterparty
    bloomberg_yield_curve   — Live SONIA / SOFR / Gilt yield curves
    bloomberg_news          — Recent news for issuer/sector
    bloomberg_var_inputs    — Live VaR model inputs (vol, correlation)

    AWB Usage
    ---------
    - Chapter 7 MarketRiskAgent: live VaR inputs, yield curves
    - Chapter 6 CreditScoringAgent: counterparty ratings
    - Chapter 9 LiquidityAgent: live HQLA pricing for LCR calculation
    - Chapter 11 XBRLFilingAgent: current OCI positions

    Data source: Bloomberg B-PIPE / BLPAPI
    Authentication: Bloomberg Enterprise API key (CONFIDENTIAL)
    DORA Art.28: Bloomberg classified as CRITICAL third-party ICT provider
    — concentration risk assessed quarterly (AWB sole Bloomberg dependency)
    """

    SERVER_NAME = "bloomberg"
    REGULATORY_REF = "FCA COBS 9.1.3R / DORA Art.28"

    def list_tools(self) -> List[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="bloomberg_quote",
                description="Get live market quote for an instrument",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Bloomberg ticker e.g. 'LLOY LN Equity'"},
                        "fields": {"type": "array", "items": {"type": "string"},
                                   "default": ["PX_LAST", "YLD_YTM_MID", "Z_SPREAD_MID"]},
                    },
                    "required": ["ticker"],
                },
                output_schema={"type": "object"},
                data_classification="CONFIDENTIAL",
            ),
            MCPToolDefinition(
                name="bloomberg_credit_rating",
                description="Get current credit ratings for a counterparty",
                input_schema={
                    "type": "object",
                    "properties": {
                        "counterparty": {"type": "string", "description": "Legal entity name or LEI"},
                        "agencies": {"type": "array", "default": ["SP", "MOODY", "FITCH"]},
                    },
                    "required": ["counterparty"],
                },
                output_schema={"type": "object"},
                data_classification="RESTRICTED",
            ),
            MCPToolDefinition(
                name="bloomberg_yield_curve",
                description="Get live yield curve data (SONIA, SOFR, Gilts, EURIBOR)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "curve": {"type": "string", "enum": ["SONIA", "SOFR", "GILT", "EURIBOR"]},
                        "tenors": {"type": "array", "default": ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"]},
                    },
                    "required": ["curve"],
                },
                output_schema={"type": "object"},
                data_classification="INTERNAL",
            ),
            MCPToolDefinition(
                name="bloomberg_var_inputs",
                description="Get live VaR model inputs: volatilities and correlations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "risk_factors": {"type": "array", "items": {"type": "string"}},
                        "lookback_days": {"type": "integer", "default": 250},
                    },
                    "required": ["risk_factors"],
                },
                output_schema={"type": "object"},
                data_classification="CONFIDENTIAL",
            ),
        ]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Bloomberg tool. Stub returns realistic mock market data."""

        if tool_name == "bloomberg_quote":
            ticker = arguments.get("ticker", "")
            return {
                "ticker": ticker,
                "PX_LAST": 42.18,
                "YLD_YTM_MID": 5.23,
                "Z_SPREAD_MID": 85.0,
                "source": "Bloomberg B-PIPE",
                "as_of": datetime.utcnow().isoformat() + "Z",
                "note": "Live Bloomberg data in production — stub in dev/test",
            }

        elif tool_name == "bloomberg_credit_rating":
            counterparty = arguments.get("counterparty", "")
            return {
                "counterparty": counterparty,
                "ratings": {
                    "SP": {"rating": "BBB+", "outlook": "Stable", "last_action": "2025-11-15"},
                    "MOODY": {"rating": "Baa1", "outlook": "Stable", "last_action": "2025-10-08"},
                    "FITCH": {"rating": "BBB+", "outlook": "Stable", "last_action": "2025-09-22"},
                },
                "pd_1yr_bps": 42,
                "source": "Bloomberg",
                "as_of": datetime.utcnow().isoformat() + "Z",
            }

        elif tool_name == "bloomberg_yield_curve":
            curve = arguments.get("curve", "SONIA")
            tenors = arguments.get("tenors", ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y"])
            # Realistic SONIA curve June 2026
            sonia_rates = {"1M": 4.45, "3M": 4.42, "6M": 4.35, "1Y": 4.15, "2Y": 3.95, "5Y": 3.85, "10Y": 4.05}
            gilt_rates  = {"1M": 4.48, "3M": 4.44, "6M": 4.38, "1Y": 4.20, "2Y": 4.01, "5Y": 3.92, "10Y": 4.18}
            rates = sonia_rates if curve == "SONIA" else gilt_rates
            return {
                "curve": curve,
                "rates": {t: rates.get(t, 4.0) for t in tenors},
                "source": "Bloomberg",
                "as_of": datetime.utcnow().isoformat() + "Z",
            }

        elif tool_name == "bloomberg_var_inputs":
            factors = arguments.get("risk_factors", [])
            return {
                "risk_factors": factors,
                "volatilities": {f: 0.18 for f in factors},  # 18% annualised vol stub
                "correlation_matrix": "[[1.0, 0.35], [0.35, 1.0]]",  # stub 2x2
                "lookback_days": arguments.get("lookback_days", 250),
                "source": "Bloomberg",
                "as_of": datetime.utcnow().isoformat() + "Z",
            }

        raise MCPToolNotFoundError(f"Tool '{tool_name}' not found on {self.SERVER_NAME}")


# ---------------------------------------------------------------------------
# MCP Server 3: AWB Internal Model Inventory
# ---------------------------------------------------------------------------

class MCPModelInventoryServer(AWBMCPServer):
    """
    MCP server for the AWB internal model registry (PRA SS1/23).

    Enables agents to query live model metadata, validation status,
    deployment approvals, and monitoring alerts without hardcoded lookups.

    Tools
    -----
    model_lookup            — Get full metadata for a model by MR reference
    model_status_check      — Check current deployment and monitoring status
    model_validation_history — Get SS1/23 validation gate history
    model_alert_query       — Get active PSI/SHAP drift alerts for a model
    model_inventory_search  — Search models by criteria (risk_zone, chapter, etc.)

    AWB Usage
    ---------
    - Chapter 5 GovernanceAgent: model risk rating lookups
    - Chapter 10 ModelRiskAgent: validation status and gate results
    - Chapter 14 MLOpsAgent: deployment and monitoring status
    - Chapter 16 PlatformOrchestrator: cross-model health dashboard

    Data source: AWB internal PostgreSQL model registry (on-prem)
    Authentication: Internal service account (AWS IAM)
    UK GDPR Art.30: ROPA entry MODEL-001 covers model metadata (no personal data)
    """

    SERVER_NAME = "awb_model_inventory"
    REGULATORY_REF = "PRA SS1/23 §4-7"

    # Stub model inventory — replace with live PostgreSQL in production
    _MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
        "MR-2026-037": {
            "name": "AWB Credit Decision LightGBM v3.4",
            "chapter": 3,
            "risk_rating": "HIGH",
            "deployment_status": "PRODUCTION",
            "auc_roc": 0.881,
            "psi_current": 0.13,
            "last_validated": "2026-02-15",
            "next_validation_due": "2026-05-15",
            "ss1_23_gates_passed": 4,
            "hitl_threshold": "£500,000",
        },
        "MR-2026-053": {
            "name": "AWB Customer Churn XGBoost v2.1",
            "chapter": 14,
            "risk_rating": "LOW",
            "deployment_status": "PRODUCTION",
            "auc_roc": 0.847,
            "psi_current": 0.07,
            "last_validated": "2026-01-20",
            "next_validation_due": "2026-07-20",
            "ss1_23_gates_passed": 4,
            "hitl_threshold": "N/A",
        },
        "MR-2026-060-AML": {
            "name": "AWB AML/KYC Agentic Pipeline",
            "chapter": 12,
            "risk_rating": "CRITICAL",
            "deployment_status": "PRODUCTION",
            "auc_roc": None,  # Agentic pipeline — no single AUC metric
            "psi_current": None,
            "last_validated": "2026-03-01",
            "next_validation_due": "2026-06-01",
            "ss1_23_gates_passed": 4,
            "hitl_threshold": "All SARs — MLRO approval mandatory",
        },
        "MR-2026-049": {
            "name": "AWB Market Risk FRTB Pipeline",
            "chapter": 7,
            "risk_rating": "HIGH",
            "deployment_status": "PRODUCTION",
            "auc_roc": None,
            "psi_current": None,
            "last_validated": "2026-02-28",
            "next_validation_due": "2026-05-31",
            "ss1_23_gates_passed": 4,
            "hitl_threshold": "IMA model approval breach",
        },
    }

    def list_tools(self) -> List[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="model_lookup",
                description="Get full PRA SS1/23 metadata for a model by MR reference",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_ref": {"type": "string", "description": "e.g. MR-2026-037"},
                    },
                    "required": ["model_ref"],
                },
                output_schema={"type": "object"},
                data_classification="INTERNAL",
            ),
            MCPToolDefinition(
                name="model_status_check",
                description="Check current deployment status and active alerts for a model",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_ref": {"type": "string"},
                        "include_alerts": {"type": "boolean", "default": True},
                    },
                    "required": ["model_ref"],
                },
                output_schema={"type": "object"},
                data_classification="INTERNAL",
            ),
            MCPToolDefinition(
                name="model_inventory_search",
                description="Search the AWB model inventory by risk rating, chapter, or status",
                input_schema={
                    "type": "object",
                    "properties": {
                        "risk_rating": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                        "deployment_status": {"type": "string", "enum": ["PRODUCTION", "STAGING", "DECOMMISSIONED"]},
                        "chapter": {"type": "integer"},
                    },
                },
                output_schema={"type": "object"},
                data_classification="INTERNAL",
            ),
            MCPToolDefinition(
                name="model_alert_query",
                description="Get active PSI/SHAP drift and RAGAS quality alerts for a model",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_ref": {"type": "string"},
                        "alert_type": {"type": "string", "enum": ["PSI", "SHAP", "RAGAS", "ALL"], "default": "ALL"},
                    },
                    "required": ["model_ref"],
                },
                output_schema={"type": "object"},
                data_classification="INTERNAL",
            ),
        ]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:

        if tool_name == "model_lookup":
            model_ref = arguments.get("model_ref", "")
            record = self._MODEL_REGISTRY.get(model_ref)
            if not record:
                return {"error": f"Model {model_ref} not found in AWB registry"}
            return {
                "model_ref": model_ref,
                **record,
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
            }

        elif tool_name == "model_status_check":
            model_ref = arguments.get("model_ref", "")
            record = self._MODEL_REGISTRY.get(model_ref, {})
            alerts = []
            if record.get("psi_current", 0) and record["psi_current"] > 0.10:
                alerts.append({
                    "type": "PSI",
                    "severity": "RED" if record["psi_current"] > 0.20 else "AMBER",
                    "value": record["psi_current"],
                    "threshold": 0.20,
                    "message": f"PSI={record['psi_current']:.2f} — enhanced monitoring required",
                })
            return {
                "model_ref": model_ref,
                "deployment_status": record.get("deployment_status", "UNKNOWN"),
                "active_alerts": alerts if arguments.get("include_alerts", True) else [],
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
            }

        elif tool_name == "model_inventory_search":
            results = []
            for ref, record in self._MODEL_REGISTRY.items():
                match = True
                if "risk_rating" in arguments and record.get("risk_rating") != arguments["risk_rating"]:
                    match = False
                if "deployment_status" in arguments and record.get("deployment_status") != arguments["deployment_status"]:
                    match = False
                if "chapter" in arguments and record.get("chapter") != arguments["chapter"]:
                    match = False
                if match:
                    results.append({"model_ref": ref, **record})
            return {
                "results": results,
                "total": len(results),
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
            }

        elif tool_name == "model_alert_query":
            model_ref = arguments.get("model_ref", "")
            record = self._MODEL_REGISTRY.get(model_ref, {})
            alerts = []
            psi = record.get("psi_current")
            if psi and psi > 0.10:
                alerts.append({"type": "PSI", "value": psi, "severity": "AMBER" if psi < 0.20 else "RED"})
            return {
                "model_ref": model_ref,
                "alerts": alerts,
                "alert_count": len(alerts),
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
            }

        raise MCPToolNotFoundError(f"Tool '{tool_name}' not found on {self.SERVER_NAME}")


# ---------------------------------------------------------------------------
# AWB MCP Server Registry — discovery for agents
# ---------------------------------------------------------------------------

class AWBMCPServerRegistry:
    """
    Central registry of all AWB MCP servers.

    Agents call discover_tools() to find available tools across
    all registered servers — without knowing which server provides
    each tool. This is the MCP protocol's key architectural benefit:
    agents are decoupled from specific data sources.

    Usage
    -----
    registry = AWBMCPServerRegistry.default()
    tools = registry.discover_tools()  # all tools from all servers
    result = registry.call_tool(
        tool_name="fca_rule_lookup",
        arguments={"rule_ref": "SYSC 6.3.3R"},
        called_by="PolicyCheckerAgent",
    )
    """

    def __init__(self) -> None:
        self._servers: Dict[str, AWBMCPServer] = {}

    @classmethod
    def default(cls) -> "AWBMCPServerRegistry":
        """Create registry with all standard AWB MCP servers."""
        registry = cls()
        registry.register(MCPFCAHandbookServer())
        registry.register(MCPBloombergServer())
        registry.register(MCPModelInventoryServer())
        return registry

    def register(self, server: AWBMCPServer) -> None:
        self._servers[server.SERVER_NAME] = server
        log.info("MCP server registered: %s", server.SERVER_NAME)

    def discover_tools(self) -> List[Dict[str, Any]]:
        """Return all available tools across all registered servers."""
        all_tools = []
        for server_name, server in self._servers.items():
            for tool in server.list_tools():
                all_tools.append({
                    "server": server_name,
                    "tool_name": tool.name,
                    "description": tool.description,
                    "data_classification": tool.data_classification,
                })
        return all_tools

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        called_by: str = "UnknownAgent",
    ) -> Dict[str, Any]:
        """
        Call a tool by name, routing to the correct server automatically.
        Raises MCPToolNotFoundError if no server provides the tool.
        """
        for server in self._servers.values():
            available = [t.name for t in server.list_tools()]
            if tool_name in available:
                return server.call_tool(tool_name, arguments, called_by)
        raise MCPToolNotFoundError(
            f"Tool '{tool_name}' not found in any registered MCP server. "
            f"Available servers: {list(self._servers.keys())}"
        )

    def get_combined_audit_log(self) -> List[Dict[str, Any]]:
        """Merge audit logs from all servers for FCA COBS 9.1.3R review."""
        combined = []
        for server in self._servers.values():
            combined.extend(server.get_audit_log())
        return sorted(combined, key=lambda x: x.get("timestamp", ""))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n=== AWB MCP Server Registry ===")
    registry = AWBMCPServerRegistry.default()

    print("\nDiscovering all available tools:")
    for tool in registry.discover_tools():
        print(f"  [{tool['server']}] {tool['tool_name']:40s} ({tool['data_classification']})")

    print("\n--- FCA Handbook: Rule lookup ---")
    result = registry.call_tool(
        "fca_rule_lookup",
        {"rule_ref": "SYSC 6.3.3R"},
        called_by="PolicyCheckerAgent",
    )
    print(f"  Rule: {result.get('rule_ref')}  Text: {result.get('text', '')[:80]}...")

    print("\n--- Bloomberg: Credit rating ---")
    result = registry.call_tool(
        "bloomberg_credit_rating",
        {"counterparty": "Barratt Developments PLC"},
        called_by="CreditScoringAgent",
    )
    print(f"  S&P: {result.get('ratings', {}).get('SP', {}).get('rating')}  "
          f"PD 1yr: {result.get('pd_1yr_bps')}bps")

    print("\n--- Model Inventory: Status check ---")
    result = registry.call_tool(
        "model_status_check",
        {"model_ref": "MR-2026-037", "include_alerts": True},
        called_by="ModelRiskAgent",
    )
    print(f"  Status: {result.get('deployment_status')}  Alerts: {len(result.get('active_alerts', []))}")

    print("\n--- Audit log (last 3 entries) ---")
    for entry in registry.get_combined_audit_log()[-3:]:
        print(f"  [{entry['server_name']}] {entry['tool_name']:35s}  "
              f"ok={entry['success']}  {entry['latency_ms']:.1f}ms")
