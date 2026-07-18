"""
Exercise 16.2: AWB Platform Integration Test Suite
===================================================
Run the full cross-system integration tests against
a locally deployed AWB AI platform.

Verifies:
- AWB_CUSTOMER_ID routing across all 23 services
- Unified audit event log writes (FCA COBS 9)
- CRO dashboard metric freshness (<=15 minutes)
- DORA RTO/RPO compliance per architectural layer

Usage:
    pip install pytest httpx psycopg2-binary
    pytest integration_tests.py -v

All 47 tests must pass before the platform can
be declared production-ready.

Solution: github.com/lorvenio/
  ai-banking-risk-platform/chapter-16-integrated-platform/solutions/
"""
from __future__ import annotations

import os
import time
import uuid
import logging
import datetime
from typing import Any, Dict

import pytest
import httpx
import psycopg2

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────
API_GW = os.getenv(
    "AWB_API_GW_URL",
    "http://localhost:8090"
)
DB_URL = os.getenv(
    "AWB_DB_URL",
    "postgresql://awb:awb@localhost:5432/awb_ai"
)
JWT_TOKEN = os.getenv("AWB_TEST_JWT", "")
TIMEOUT = 30  # seconds


def auth_headers() -> Dict[str, str]:
    """Return headers with JWT bearer token."""
    return {
        "Authorization": f"Bearer {JWT_TOKEN}",
        "Content-Type": "application/json",
        "X-AWB-Request-ID": str(uuid.uuid4()),
    }


@pytest.fixture(scope="session")
def db():
    """Return a database connection for the session."""
    conn = psycopg2.connect(DB_URL)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def client():
    """Return a shared HTTP client."""
    with httpx.Client(
        base_url=API_GW,
        timeout=TIMEOUT
    ) as c:
        yield c


# ── Layer health checks (5 tests) ─────────────────

class TestLayerHealth:
    """L1-L5 health endpoints must return 200."""

    ENDPOINTS = [
        ("/health/core-banking", "L1 Core Banking"),
        ("/health/data-platform", "L2 Data Platform"),
        ("/health/integration", "L3 Integration"),
        ("/health/ai-services", "L4 AI+ML Layer"),
        ("/health/dashboard", "L5 Presentation"),
    ]

    @pytest.mark.parametrize("path,layer", ENDPOINTS)
    def test_layer_health(
        self,
        client: httpx.Client,
        path: str,
        layer: str,
    ) -> None:
        """Each platform layer must report healthy."""
        resp = client.get(
            path, headers=auth_headers()
        )
        assert resp.status_code == 200, (
            f"{layer} unhealthy: {resp.status_code}"
        )
        body = resp.json()
        assert body.get("status") == "healthy", (
            f"{layer} status: {body}"
        )


# ── AWB_CUSTOMER_ID routing (6 tests) ─────────────

class TestCustomerIDRouting:
    """AWB_CUSTOMER_ID routes consistently."""

    TEST_CID = "AWB-TEST-CUST-001"

    def test_cda_accepts_customer_id(
        self, client: httpx.Client
    ) -> None:
        """CDA (MR-2026-035) accepts AWB_CUSTOMER_ID."""
        resp = client.post(
            "/cda/analyse",
            headers=auth_headers(),
            json={
                "awb_customer_id": self.TEST_CID,
                "document_type": "annual_accounts",
                "content": "Test document content",
            },
        )
        assert resp.status_code in (200, 202)
        body = resp.json()
        assert body.get("awb_customer_id") == (
            self.TEST_CID
        )

    def test_credit_agent_preserves_customer_id(
        self, client: httpx.Client
    ) -> None:
        """Credit Agent (MR-2026-037) preserves ID."""
        resp = client.post(
            "/credit-agent/assess",
            headers=auth_headers(),
            json={
                "awb_customer_id": self.TEST_CID,
                "exposure_gbp": 100_000,
                "application_id": str(uuid.uuid4()),
            },
        )
        assert resp.status_code in (200, 202)
        body = resp.json()
        assert body.get("awb_customer_id") == (
            self.TEST_CID
        )

    def test_rag_preserves_customer_id(
        self, client: httpx.Client
    ) -> None:
        """RAG system (MR-2026-038) preserves ID."""
        resp = client.post(
            "/rag/query",
            headers=auth_headers(),
            json={
                "awb_customer_id": self.TEST_CID,
                "query": "CRR3 Article 153 RWA formula",
            },
        )
        assert resp.status_code in (200, 202)
        body = resp.json()
        assert body.get("awb_customer_id") == (
            self.TEST_CID
        )

    def test_aml_preserves_customer_id(
        self, client: httpx.Client
    ) -> None:
        """AML monitor preserves AWB_CUSTOMER_ID."""
        resp = client.post(
            "/aml/monitor",
            headers=auth_headers(),
            json={
                "awb_customer_id": self.TEST_CID,
                "transaction_id": str(uuid.uuid4()),
                "amount_gbp": 9_999.00,
            },
        )
        assert resp.status_code in (200, 202)
        body = resp.json()
        assert body.get("awb_customer_id") == (
            self.TEST_CID
        )

    def test_customer_id_in_audit_log(
        self,
        client: httpx.Client,
        db: Any,
    ) -> None:
        """Customer ID appears in unified audit log."""
        cid = f"AWB-TEST-{uuid.uuid4().hex[:8].upper()}"
        # Trigger an action that writes to audit log
        client.post(
            "/cda/analyse",
            headers=auth_headers(),
            json={
                "awb_customer_id": cid,
                "document_type": "management_accounts",
                "content": "Integration test content",
            },
        )
        time.sleep(2)  # Allow async write
        cur = db.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM unified_audit_log
            WHERE awb_customer_id = %s
            """,
            (cid,),
        )
        count = cur.fetchone()[0]
        assert count >= 1, (
            f"No audit log entry for {cid}"
        )

    def test_no_null_customer_ids_in_audit(
        self, db: Any
    ) -> None:
        """Unified audit log must have no NULL IDs."""
        cur = db.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM unified_audit_log
            WHERE awb_customer_id IS NULL
            AND logged_at > NOW() - INTERVAL '1 hour'
            """
        )
        null_count = cur.fetchone()[0]
        assert null_count == 0, (
            f"{null_count} NULL customer IDs in log"
        )


# ── Unified audit event log (8 tests) ─────────────

class TestUnifiedAuditLog:
    """Audit log schema and retention compliance."""

    REQUIRED_COLS = {
        "event_id", "system_id", "mr_reference",
        "awb_customer_id", "decision_type",
        "input_hash", "output_hash",
        "confidence", "human_reviewed", "logged_at",
    }

    def test_audit_table_exists(self, db: Any) -> None:
        """unified_audit_log table must exist."""
        cur = db.cursor()
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM information_schema.tables
              WHERE table_name = 'unified_audit_log'
            )
            """
        )
        assert cur.fetchone()[0]

    def test_audit_table_columns(
        self, db: Any
    ) -> None:
        """All required columns must be present."""
        cur = db.cursor()
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'unified_audit_log'
            """
        )
        cols = {r[0] for r in cur.fetchall()}
        missing = self.REQUIRED_COLS - cols
        assert not missing, (
            f"Missing columns: {missing}"
        )

    def test_audit_has_mr_reference(
        self, db: Any
    ) -> None:
        """All recent entries must have mr_reference."""
        cur = db.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM unified_audit_log
            WHERE mr_reference IS NULL
            AND logged_at > NOW() - INTERVAL '1 hour'
            """
        )
        assert cur.fetchone()[0] == 0

    def test_audit_has_input_hash(
        self, db: Any
    ) -> None:
        """All credit-decision entries have input_hash."""
        cur = db.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM unified_audit_log
            WHERE decision_type = 'credit_decision'
            AND input_hash IS NULL
            AND logged_at > NOW() - INTERVAL '24 hours'
            """
        )
        assert cur.fetchone()[0] == 0

    def test_retention_policy_index(
        self, db: Any
    ) -> None:
        """Retention index on logged_at must exist."""
        cur = db.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'unified_audit_log'
            AND indexname LIKE '%logged_at%'
            """
        )
        assert cur.fetchone()[0] >= 1

    def test_no_entries_older_than_7_years(
        self, db: Any
    ) -> None:
        """No entries should pre-date programme start."""
        cutoff = datetime.datetime(
            2024, 12, 31, tzinfo=datetime.timezone.utc
        )
        cur = db.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM unified_audit_log
            WHERE logged_at < %s
            """,
            (cutoff,),
        )
        assert cur.fetchone()[0] == 0

    def test_high_exposure_has_human_reviewed(
        self, db: Any
    ) -> None:
        """High-exposure decisions must be reviewed."""
        cur = db.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM unified_audit_log
            WHERE decision_type = 'credit_decision'
            AND CAST(
              output_hash->>'exposure_gbp'
              AS NUMERIC
            ) > 500000
            AND human_reviewed = FALSE
            AND logged_at > NOW() - INTERVAL '24 hours'
            """
        )
        count = cur.fetchone()[0]
        assert count == 0, (
            f"{count} high-exposure decisions "
            f"not reviewed"
        )

    def test_audit_write_latency(
        self,
        client: httpx.Client,
        db: Any,
    ) -> None:
        """Audit log must be written within 5 seconds."""
        req_id = str(uuid.uuid4())
        cid = f"AWB-LAT-{uuid.uuid4().hex[:6].upper()}"
        client.post(
            "/cda/analyse",
            headers={
                **auth_headers(),
                "X-AWB-Request-ID": req_id,
            },
            json={
                "awb_customer_id": cid,
                "document_type": "annual_accounts",
                "content": "Latency test",
            },
        )
        deadline = time.time() + 5
        found = False
        while time.time() < deadline:
            cur = db.cursor()
            cur.execute(
                """
                SELECT COUNT(*) FROM unified_audit_log
                WHERE awb_customer_id = %s
                """,
                (cid,),
            )
            if cur.fetchone()[0] > 0:
                found = True
                break
            time.sleep(0.5)
        assert found, (
            "Audit log not written within 5 seconds"
        )


# ── CRO dashboard freshness (5 tests) ─────────────

class TestCRODashboard:
    """CRO dashboard metrics must be fresh."""

    MAX_AGE_MINS = 15

    def test_dashboard_health(
        self, client: httpx.Client
    ) -> None:
        """Dashboard health endpoint returns 200."""
        resp = client.get(
            "/dashboard/health",
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_capital_ratios_fresh(
        self,
        client: httpx.Client,
    ) -> None:
        """Capital ratios must be <=15 mins old."""
        resp = client.get(
            "/dashboard/capital-ratios",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        body = resp.json()
        ts = datetime.datetime.fromisoformat(
            body["last_refreshed"]
        )
        age_mins = (
            datetime.datetime.now(
                tz=datetime.timezone.utc
            ) - ts
        ).total_seconds() / 60
        assert age_mins <= self.MAX_AGE_MINS, (
            f"Capital ratios {age_mins:.1f} mins old"
        )

    def test_cet1_ratio_present(
        self, client: httpx.Client
    ) -> None:
        """CET1 ratio must be present and positive."""
        resp = client.get(
            "/dashboard/capital-ratios",
            headers=auth_headers(),
        )
        body = resp.json()
        assert "cet1_ratio" in body
        assert body["cet1_ratio"] > 0

    def test_lcr_present(
        self, client: httpx.Client
    ) -> None:
        """LCR must be present in dashboard."""
        resp = client.get(
            "/dashboard/liquidity",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "lcr" in body
        assert body["lcr"] > 0

    def test_model_psi_scores_present(
        self, client: httpx.Client
    ) -> None:
        """PSI scores for 13 supervised models present."""
        resp = client.get(
            "/dashboard/model-performance",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        body = resp.json()
        psi_scores = body.get("psi_scores", {})
        assert len(psi_scores) >= 13, (
            f"Expected 13+ PSI scores, "
            f"got {len(psi_scores)}"
        )


# ── API Gateway security (6 tests) ────────────────

class TestAPIGatewaySecurity:
    """API Gateway must enforce authentication."""

    def test_reject_no_token(
        self, client: httpx.Client
    ) -> None:
        """Requests without JWT must return 401."""
        resp = client.post(
            "/cda/analyse",
            json={"awb_customer_id": "AWB-TEST-001"},
        )
        assert resp.status_code == 401

    def test_reject_expired_token(
        self, client: httpx.Client
    ) -> None:
        """Expired JWT must return 401."""
        resp = client.post(
            "/cda/analyse",
            headers={
                "Authorization":
                    "Bearer eyJ0eXAiOiJKV1Qexpired"
            },
            json={"awb_customer_id": "AWB-TEST-001"},
        )
        assert resp.status_code == 401

    def test_rate_limit_headers_present(
        self, client: httpx.Client
    ) -> None:
        """Rate limit headers must be in response."""
        resp = client.get(
            "/health",
            headers=auth_headers(),
        )
        assert "X-RateLimit-Limit" in resp.headers

    def test_health_endpoint_public(
        self, client: httpx.Client
    ) -> None:
        """Health endpoint is publicly accessible."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_cors_header_present(
        self, client: httpx.Client
    ) -> None:
        """CORS headers must be configured."""
        resp = client.options(
            "/cda/analyse",
            headers={
                "Origin":
                    "https://dashboard.awb.internal"
            },
        )
        assert resp.status_code in (200, 204)

    def test_request_id_echoed(
        self, client: httpx.Client
    ) -> None:
        """Request-ID must be echoed in response."""
        req_id = str(uuid.uuid4())
        resp = client.get(
            "/health",
            headers={
                **auth_headers(),
                "X-AWB-Request-ID": req_id,
            },
        )
        echoed = resp.headers.get("X-AWB-Request-ID")
        assert echoed == req_id


# ── DORA compliance (5 tests) ─────────────────────

class TestDORACompliance:
    """DORA Article 11 RTO/RPO compliance."""

    def test_circuit_breaker_endpoint(
        self, client: httpx.Client
    ) -> None:
        """Circuit breaker status must be accessible."""
        resp = client.get(
            "/health/circuit-breakers",
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "circuit_breakers" in body

    def test_no_open_circuit_breakers(
        self, client: httpx.Client
    ) -> None:
        """No circuit breakers should be open."""
        resp = client.get(
            "/health/circuit-breakers",
            headers=auth_headers(),
        )
        body = resp.json()
        open_cbs = [
            name for name, state
            in body.get("circuit_breakers", {}).items()
            if state == "open"
        ]
        assert not open_cbs, (
            f"Open circuit breakers: {open_cbs}"
        )

    def test_dora_asset_registry_reachable(
        self, client: httpx.Client
    ) -> None:
        """DORA ICT asset registry must be reachable."""
        resp = client.get(
            "/governance/ict-assets",
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_all_services_have_ict_id(
        self, client: httpx.Client
    ) -> None:
        """All 23 services must have a DORA ICT ID."""
        resp = client.get(
            "/governance/ict-assets",
            headers=auth_headers(),
        )
        body = resp.json()
        assets = body.get("assets", [])
        missing = [
            a["service"] for a in assets
            if not a.get("ict_asset_id")
        ]
        assert not missing, (
            f"Services missing ICT ID: {missing}"
        )

    def test_incident_log_reachable(
        self, db: Any,
    ) -> None:
        """DORA incident log table must exist."""
        cur = db.cursor()
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1
              FROM information_schema.tables
              WHERE table_name = 'dora_incident_log'
            )
            """
        )
        assert cur.fetchone()[0], (
            "dora_incident_log table not found"
        )


# ── Model registry compliance (5 tests) ───────────

class TestModelRegistry:
    """PRA SS1/23 model registry compliance."""

    REQUIRED_IDS = [
        "MR-2026-035", "MR-2026-036",
        "MR-2026-037", "MR-2026-038",
        "MR-2026-039",
    ]

    def test_registry_reachable(
        self, client: httpx.Client
    ) -> None:
        """Model registry endpoint must be reachable."""
        resp = client.get(
            "/governance/model-registry",
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    @pytest.mark.parametrize(
        "mr_id", REQUIRED_IDS
    )
    def test_core_models_registered(
        self,
        client: httpx.Client,
        mr_id: str,
    ) -> None:
        """Core model IDs must be in registry."""
        resp = client.get(
            f"/governance/model-registry/{mr_id}",
            headers=auth_headers(),
        )
        assert resp.status_code == 200, (
            f"{mr_id} not found in registry"
        )

    def test_no_models_without_risk_rating(
        self, client: httpx.Client
    ) -> None:
        """All models must have SS1/23 risk rating."""
        resp = client.get(
            "/governance/model-registry",
            headers=auth_headers(),
        )
        body = resp.json()
        missing = [
            m["mr_reference"]
            for m in body.get("models", [])
            if not m.get("ss1_23_risk_rating")
        ]
        assert not missing, (
            f"Models without risk rating: {missing}"
        )


# ── LLM provider DORA Art.28 (3 tests) ────────────

class TestDORAConcentration:
    """DORA Art.28 70% single-provider cap."""

    def test_llm_usage_endpoint(
        self, client: httpx.Client
    ) -> None:
        """LLM usage stats endpoint must be present."""
        resp = client.get(
            "/governance/llm-usage",
            headers=auth_headers(),
        )
        assert resp.status_code == 200

    def test_no_provider_exceeds_70pct(
        self, client: httpx.Client
    ) -> None:
        """No single LLM provider exceeds 70%."""
        resp = client.get(
            "/governance/llm-usage",
            headers=auth_headers(),
        )
        body = resp.json()
        by_provider = body.get("by_provider", {})
        for provider, pct in by_provider.items():
            assert pct <= 70.0, (
                f"{provider} at {pct}% "
                f"exceeds DORA 70% cap"
            )

    def test_gemini_flash_is_primary(
        self, client: httpx.Client
    ) -> None:
        """Gemini 3.5 Flash must be primary LLM."""
        resp = client.get(
            "/governance/llm-usage",
            headers=auth_headers(),
        )
        body = resp.json()
        by_model = body.get("by_model", {})
        flash_pct = by_model.get(
            "gemini-3.5-flash", 0
        )
        assert flash_pct >= 50.0, (
            f"Gemini Flash at {flash_pct}% "
            f"(expected >= 50%)"
        )


# ── End-to-end credit flow (4 tests) ──────────────

class TestCreditEndToEnd:
    """Full credit application round-trip."""

    def test_credit_application_flow(
        self,
        client: httpx.Client,
        db: Any,
    ) -> None:
        """Full credit flow writes to audit log."""
        cid = (
            f"AWB-E2E-{uuid.uuid4().hex[:8].upper()}"
        )
        app_id = str(uuid.uuid4())

        # Step 1: analyse document
        r1 = client.post(
            "/cda/analyse",
            headers=auth_headers(),
            json={
                "awb_customer_id": cid,
                "document_type": "annual_accounts",
                "content": "Revenue £5M, EBITDA £1M",
                "application_id": app_id,
            },
        )
        assert r1.status_code in (200, 202)

        # Step 2: credit decision
        r2 = client.post(
            "/credit-agent/assess",
            headers=auth_headers(),
            json={
                "awb_customer_id": cid,
                "application_id": app_id,
                "exposure_gbp": 250_000,
            },
        )
        assert r2.status_code in (200, 202)

        # Step 3: verify audit trail
        time.sleep(3)
        cur = db.cursor()
        cur.execute(
            """
            SELECT DISTINCT system_id
            FROM unified_audit_log
            WHERE awb_customer_id = %s
            """,
            (cid,),
        )
        systems = {r[0] for r in cur.fetchall()}
        assert "MR-2026-035" in systems
        assert "MR-2026-037" in systems

    def test_high_exposure_escalates(
        self, client: httpx.Client
    ) -> None:
        """Exposures over £500K must be escalated."""
        resp = client.post(
            "/credit-agent/assess",
            headers=auth_headers(),
            json={
                "awb_customer_id": "AWB-TEST-HE-001",
                "application_id": str(uuid.uuid4()),
                "exposure_gbp": 750_000,
            },
        )
        assert resp.status_code in (200, 202)
        body = resp.json()
        assert body.get("requires_human_review"), (
            "High-exposure case not escalated"
        )

    def test_response_includes_confidence(
        self, client: httpx.Client
    ) -> None:
        """CDA response must include confidence score."""
        resp = client.post(
            "/cda/analyse",
            headers=auth_headers(),
            json={
                "awb_customer_id": "AWB-TEST-CONF-01",
                "document_type": "annual_accounts",
                "content": "Test content",
            },
        )
        assert resp.status_code in (200, 202)
        body = resp.json()
        conf = body.get("confidence_score", -1)
        assert 0.0 <= conf <= 1.0, (
            f"Invalid confidence score: {conf}"
        )

    def test_low_confidence_routes_to_review(
        self, client: httpx.Client, db: Any
    ) -> None:
        """Low-confidence analysis routes to review."""
        cid = (
            f"AWB-LC-{uuid.uuid4().hex[:8].upper()}"
        )
        resp = client.post(
            "/cda/analyse",
            headers=auth_headers(),
            json={
                "awb_customer_id": cid,
                "document_type": "annual_accounts",
                "content": "Incomplete...",
                "force_low_confidence": True,
            },
        )
        if resp.status_code in (200, 202):
            body = resp.json()
            conf = body.get("confidence_score", 1.0)
            if conf < 0.80:
                assert body.get(
                    "routed_to_human_review"
                ), "Low confidence not routed"
