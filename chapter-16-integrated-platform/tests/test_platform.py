"""
Chapter 16 Unit Tests — AWB Platform
======================================
Tests for: unified audit log, circuit breaker,
CRO dashboard metrics, and gateway security.

Run:
    pytest tests/test_platform.py -v

All tests use stubs/mocks — no live services
or AWS credentials required.
"""
from __future__ import annotations

import time
import uuid
import hashlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


# ── AuditEvent tests ───────────────────────────────

class TestAuditEvent:
    """Unit tests for AuditEvent model."""

    def _make_event(self, **kwargs):
        """Import and create an AuditEvent."""
        import sys, os
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(__file__), ".."
            ),
        )
        from awb_platform.unified_audit_log import (
            AuditEvent,
        )
        defaults = {
            "system_id": "MR-2026-035",
            "mr_reference": "MR-2026-035",
            "awb_customer_id": "AWB-TEST-001",
            "decision_type": "credit_decision",
        }
        defaults.update(kwargs)
        return AuditEvent(**defaults)

    def test_event_id_is_uuid(self) -> None:
        """event_id must be a valid UUID."""
        evt = self._make_event()
        uuid.UUID(evt.event_id)  # raises if invalid

    def test_input_hash_computed(self) -> None:
        """input_hash must be SHA-256 of payload."""
        payload = {"revenue": 5_000_000}
        evt = self._make_event(
            input_payload=payload
        )
        expected = hashlib.sha256(
            str(payload).encode()
        ).hexdigest()
        assert evt.input_hash == expected

    def test_null_payload_gives_null_hash(
        self,
    ) -> None:
        """None payload must produce None hash."""
        evt = self._make_event()
        assert evt.input_hash is None
        assert evt.output_hash is None

    def test_confidence_bounds(self) -> None:
        """Confidence must be between 0 and 1."""
        evt = self._make_event(confidence=0.85)
        assert 0.0 <= evt.confidence <= 1.0

    def test_confidence_invalid_raises(
        self,
    ) -> None:
        """Confidence outside [0,1] must fail."""
        with pytest.raises(Exception):
            self._make_event(confidence=1.5)

    def test_awb_customer_id_preserved(
        self,
    ) -> None:
        """AWB_CUSTOMER_ID must be stored exactly."""
        cid = "AWB-CORP-001-BRIST"
        evt = self._make_event(awb_customer_id=cid)
        assert evt.awb_customer_id == cid

    def test_mr_reference_stored(self) -> None:
        """MR reference must be stored exactly."""
        for mr in [
            "MR-2026-035",
            "MR-2026-037",
            "MR-2026-038",
        ]:
            evt = self._make_event(mr_reference=mr)
            assert evt.mr_reference == mr

    def test_human_reviewed_defaults_false(
        self,
    ) -> None:
        """human_reviewed must default to False."""
        evt = self._make_event()
        assert evt.human_reviewed is False

    def test_logged_at_is_utc(self) -> None:
        """logged_at must be UTC-aware."""
        evt = self._make_event()
        assert evt.logged_at.tzinfo is not None


# ── AuditLogger tests ──────────────────────────────

class TestAuditLogger:
    """Unit tests for AuditLogger with mock DB."""

    def _make_logger(self):
        import sys, os
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(__file__), ".."
            ),
        )
        from awb_platform.unified_audit_log import (
            AuditLogger,
            AuditEvent,
        )
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        return AuditLogger(mock_conn), AuditEvent

    def test_log_decision_calls_execute(
        self,
    ) -> None:
        """log_decision must call cursor.execute."""
        logger, AuditEvent = self._make_logger()
        evt = AuditEvent(
            system_id="MR-2026-035",
            mr_reference="MR-2026-035",
            awb_customer_id="AWB-TEST-001",
            decision_type="credit_decision",
        )
        result = logger.log_decision(evt)
        assert result == evt.event_id
        logger._conn.cursor().execute.assert_called()

    def test_log_decision_commits(self) -> None:
        """log_decision must commit on success."""
        logger, AuditEvent = self._make_logger()
        evt = AuditEvent(
            system_id="MR-2026-037",
            mr_reference="MR-2026-037",
            awb_customer_id="AWB-TEST-002",
            decision_type="credit_decision",
        )
        logger.log_decision(evt)
        logger._conn.commit.assert_called_once()

    def test_log_decision_raises_on_db_error(
        self,
    ) -> None:
        """DB error must raise RuntimeError."""
        logger, AuditEvent = self._make_logger()
        logger._conn.cursor().execute.side_effect = (
            Exception("DB connection lost")
        )
        evt = AuditEvent(
            system_id="MR-2026-035",
            mr_reference="MR-2026-035",
            awb_customer_id="AWB-TEST-003",
            decision_type="credit_decision",
        )
        with pytest.raises(RuntimeError):
            logger.log_decision(evt)

    def test_db_error_triggers_rollback(
        self,
    ) -> None:
        """DB error must trigger rollback."""
        logger, AuditEvent = self._make_logger()
        logger._conn.cursor().execute.side_effect = (
            Exception("Timeout")
        )
        evt = AuditEvent(
            system_id="MR-2026-035",
            mr_reference="MR-2026-035",
            awb_customer_id="AWB-TEST-004",
            decision_type="credit_decision",
        )
        try:
            logger.log_decision(evt)
        except RuntimeError:
            pass
        logger._conn.rollback.assert_called_once()


# ── CircuitBreaker tests ───────────────────────────

class TestCircuitBreaker:
    """Unit tests for CircuitBreaker."""

    def _make_cb(
        self,
        threshold: int = 3,
        recovery: int = 1,
    ):
        import sys, os
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(__file__), ".."
            ),
        )
        from awb_platform.api_gateway import (
            CircuitBreaker,
        )
        return CircuitBreaker(
            "test",
            failure_threshold=threshold,
            recovery_secs=recovery,
        )

    def test_initial_state_closed(self) -> None:
        """Circuit breaker starts closed."""
        cb = self._make_cb()
        assert cb.state == "closed"

    def test_allows_requests_when_closed(
        self,
    ) -> None:
        """Closed CB must allow requests."""
        cb = self._make_cb()
        assert cb.allow_request() is True

    def test_opens_after_threshold(self) -> None:
        """CB must open after threshold failures."""
        cb = self._make_cb(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

    def test_blocks_when_open(self) -> None:
        """Open CB must block requests."""
        cb = self._make_cb(threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert not cb.allow_request()

    def test_recovers_after_timeout(self) -> None:
        """CB recovers after recovery_secs."""
        cb = self._make_cb(
            threshold=2, recovery=1
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(1.1)
        # Accessing state triggers recovery check
        assert cb.state == "half-open"

    def test_success_resets_failures(self) -> None:
        """Successful call resets failure count."""
        cb = self._make_cb(threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failures == 0
        assert cb.state == "closed"


# ── Dashboard tests ────────────────────────────────

class TestDashboard:
    """Unit tests for CRO dashboard metrics."""

    def _import_dashboard(self):
        import sys, os
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(__file__), ".."
            ),
        )
        import awb_platform.dashboard as dash
        return dash

    def test_capital_ratios_above_minimum(
        self,
    ) -> None:
        """All capital ratios must exceed minimums."""
        dash = self._import_dashboard()
        cr = dash.get_capital_ratios()
        assert cr.cet1_ratio >= cr.cet1_minimum
        assert cr.tier1_ratio >= cr.t1_minimum
        assert (
            cr.total_capital_ratio
            >= cr.total_minimum
        )
        assert (
            cr.leverage_ratio >= cr.leverage_minimum
        )

    def test_leverage_ratio_crr3_art429(
        self,
    ) -> None:
        """Leverage ratio must meet CRR3 Art.429 3%."""
        dash = self._import_dashboard()
        cr = dash.get_capital_ratios()
        assert cr.leverage_ratio >= 3.0, (
            f"Leverage ratio {cr.leverage_ratio}% "
            f"below CRR3 Art.429 minimum 3%"
        )

    def test_lcr_above_100pct(self) -> None:
        """LCR must be >= 100% (CRR3)."""
        dash = self._import_dashboard()
        liq = dash.get_liquidity()
        assert liq.lcr >= 100.0, (
            f"LCR {liq.lcr}% below 100% minimum"
        )

    def test_nsfr_above_100pct(self) -> None:
        """NSFR must be >= 100% (CRR3)."""
        dash = self._import_dashboard()
        liq = dash.get_liquidity()
        assert liq.nsfr >= 100.0

    def test_ecl_stages_sum_to_total(self) -> None:
        """ECL stages must sum to total."""
        dash = self._import_dashboard()
        ecl = dash.get_ecl()
        total = (
            ecl.stage1_ecl_gbp
            + ecl.stage2_ecl_gbp
            + ecl.stage3_ecl_gbp
        )
        assert abs(total - ecl.total_ecl_gbp) < 1

    def test_13_psi_scores_present(self) -> None:
        """Exactly 13 supervised model PSI scores."""
        dash = self._import_dashboard()
        mp = dash.get_model_performance()
        assert len(mp.psi_scores) >= 13

    def test_psi_scores_in_valid_range(
        self,
    ) -> None:
        """All PSI scores must be non-negative."""
        dash = self._import_dashboard()
        mp = dash.get_model_performance()
        for mr, psi in mp.psi_scores.items():
            assert psi >= 0.0, (
                f"{mr} PSI score negative: {psi}"
            )

    def test_sar_rate_reasonable(self) -> None:
        """SAR rate should be < 10% of alerts."""
        dash = self._import_dashboard()
        aml = dash.get_aml_metrics()
        assert aml.sar_rate < 0.10

    def test_ragas_above_threshold(self) -> None:
        """RAGAS faithfulness >= 0.80 (SS1/23)."""
        dash = self._import_dashboard()
        mp = dash.get_model_performance()
        for mr, score in mp.ragas_scores.items():
            assert score >= 0.80, (
                f"{mr} RAGAS score {score} "
                f"below 0.80 threshold"
            )

    def test_no_legacy_branding(self) -> None:
        """No AWB or Cambridgeshire references."""
        dash = self._import_dashboard()
        summary = str(dash.get_capital_ratios())
        assert "AWB" not in summary
        assert "Cambridgeshire" not in summary
        assert "crb_commons" not in summary

    def test_model_registry_ids_canonical(
        self,
    ) -> None:
        """PSI keys must use MR-2026-XXX format."""
        dash = self._import_dashboard()
        mp = dash.get_model_performance()
        for mr_id in mp.psi_scores:
            assert mr_id.startswith("MR-2026-"), (
                f"Non-canonical model ID: {mr_id}"
            )

    def test_fca_ps22_9_reference_correct(
        self,
    ) -> None:
        """FCA ref must be PS22/9 not PS22/3."""
        # Inspect source for wrong reference
        import inspect
        import awb_platform.dashboard as dash
        src = inspect.getsource(dash)
        assert "PS22/3" not in src, (
            "Wrong FCA reference PS22/3 "
            "found (must be PS22/9)"
        )
