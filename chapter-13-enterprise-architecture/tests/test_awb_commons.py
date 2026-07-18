"""tests/test_awb_commons.py
Unit tests — awb_commons core components.
Chapter 13: Enterprise AI Architecture for Risk and Compliance.
All tests use stubs — no live API keys or DB required.
Run: pytest tests/test_awb_commons.py -v
"""
import os
import sys
import time
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── circuit_breaker ────────────────────────────────────────────────

from awb_commons.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


class TestCircuitBreaker:
    def test_closed_by_default(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_successful_call_returns_value(self):
        cb = CircuitBreaker()
        result = cb.call(lambda: 42)
        assert result == 42

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker(
            failure_threshold=0.5,
            window_seconds=60,
            recovery_timeout=3600,
        )
        def fail():
            raise RuntimeError("service down")

        # Drive 5 calls, all failures → rate = 1.0 > 0.5
        for _ in range(5):
            with pytest.raises(RuntimeError):
                cb.call(fail)

        assert cb.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        cb = CircuitBreaker(
            failure_threshold=0.1,
            window_seconds=60,
            recovery_timeout=3600,
        )
        for _ in range(10):
            with pytest.raises((RuntimeError, CircuitOpenError)):
                cb.call(lambda: (_ for _ in ()).throw(
                    RuntimeError("fail")))

        # Force open
        cb._state = CircuitState.OPEN
        cb._opened_at = time.time()
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: None)

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(recovery_timeout=0)
        cb._state = CircuitState.OPEN
        cb._opened_at = time.time() - 1   # already expired
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED


# ── t24_client ────────────────────────────────────────────────────

from awb_commons.t24_client import (
    T24IdempotencyStore,
    T24WriteClient,
    T24WriteResult,
)


class _MockDB:
    """Minimal in-memory DB for T24IdempotencyStore tests."""
    def __init__(self):
        self._rows: dict[str, dict] = {}

    def fetchone(self, query, params=()):
        key = params[0] if params else None
        return self._rows.get(key)

    def execute(self, query, params=()):
        if "INSERT" in query:
            self._rows[params[0]] = {
                "t24_reference": None,
                "completed_at": None,
            }
        elif "UPDATE" in query:
            if params[-1] in self._rows:
                self._rows[params[-1]]["t24_reference"] = params[0]
        elif "DELETE" in query:
            self._rows.pop(params[0], None)


class _StubT24API:
    def __init__(self, fail=False):
        self._fail = fail
        self.compensation_calls = []

    def post_credit_facility(self, customer_id, data):
        if self._fail:
            raise ConnectionError("T24 unavailable")
        return f"T24-REF-{customer_id}"

    def post_compensation_event(
        self, customer_id, original_data, idempotency_key
    ):
        self.compensation_calls.append(idempotency_key)


class TestT24WriteClient:
    def _make_client(self, fail=False):
        db = _MockDB()
        store = T24IdempotencyStore(db)
        api = _StubT24API(fail=fail)
        circuit = CircuitBreaker(
            failure_threshold=0.9, recovery_timeout=3600
        )
        return T24WriteClient(api, store, circuit), api

    def test_successful_write_returns_t24_ref(self):
        client, _ = self._make_client()
        result = client.write_credit_facility(
            "CUST-001", {"amount": 50000}, "key-1"
        )
        assert result.success is True
        assert result.t24_reference == "T24-REF-CUST-001"
        assert result.was_duplicate is False

    def test_duplicate_key_skipped(self):
        client, _ = self._make_client()
        key = str(uuid.uuid4())
        r1 = client.write_credit_facility("CUST-002", {}, key)
        r2 = client.write_credit_facility("CUST-002", {}, key)
        assert r1.success is True
        assert r2.was_duplicate is True

    def test_failure_triggers_compensation(self):
        client, api = self._make_client(fail=True)
        result = client.write_credit_facility(
            "CUST-003", {"amount": 100000}, "key-fail"
        )
        assert result.success is False
        assert result.compensated is True
        assert len(api.compensation_calls) == 1


# ── logging_client ────────────────────────────────────────────────

import json
import logging

from awb_commons.logging_client import (
    StructuredFormatter,
    get_structured_logger,
)


class TestStructuredFormatter:
    def _make_record(self, msg="test", level=logging.INFO):
        record = logging.LogRecord(
            name="test", level=level, pathname="", lineno=0,
            msg=msg, args=(), exc_info=None,
        )
        return record

    def test_output_is_valid_json(self):
        fmt = StructuredFormatter("my-service", "1.0.0")
        output = fmt.format(self._make_record())
        parsed = json.loads(output)
        assert "timestamp" in parsed
        assert "service_name" in parsed
        assert "message" in parsed

    def test_mandatory_fields_present(self):
        fmt = StructuredFormatter("svc", "2.3.1")
        parsed = json.loads(fmt.format(self._make_record("hello")))
        for field in (
            "timestamp", "service_name", "service_version",
            "log_level", "correlation_id", "user_id",
            "event_type", "message",
        ):
            assert field in parsed, f"Missing field: {field}"

    def test_service_name_correct(self):
        fmt = StructuredFormatter("credit-doc-analyser", "1.4.2")
        parsed = json.loads(fmt.format(self._make_record()))
        assert parsed["service_name"] == "credit-doc-analyser"
        assert parsed["service_version"] == "1.4.2"


# ── kafka topics ──────────────────────────────────────────────────

from kafka.topics import AWB_TOPICS, TOPICS_BY_NAME, get_topic


class TestKafkaTopics:
    def test_eight_topics_defined(self):
        assert len(AWB_TOPICS) == 8

    def test_transactions_topic_exists(self):
        topic = get_topic("awb.transactions")
        assert topic.partitions == 12
        assert topic.replication_factor == 3

    def test_unknown_topic_raises(self):
        with pytest.raises(KeyError):
            get_topic("awb.unknown-topic")

    def test_all_topics_have_required_fields(self):
        for topic in AWB_TOPICS:
            assert topic.name.startswith("awb.")
            assert topic.partitions >= 2
            assert topic.replication_factor == 3
            assert topic.retention_hours > 0
