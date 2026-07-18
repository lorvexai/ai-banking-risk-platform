"""tests/test_exercise_2.py
Tests for Exercise 13.2 — T24 idempotent write.
Run: pytest tests/test_exercise_2.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from awb_commons.circuit_breaker import CircuitBreaker, CircuitState
from exercises.exercise_2 import (
    StubIdempotencyStore,
    StubT24API,
    make_circuit,
)

# Import the student's implementation (falls back to solution if needed)
try:
    from exercises.exercise_2 import write_credit_facility_safe
    # Check it's actually implemented
    import uuid
    _test = write_credit_facility_safe(
        "TEST", {}, StubT24API(), StubIdempotencyStore(),
        make_circuit(), str(uuid.uuid4())
    )
except NotImplementedError:
    from solutions.solution_exercise_2 import (
        write_credit_facility_safe,
    )


class TestDuplicateCallSkipped:
    """Requirement: same idempotency key → skip T24 API call."""

    def test_duplicate_call_skipped(self):
        api = StubT24API()
        store = StubIdempotencyStore()
        circuit = make_circuit()
        key = "idem-key-001"

        r1 = write_credit_facility_safe(
            "CUST-A", {"amount": 50000},
            api, store, circuit, key
        )
        r2 = write_credit_facility_safe(
            "CUST-A", {"amount": 50000},
            api, store, circuit, key
        )

        assert r1.success is True
        assert r1.was_duplicate is False
        assert r2.was_duplicate is True
        # T24 API called exactly once — not twice
        assert api._calls == 1


class TestCircuitOpensAfterFailures:
    """Requirement: repeated failures open the circuit breaker."""

    def test_circuit_opens_after_failures(self):
        # Stub always fails
        api = StubT24API(fail_times=10)
        store = StubIdempotencyStore()
        circuit = CircuitBreaker(
            failure_threshold=0.5,
            window_seconds=60,
            recovery_timeout=3600,
        )

        # 10 separate calls with unique keys
        for i in range(10):
            write_credit_facility_safe(
                f"CUST-{i}", {},
                api, store, circuit, f"key-fail-{i}"
            )

        assert circuit.state == CircuitState.OPEN


class TestCompensatingTransactionOnFailure:
    """Requirement: T24 failure triggers compensation."""

    def test_compensating_transaction_on_failure(self):
        api = StubT24API(fail_times=1)
        store = StubIdempotencyStore()
        circuit = make_circuit(failure_threshold=0.99)
        key = "key-compensate-001"

        result = write_credit_facility_safe(
            "CUST-COMP", {"amount": 75000},
            api, store, circuit, key
        )

        assert result.success is False
        assert result.compensated is True
        assert len(api.compensation_calls) == 1
        assert api.compensation_calls[0]["key"] == key
