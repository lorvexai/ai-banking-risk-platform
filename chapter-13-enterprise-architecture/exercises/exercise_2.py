"""exercises/exercise_2.py
Exercise 13.2: Implement the T24 Idempotent Write Pattern
               with Circuit Breaker.
Difficulty: ★★★★☆ | Estimated time: 45 minutes

Goal:
    Implement write_credit_facility_safe() that wraps T24 writes
    with idempotency keys and circuit-breaker protection.

Requirements:
    1. Generate a UUID4 idempotency_key if one is not supplied.
    2. Check a PostgreSQL idempotency table (via id_store) —
       return early if key already exists (duplicate call).
    3. Wrap the T24 API call in the provided CircuitBreaker instance.
    4. On failure, call _compensate() and mark the key as failed.

Success criteria:
    pytest tests/test_exercise_2.py — all 3 tests must pass:
        test_duplicate_call_skipped
        test_circuit_opens_after_failures
        test_compensating_transaction_on_failure

Stubs provided (do not modify):
    StubT24API      — controllable success/failure T24 API
    StubIdempotencyStore — in-memory idempotency store
    make_circuit    — returns a CircuitBreaker with low threshold

Solution: github.com/lorvenio/ai-banking-risk-platform/
          chapter_013/solutions/solution_exercise_2.py
"""
import uuid
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from awb_commons.circuit_breaker import CircuitBreaker
from awb_commons.t24_client import T24WriteResult


# ── Test stubs (provided — do not modify) ─────────────────────────

class StubT24API:
    """Controllable T24 API stub for testing."""

    def __init__(self, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self._calls = 0
        self.compensation_calls: list[dict] = []

    def post_credit_facility(
        self, customer_id: str, data: dict
    ) -> str:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise ConnectionError("T24 API unavailable (stub)")
        return f"T24-REF-{customer_id}-{self._calls:04d}"

    def post_compensation_event(
        self, customer_id: str, original_data: dict,
        idempotency_key: str
    ) -> None:
        self.compensation_calls.append({
            "customer_id": customer_id,
            "key": idempotency_key,
        })


class StubIdempotencyStore:
    """In-memory idempotency store for testing."""

    def __init__(self) -> None:
        self._store: dict[str, str | None] = {}

    def check_and_reserve(
        self, key: str, operation: str
    ) -> tuple[bool, str | None]:
        if key in self._store:
            return True, self._store[key]
        self._store[key] = None
        return False, None

    def mark_complete(self, key: str, t24_ref: str) -> None:
        self._store[key] = t24_ref

    def mark_failed(self, key: str) -> None:
        del self._store[key]


def make_circuit(failure_threshold: float = 0.5) -> CircuitBreaker:
    """Return a CircuitBreaker with short window for testing."""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        window_seconds=60,
        recovery_timeout=30,
    )


# ── YOUR IMPLEMENTATION BELOW ─────────────────────────────────────

def write_credit_facility_safe(
    customer_id: str,
    facility_data: dict,
    t24_api: StubT24API,
    id_store: StubIdempotencyStore,
    circuit: CircuitBreaker,
    idempotency_key: str | None = None,
) -> T24WriteResult:
    """Write a credit facility to T24 with idempotency and resilience.

    TODO: Implement this function so that all three tests pass.

    Steps:
        1. Generate key = idempotency_key or str(uuid.uuid4())
        2. Call id_store.check_and_reserve(key, "credit_facility")
           - If duplicate: return T24WriteResult(was_duplicate=True)
        3. Wrap t24_api.post_credit_facility() in circuit.call()
           - On success: call id_store.mark_complete(key, t24_ref)
           - On failure: call id_store.mark_failed(key)
                         call _compensate(...)
        4. Return T24WriteResult with appropriate fields.
    """
    # TODO: Replace this stub with your implementation.
    raise NotImplementedError(
        "Implement write_credit_facility_safe() — see docstring."
    )


def _compensate(
    customer_id: str,
    facility_data: dict,
    key: str,
    t24_api: StubT24API,
) -> None:
    """Execute a compensating transaction via the T24 API stub."""
    # TODO: Call t24_api.post_compensation_event() here.
    raise NotImplementedError("Implement _compensate() — see docstring.")
