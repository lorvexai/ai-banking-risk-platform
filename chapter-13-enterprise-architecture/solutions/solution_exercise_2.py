"""solutions/solution_exercise_2.py
Exercise 13.2 — Reference solution.
Do not read before attempting the exercise!

T24 idempotent write with circuit breaker.
"""
import uuid

from awb_commons.circuit_breaker import CircuitBreaker
from awb_commons.t24_client import T24WriteResult

# Import stubs from exercise module
from exercises.exercise_2 import (
    StubIdempotencyStore,
    StubT24API,
)


def write_credit_facility_safe(
    customer_id: str,
    facility_data: dict,
    t24_api: StubT24API,
    id_store: StubIdempotencyStore,
    circuit: CircuitBreaker,
    idempotency_key: str | None = None,
) -> T24WriteResult:
    """Write a credit facility to T24 with full resilience.

    Implements: idempotency check, circuit breaker,
    compensating transaction on failure.
    """
    key = idempotency_key or str(uuid.uuid4())

    # Step 1: Check and reserve idempotency key
    is_dup, existing_ref = id_store.check_and_reserve(
        key, "credit_facility"
    )
    if is_dup:
        return T24WriteResult(
            idempotency_key=key,
            t24_reference=existing_ref or "",
            success=True,
            was_duplicate=True,
        )

    # Step 2: Call T24 through the circuit breaker
    try:
        t24_ref = circuit.call(
            t24_api.post_credit_facility,
            customer_id=customer_id,
            data=facility_data,
        )
        id_store.mark_complete(key, t24_ref)
        return T24WriteResult(
            idempotency_key=key,
            t24_reference=t24_ref,
            success=True,
        )
    except Exception:
        id_store.mark_failed(key)
        _compensate(customer_id, facility_data, key, t24_api)
        return T24WriteResult(
            idempotency_key=key,
            t24_reference="",
            success=False,
            compensated=True,
        )


def _compensate(
    customer_id: str,
    facility_data: dict,
    key: str,
    t24_api: StubT24API,
) -> None:
    """Post a compensating transaction to T24."""
    t24_api.post_compensation_event(
        customer_id=customer_id,
        original_data=facility_data,
        idempotency_key=key,
    )
