"""awb_commons/t24_client.py
Temenos T24 integration client — three approved patterns.
READ: Oracle read replica (5-min lag, read-only IAM role).
WRITE: UUID idempotency key + compensating transaction.
EVENT: Kafka consumer base (see kafka_consumer.py).
PRA SS1/23 | DORA Art.9 change management
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class T24WriteResult:
    """Outcome of a T24 WRITE operation."""
    idempotency_key: str
    t24_reference: str
    success: bool
    was_duplicate: bool = False
    compensated: bool = False


class T24IdempotencyStore:
    """PostgreSQL-backed idempotency key store.

    Prevents double-posting to T24 — war story: Ch 6 double-
    posting incident cost £62K per occurrence (3×/yr).
    In production, backed by awb_commons DB client.
    """

    def __init__(self, db_client: Any) -> None:
        self._db = db_client

    def check_and_reserve(
        self, key: str, operation: str
    ) -> tuple[bool, str | None]:
        """Check if key exists; reserve if not.

        Returns:
            (is_duplicate, existing_t24_ref)
        """
        row = self._db.fetchone(
            "SELECT t24_reference, completed_at "
            "FROM t24_idempotency_log "
            "WHERE idempotency_key = %s",
            (key,),
        )
        if row:
            return True, row["t24_reference"]
        self._db.execute(
            "INSERT INTO t24_idempotency_log"
            " (idempotency_key, operation, reserved_at)"
            " VALUES (%s, %s, %s)",
            (key, operation, datetime.now(timezone.utc)),
        )
        return False, None

    def mark_complete(
        self, key: str, t24_reference: str
    ) -> None:
        self._db.execute(
            "UPDATE t24_idempotency_log "
            "SET t24_reference = %s, completed_at = %s "
            "WHERE idempotency_key = %s",
            (t24_reference, datetime.now(timezone.utc), key),
        )

    def mark_failed(self, key: str) -> None:
        self._db.execute(
            "DELETE FROM t24_idempotency_log "
            "WHERE idempotency_key = %s",
            (key,),
        )


class T24WriteClient:
    """AWB T24 WRITE pattern — idempotent credit writes.

    Args:
        t24_api:    T24 REST API client (or stub in tests).
        id_store:   T24IdempotencyStore instance.
        circuit:    CircuitBreaker wrapping T24 API calls.
    """

    def __init__(
        self,
        t24_api: Any,
        id_store: T24IdempotencyStore,
        circuit: Any,
    ) -> None:
        self._api = t24_api
        self._store = id_store
        self._circuit = circuit

    def write_credit_facility(
        self,
        customer_id: str,
        facility_data: dict,
        idempotency_key: str | None = None,
    ) -> T24WriteResult:
        """Write a credit facility to T24 with idempotency.

        Args:
            customer_id:      AWB customer identifier.
            facility_data:    Facility attributes dict.
            idempotency_key:  Caller-supplied key; generated if None.

        Returns:
            T24WriteResult with outcome details.

        Raises:
            RuntimeError: On T24 API failure after compensation.
        """
        key = idempotency_key or str(uuid.uuid4())
        is_dup, existing_ref = self._store.check_and_reserve(
            key, "credit_facility_write"
        )
        if is_dup:
            logger.info(
                "t24_duplicate_write_skipped",
                extra={
                    "key": key,
                    "customer_id": customer_id,
                    "t24_ref": existing_ref,
                },
            )
            return T24WriteResult(
                idempotency_key=key,
                t24_reference=existing_ref or "",
                success=True,
                was_duplicate=True,
            )

        try:
            t24_ref = self._circuit.call(
                self._api.post_credit_facility,
                customer_id=customer_id,
                data=facility_data,
            )
            self._store.mark_complete(key, t24_ref)
            logger.info(
                "t24_write_success",
                extra={
                    "key": key,
                    "customer_id": customer_id,
                    "t24_ref": t24_ref,
                },
            )
            return T24WriteResult(
                idempotency_key=key,
                t24_reference=t24_ref,
                success=True,
            )
        except Exception as exc:
            logger.error(
                "t24_write_failed_compensating",
                extra={
                    "key": key,
                    "customer_id": customer_id,
                    "error": str(exc),
                },
            )
            self._store.mark_failed(key)
            self._compensate(customer_id, facility_data, key)
            return T24WriteResult(
                idempotency_key=key,
                t24_reference="",
                success=False,
                compensated=True,
            )

    def _compensate(
        self,
        customer_id: str,
        facility_data: dict,
        key: str,
    ) -> None:
        """Execute compensating transaction on T24 write failure."""
        try:
            self._api.post_compensation_event(
                customer_id=customer_id,
                original_data=facility_data,
                idempotency_key=key,
            )
            logger.info(
                "t24_compensation_posted",
                extra={"key": key, "customer_id": customer_id},
            )
        except Exception as comp_exc:
            logger.critical(
                "t24_compensation_failed",
                extra={
                    "key": key,
                    "customer_id": customer_id,
                    "error": str(comp_exc),
                },
            )
