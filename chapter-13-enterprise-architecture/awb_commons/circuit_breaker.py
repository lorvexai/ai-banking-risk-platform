"""awb_commons/circuit_breaker.py
DORA Art.17 ICT resilience — circuit breaker pattern.
Prevents cascade failures across the AWB service mesh.
War story: connection exhaustion → chatbot outage (Ch 13).
"""
import logging
import threading
import time
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing — reject all calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(RuntimeError):
    """Raised when circuit is OPEN and call is rejected."""
    def __init__(self, func_name: str) -> None:
        super().__init__(
            f"Circuit OPEN — {func_name} call rejected. "
            "Service unavailable."
        )


class CircuitBreaker:
    """AWB circuit breaker — DORA Art.17 ICT resilience.

    Prevents cascade failures across the service mesh.
    War story root cause: no circuit breaker allowed
    retry storm to exhaust DB connections.

    Args:
        failure_threshold: Fraction of failures to open (0.0–1.0).
        window_seconds:    Rolling window for failure counting.
        recovery_timeout:  Seconds before HALF_OPEN attempt.
    """

    def __init__(
        self,
        failure_threshold: float = 0.5,
        window_seconds: int = 60,
        recovery_timeout: int = 30,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failures: list[float] = []
        self._calls: list[float] = []
        self._opened_at: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute func through the circuit breaker.

        Raises:
            CircuitOpenError: When circuit is OPEN.
            Exception: Re-raised from func on failure.
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.time() - self._opened_at
                if elapsed > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        "circuit_half_open",
                        extra={"func": func.__name__}
                    )
                else:
                    raise CircuitOpenError(func.__name__)

        try:
            result = func(*args, **kwargs)
            self._on_success(func.__name__)
            return result
        except Exception as exc:
            self._on_failure(func.__name__, exc)
            raise

    def _on_success(self, func_name: str) -> None:
        with self._lock:
            now = time.time()
            self._calls.append(now)
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failures = []
                self._calls = []
                logger.info(
                    "circuit_closed",
                    extra={"func": func_name}
                )
            self._prune(now)

    def _on_failure(
        self, func_name: str, exc: Exception
    ) -> None:
        with self._lock:
            now = time.time()
            self._failures.append(now)
            self._calls.append(now)
            self._prune(now)
            recent_calls = len(self._calls)
            recent_fails = len(self._failures)
            if recent_calls >= 5:
                rate = recent_fails / recent_calls
                if rate >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = now
                    logger.error(
                        "circuit_opened",
                        extra={
                            "func": func_name,
                            "failure_rate": round(rate, 2),
                            "error": str(exc),
                        }
                    )

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._failures = [t for t in self._failures if t > cutoff]
        self._calls = [t for t in self._calls if t > cutoff]
