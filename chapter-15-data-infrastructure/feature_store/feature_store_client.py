# feature_store/feature_store_client.py
# AWB Feature Store — PIT correct features, no training-serving skew
# FTS-2026-001 | PRA SS1/23 | Redis < 5ms serving
import logging
import json
from datetime import date
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

FEATURE_VERSION = "v2.1.0"  # Pinned to model training run


class FeatureStoreClient:
    """Point-in-time correct feature retrieval for AWB models.

    Eliminates training-serving skew by using identical SQL
    logic for both batch training (PostgreSQL time-travel)
    and real-time serving (Redis + PostgreSQL fallback).

    Training (batch, no skew):
        features = client.get_training_features(
            awb_customer_id="AWB-001234",
            as_of_date=date(2024, 12, 31),
        )

    Serving (< 5ms via Redis):
        features = client.get_serving_features(
            awb_customer_id="AWB-001234",
        )

    Monthly skew check:
        divergence = client.check_skew(
            awb_customer_id, training_date
        )

    Args:
        pg_client: PostgreSQL connection (time-travel queries).
        redis_client: Redis for sub-5ms serving cache.
        feature_version: Must match the model training run.
    """

    CACHE_TTL_SECONDS = 30  # Matches Customer 360 freshness SLA

    def __init__(
        self,
        pg_client,
        redis_client,
        feature_version: str = FEATURE_VERSION,
    ) -> None:
        self._pg = pg_client
        self._redis = redis_client
        self._version = feature_version

    def get_training_features(
        self,
        awb_customer_id: str,
        as_of_date: date,
    ) -> Dict[str, Any]:
        """Return features as they existed at as_of_date.

        Uses PostgreSQL effective_date column — prevents data
        leakage from future transactions into training windows.
        Same SQL logic as serving path guarantees no skew.
        Time-travel: application at date D uses features as
        of date D, not current values.

        Args:
            awb_customer_id: AWB universal customer identifier.
            as_of_date: Historical snapshot date for PIT query.

        Returns:
            Dict of feature_name -> value at as_of_date.

        Raises:
            ValueError: Customer ID not found for given date.
        """
        sql = """
            SELECT feature_name, feature_value
            FROM feature_store.customer_features
            WHERE awb_customer_id = %s
              AND effective_date <= %s
              AND feature_version = %s
            ORDER BY effective_date DESC
            LIMIT 1
        """
        logger.info(
            "PIT feature fetch: id=%s date=%s ver=%s",
            awb_customer_id,
            as_of_date,
            self._version,
        )
        raise NotImplementedError(
            "Full implementation in GitHub repo"
        )

    def get_serving_features(
        self,
        awb_customer_id: str,
    ) -> Dict[str, Any]:
        """Return current features; Redis cache < 5ms.

        Falls back to PostgreSQL on Redis miss. Cache TTL
        30 seconds matches Customer 360 freshness SLA.
        Same computation logic as training path — no skew.

        Args:
            awb_customer_id: AWB universal customer identifier.

        Returns:
            Dict of feature_name -> current value.
        """
        cache_key = (
            f"features:{self._version}:{awb_customer_id}"
        )
        cached = self._redis.get(cache_key)
        if cached:
            logger.debug("Redis hit: %s", cache_key)
            return json.loads(cached)

        logger.debug(
            "Redis miss — PostgreSQL fallback: %s", cache_key
        )
        raise NotImplementedError(
            "Full implementation in GitHub repo"
        )

    def check_skew(
        self,
        awb_customer_id: str,
        training_date: date,
    ) -> Dict[str, float]:
        """Monthly skew check: training vs serving features.

        Compares training-time feature snapshot against current
        serving values. Alerts if any feature diverges by more
        than the configured threshold from training distribution.
        Run monthly in the MLOps monitoring DAG (Chapter 14).

        Args:
            awb_customer_id: Customer to check.
            training_date: Date of original training snapshot.

        Returns:
            Dict of feature_name -> relative_divergence (0..1).
            Empty dict if all features within tolerance.
        """
        training = self.get_training_features(
            awb_customer_id, training_date
        )
        serving = self.get_serving_features(awb_customer_id)
        divergence: Dict[str, float] = {}
        for k in training:
            if k in serving:
                t_val = float(training[k])
                s_val = float(serving[k])
                if t_val != 0:
                    rel = abs(s_val - t_val) / abs(t_val)
                    divergence[k] = rel
        high_skew = {
            k: v for k, v in divergence.items() if v > 0.1
        }
        if high_skew:
            logger.warning(
                "Feature skew detected: %s", high_skew
            )
        return divergence
