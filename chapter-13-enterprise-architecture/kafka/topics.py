"""kafka/topics.py
AWB MSK Kafka — 8-topic event architecture.
4.2M transactions/month | 3-broker cluster | EU-West-2
DORA Art.9 change management | 7-day retention
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class KafkaTopic:
    name: str
    partitions: int
    replication_factor: int
    retention_hours: int
    description: str
    consumer_group: str
    chapter_owner: str


# ── AWB canonical 8-topic topology ───────────────────────────────
AWB_TOPICS: list[KafkaTopic] = [
    KafkaTopic(
        name="awb.transactions",
        partitions=12,
        replication_factor=3,
        retention_hours=168,   # 7 days
        description=(
            "T24 committed transactions via TCBP event framework. "
            "4.2M events/month. Partitioned by customer_account_id "
            "for ordered AML feature engineering."
        ),
        consumer_group="awb-aml-monitor",
        chapter_owner="Ch 12 AML",
    ),
    KafkaTopic(
        name="awb.credit-events",
        partitions=6,
        replication_factor=3,
        retention_hours=168,
        description=(
            "Credit decision lifecycle events from the LangGraph "
            "credit agent pipeline (MR-2026-037). Consumed by "
            "portfolio monitoring and early warning systems."
        ),
        consumer_group="awb-credit-monitor",
        chapter_owner="Ch 6 Credit Risk",
    ),
    KafkaTopic(
        name="awb.fx-rates",
        partitions=3,
        replication_factor=3,
        retention_hours=24,
        description=(
            "Reuters FX rate updates every 30 seconds. "
            "Consumed by Treasury Operations Agent for "
            "intraday FX exposure recalculation."
        ),
        consumer_group="awb-treasury-agent",
        chapter_owner="Ch 3 Treasury Agent",
    ),
    KafkaTopic(
        name="awb.market-data",
        partitions=6,
        replication_factor=3,
        retention_hours=72,
        description=(
            "Live equity and rates data for real-time VaR "
            "engine (MR-2026-046). Bloomberg/Reuters feed "
            "normalised to AWB schema."
        ),
        consumer_group="awb-var-engine",
        chapter_owner="Ch 7 Market Risk",
    ),
    KafkaTopic(
        name="awb.kyc-events",
        partitions=4,
        replication_factor=3,
        retention_hours=168,
        description=(
            "KYC status change events from the Digital Identity "
            "platform. Consumed by Credit Decision Agent to gate "
            "automated approvals on KYC clearance."
        ),
        consumer_group="awb-credit-agent",
        chapter_owner="Ch 12 KYC",
    ),
    KafkaTopic(
        name="awb.model-alerts",
        partitions=3,
        replication_factor=3,
        retention_hours=720,   # 30 days
        description=(
            "Model performance drift alerts from the MLOps "
            "monitoring platform. Triggers model revalidation "
            "workflow under PRA SS1/23 Section 4.3."
        ),
        consumer_group="awb-mlops-monitor",
        chapter_owner="Ch 14 MLOps",
    ),
    KafkaTopic(
        name="awb.audit-trail",
        partitions=6,
        replication_factor=3,
        retention_hours=168,
        description=(
            "AI decision audit events streamed to S3 Glacier "
            "for 7-year retention. FCA COBS 9 compliance. "
            "All credit, KYC, and AML decisions captured."
        ),
        consumer_group="awb-audit-archiver",
        chapter_owner="Ch 13 Infrastructure",
    ),
    KafkaTopic(
        name="awb.regulatory-filings",
        partitions=2,
        replication_factor=3,
        retention_hours=8760,   # 1 year
        description=(
            "Regulatory filing lifecycle events: COREP "
            "submission trigger, PRA API submission result, "
            "EBA XBRL validation outcome. Consumed by the "
            "MJRRP filing tracker."
        ),
        consumer_group="awb-regulatory-tracker",
        chapter_owner="Ch 11 Regulatory Reporting",
    ),
]

# Indexed by name for fast lookup
TOPICS_BY_NAME: dict[str, KafkaTopic] = {
    t.name: t for t in AWB_TOPICS
}


def get_topic(name: str) -> KafkaTopic:
    """Return KafkaTopic by name.

    Raises:
        KeyError: If topic name not in AWB topology.
    """
    if name not in TOPICS_BY_NAME:
        raise KeyError(
            f"Topic '{name}' not in AWB topology. "
            f"Approved topics: {list(TOPICS_BY_NAME)}"
        )
    return TOPICS_BY_NAME[name]


def total_partitions() -> int:
    """Return total partition count across all topics."""
    return sum(t.partitions for t in AWB_TOPICS)
