# mlops/airflow_dags.py | AWB MLOps Airflow DAGs
# Chapter 14 | awb_commons namespace
# MR-2026-043 to -046, MR-2026-053 | PRA SS1/23
from __future__ import annotations
import logging
from datetime import datetime, timedelta
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
except ModuleNotFoundError:
    # Lightweight fallback for unit tests in environments without Airflow.
    _DAG_STACK = []

    class DAG:  # type: ignore[override]
        def __init__(
            self,
            dag_id: str,
            default_args=None,
            description: str = "",
            schedule_interval=None,
            start_date=None,
            catchup: bool = False,
            tags=None,
        ):
            self.dag_id = dag_id
            self.default_args = default_args or {}
            self.description = description
            self.schedule_interval = schedule_interval
            self.start_date = start_date
            self.catchup = catchup
            self.tags = tags or []
            self.tasks = []

        def __enter__(self):
            _DAG_STACK.append(self)
            return self

        def __exit__(self, exc_type, exc, tb):
            _DAG_STACK.pop()
            return False

    class PythonOperator:  # type: ignore[override]
        def __init__(self, task_id: str, python_callable):
            self.task_id = task_id
            self.python_callable = python_callable
            self.downstream_tasks = []
            if _DAG_STACK:
                _DAG_STACK[-1].tasks.append(self)

        def __rshift__(self, other):
            self.downstream_tasks.append(other)
            return other

    def days_ago(days: int):
        return datetime.now() - timedelta(days=days)

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "awb-mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def extract_t24_accounts(**ctx) -> int:
    """Extract active accounts from T24 PostgreSQL mirror.

    Returns:
        Row count extracted.
    Raises:
        RuntimeError: If T24 mirror is unavailable.
    """
    from awb_commons.db import get_pg_connection
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO mlops.churn_staging
        SELECT account_id, customer_id, balance,
               product_count, last_login_dt,
               salary_credit_dt
        FROM t24_mirror.active_accounts
        WHERE account_age_months >= 12
        """
    )
    row_count = cur.rowcount
    conn.commit()
    log.info("Extracted %d accounts from T24", row_count)
    ctx["ti"].xcom_push(key="row_count", value=row_count)
    return row_count


def validate_with_great_expectations(**ctx) -> bool:
    """Run 24-rule GE suite; halt DAG on any failure.

    Raises:
        ValueError: If any GE expectation fails.
    """
    import great_expectations as ge
    context = ge.get_context()
    result = context.run_checkpoint(
        checkpoint_name="churn_scoring_checkpoint"
    )
    if not result["success"]:
        failed = [
            r for r in result["run_results"].values()
            if not r["validation_result"]["success"]
        ]
        raise ValueError(
            f"GE validation failed: "
            f"{len(failed)} rules violated."
        )
    log.info("GE validation passed — 24 rules OK")
    return True


def score_with_xgboost(**ctx) -> str:
    """Load production model from MLflow; score accounts.

    Returns:
        MLflow run_id of scoring run.
    Raises:
        RuntimeError: If no Production model found.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.get_latest_versions(
        "awb-churn-predictor", stages=["Production"]
    )
    if not versions:
        raise RuntimeError(
            "No Production version of "
            "awb-churn-predictor in registry."
        )
    model_uri = (
        "models:/awb-churn-predictor/Production"
    )
    model = mlflow.xgboost.load_model(model_uri)
    log.info(
        "Loaded model version %s",
        versions[0].version,
    )
    return versions[0].run_id


def write_to_crm(**ctx) -> int:
    """Push Tier 1 & 2 scores to Salesforce Bulk API.

    Returns:
        Records written to Salesforce.
    Raises:
        ConnectionError: If Salesforce unavailable.
    """
    from awb_commons.salesforce import (
        SalesforceBulkClient,
    )
    from awb_commons.db import get_pg_connection

    conn = get_pg_connection()
    df = conn.execute(
        """
        SELECT customer_id, churn_score,
               churn_tier, top3_shap_json,
               retention_action
        FROM mlops.churn_scores
        WHERE churn_tier IN (1, 2)
          AND scored_at::date = CURRENT_DATE
        """
    ).fetchdf()
    sf = SalesforceBulkClient()
    written = sf.upsert_churn_risks(df)
    log.info(
        "Wrote %d records to Salesforce CRM", written
    )
    return written


with DAG(
    dag_id="awb_churn_weekly_scoring",
    default_args=DEFAULT_ARGS,
    description="AWB MR-2026-053 weekly scoring",
    schedule_interval="0 6 * * 1",
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "churn", "MR-2026-053"],
) as churn_dag:
    t1 = PythonOperator(
        task_id="extract_t24_accounts",
        python_callable=extract_t24_accounts,
    )
    t2 = PythonOperator(
        task_id="validate_great_expectations",
        python_callable=validate_with_great_expectations,
    )
    t3 = PythonOperator(
        task_id="score_with_xgboost",
        python_callable=score_with_xgboost,
    )
    t4 = PythonOperator(
        task_id="write_to_crm",
        python_callable=write_to_crm,
    )
    t1 >> t2 >> t3 >> t4
