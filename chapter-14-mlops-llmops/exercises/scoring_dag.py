# exercises/scoring_dag.py | Exercise 14.1 starter
# Chapter 14 | AWB MLOps
# Build an Airflow DAG for the churn scoring pipeline.
#
# TASK: Implement a 4-task Airflow DAG:
#   1. extract_t24_accounts  — pull from T24 PostgreSQL
#   2. validate_with_ge      — run 24-rule GE suite;
#                              HALT on any failure
#   3. score_with_xgboost    — load MLflow Production
#                              model; score all accounts
#   4. write_to_crm          — push Tier 1+2 to
#                              Salesforce Bulk API 2.0
#
# SUCCESS CRITERION:
#   DAG passes 3 unit tests including:
#   - test_dag_halts_on_ge_failure  (inject null feature)
#   - test_t4_only_writes_tier1_and_tier2
#   - test_task_dependencies_correct
#
# See solutions/ for reference implementation.
from __future__ import annotations
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner": "awb-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def extract_t24_accounts(**ctx) -> int:
    """TODO: Implement T24 account extraction.

    Should:
    - Connect to t24_mirror.active_accounts
    - Filter account_age_months >= 12
    - Write to mlops.churn_staging
    - Return row count via XCom
    """
    raise NotImplementedError


def validate_with_ge(**ctx) -> bool:
    """TODO: Implement Great Expectations validation.

    Should:
    - Run 'churn_scoring_checkpoint'
    - Raise ValueError on any rule failure
    - Log count of failed rules
    """
    raise NotImplementedError


def score_with_xgboost(**ctx) -> str:
    """TODO: Implement XGBoost scoring.

    Should:
    - Get latest Production version from MLflow
    - Raise RuntimeError if no Production version
    - Score all accounts in mlops.churn_staging
    - Return MLflow run_id via XCom
    """
    raise NotImplementedError


def write_to_crm(**ctx) -> int:
    """TODO: Implement Salesforce CRM write.

    Should:
    - Read Tier 1+2 scores from mlops.churn_scores
    - Upsert to Salesforce via Bulk API 2.0
    - Return count of records written
    """
    raise NotImplementedError


# TODO: Create the DAG with schedule "0 6 * * 1"
# and chain tasks: t1 >> t2 >> t3 >> t4
with DAG(
    dag_id="awb_churn_weekly_scoring_exercise",
    default_args=DEFAULT_ARGS,
    description="Exercise 14.1 scoring DAG",
    schedule_interval=None,  # TODO: set schedule
    start_date=days_ago(1),
    catchup=False,
) as dag:

    # TODO: create PythonOperator for each task
    # t1 = ...
    # t2 = ...
    # t3 = ...
    # t4 = ...
    # TODO: set dependencies t1 >> t2 >> t3 >> t4
    pass
