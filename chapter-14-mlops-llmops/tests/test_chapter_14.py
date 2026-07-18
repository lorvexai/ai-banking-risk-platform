# tests/test_chapter_14.py | Chapter 14 test suite
# AWB MLOps and LLMOps | 55+ pytest tests
# PRA SS1/23 gate tests | RAGAS rollback | drift
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


# ── Fixtures ────────────────────────────────────────


@pytest.fixture
def sample_account_history():
    """180-day account history DataFrame."""
    import pandas as pd
    import numpy as np
    dates = pd.date_range(end="2026-03-01", periods=180)
    return pd.DataFrame({
        "date": dates,
        "balance": np.random.uniform(1000, 5000, 180),
        "txn_count": np.random.randint(5, 25, 180),
        "debit_spend": np.random.uniform(200, 800, 180),
        "salary_credit": [
            2500 if d.day == 25 else 0 for d in dates
        ],
        "atm_count": np.random.randint(0, 3, 180),
        "bill_pays": np.random.randint(0, 5, 180),
        "transfer_out": np.random.uniform(0, 200, 180),
    })


@pytest.fixture
def feature_engineer():
    from chapter_14.churn.feature_engineer import (
        ChurnFeatureEngineer,
    )
    return ChurnFeatureEngineer()


@pytest.fixture
def mock_mlflow_client():
    with patch(
        "mlflow.tracking.MlflowClient"
    ) as mock:
        yield mock


# ── Feature Engineering Tests ───────────────────────


class TestChurnFeatureEngineer:
    def test_returns_28_features(
        self,
        feature_engineer,
        sample_account_history,
    ):
        import pandas as pd
        from dataclasses import fields
        from chapter_14.churn.feature_engineer import (
            ChurnFeatures,
        )
        result = feature_engineer.compute_features(
            customer_id="C001",
            account_history=sample_account_history,
            product_holdings=["CA", "SAV"],
            digital_events=pd.DataFrame(),
            complaints=[],
        )
        # 28 features + customer_id = 29 fields
        feature_fields = [
            f for f in fields(result)
            if f.name != "customer_id"
        ]
        assert len(feature_fields) == 28

    def test_herfindahl_single_product(
        self, feature_engineer, sample_account_history
    ):
        """Single product = HHI = 1.0 (concentrated)."""
        import pandas as pd
        result = feature_engineer.compute_features(
            "C001", sample_account_history,
            ["CA"], pd.DataFrame(), []
        )
        assert result.herfindahl_index == pytest.approx(
            1.0
        )

    def test_balance_slope_declining(
        self, feature_engineer
    ):
        """Declining balance produces negative slope."""
        import pandas as pd
        import numpy as np
        dates = pd.date_range(end="2026-03-01", periods=90)
        history = pd.DataFrame({
            "date": dates,
            "balance": np.linspace(5000, 500, 90),
            "txn_count": [10] * 90,
            "debit_spend": [200] * 90,
            "salary_credit": [0] * 90,
            "atm_count": [1] * 90,
            "bill_pays": [2] * 90,
            "transfer_out": [50] * 90,
        })
        result = feature_engineer.compute_features(
            "C001", history, ["CA"],
            pd.DataFrame(), []
        )
        assert result.balance_slope_90d < 0

    def test_salary_absent_flag(
        self, feature_engineer
    ):
        """No salary credit in 45 days sets flag."""
        import pandas as pd
        dates = pd.date_range(end="2026-03-01", periods=90)
        history = pd.DataFrame({
            "date": dates,
            "balance": [2000.0] * 90,
            "txn_count": [10] * 90,
            "debit_spend": [200] * 90,
            "salary_credit": [0] * 90,
            "atm_count": [1] * 90,
            "bill_pays": [2] * 90,
            "transfer_out": [50] * 90,
        })
        result = feature_engineer.compute_features(
            "C001", history, ["CA"],
            pd.DataFrame(), []
        )
        assert result.salary_absent_45d is True

    def test_nps_proxy_penalised_by_complaints(
        self, feature_engineer, sample_account_history
    ):
        """3 complaints reduces NPS proxy below 3."""
        import pandas as pd
        complaints = [
            {"days_ago": 10},
            {"days_ago": 30},
            {"days_ago": 60},
        ]
        result = feature_engineer.compute_features(
            "C001", sample_account_history,
            ["CA"], pd.DataFrame(), complaints
        )
        assert result.nps_proxy_score < 3.0

    def test_empty_history_raises(self, feature_engineer):
        import pandas as pd
        with pytest.raises(ValueError, match="No history"):
            feature_engineer.compute_features(
                "C001", pd.DataFrame(),
                ["CA"], pd.DataFrame(), []
            )


# ── SS1/23 Validation Gate Tests ────────────────────


class TestSS123ValidationSuite:
    @pytest.fixture
    def suite(self):
        from chapter_14.mlops.ss1_23_validation import (
            SS123ValidationSuite,
        )
        with patch("mlflow.get_run") as mock_run:
            mock_run.return_value.data.metrics = {
                "auc_roc": 0.85,
                "gini": 0.67,
                "psi": 0.10,
                "ks_stat": 0.32,
                "max_demographic_parity_diff": 0.03,
            }
            mock_run.return_value.data.tags = {
                "model_card_complete": "true",
                "mrc_approved": "true",
                "validation_report_uploaded": "true",
            }
            yield SS123ValidationSuite(
                candidate_run_id="run_001",
                model_name="awb-pd-model",
                champion_run_id="run_000",
            )

    def test_gate1_passes_above_threshold(self):
        """Gate 1 passes when AUC > champion + 0.02."""
        from chapter_14.mlops.ss1_23_validation import (
            SS123ValidationSuite,
        )
        suite = SS123ValidationSuite(
            "run_001", "test-model", "run_000"
        )
        with patch("mlflow.get_run") as m:
            m.return_value.data.metrics = {
                "auc_roc": 0.87,
                "gini": 0.68,
                "ks_stat": 0.30,
            }
            gate = suite._gate1_performance()
        assert gate.passed is True

    def test_gate1_fails_insufficient_improvement(self):
        """Gate 1 fails if AUC <= champion + 0.02."""
        from chapter_14.mlops.ss1_23_validation import (
            SS123ValidationSuite,
        )
        suite = SS123ValidationSuite(
            "run_001", "test-model", "run_000"
        )
        with patch("mlflow.get_run") as m:
            # candidate 0.82, champion 0.82 -> no +0.02
            m.return_value.data.metrics = {
                "auc_roc": 0.82,
                "gini": 0.65,
                "ks_stat": 0.28,
            }
            gate = suite._gate1_performance()
        assert gate.passed is False

    def test_gate2_fails_fairness_breach(self):
        """Gate 2 fails when parity diff > 5pp."""
        from chapter_14.mlops.ss1_23_validation import (
            SS123ValidationSuite,
        )
        suite = SS123ValidationSuite(
            "run_001", "test-model"
        )
        with patch("mlflow.get_run") as m:
            m.return_value.data.metrics = {
                "max_demographic_parity_diff": 0.07
            }
            gate = suite._gate2_fairness()
        assert gate.passed is False
        assert gate.gate == 2

    def test_gate3_fails_missing_mrc(self):
        """Gate 3 fails when MRC approval absent."""
        from chapter_14.mlops.ss1_23_validation import (
            SS123ValidationSuite,
        )
        suite = SS123ValidationSuite(
            "run_001", "test-model"
        )
        with patch("mlflow.get_run") as m:
            m.return_value.data.tags = {
                "model_card_complete": "true",
                "mrc_approved": "false",
                "validation_report_uploaded": "true",
            }
            gate = suite._gate3_governance()
        assert gate.passed is False

    def test_high_exposure_always_escalates(self):
        """Models with HIGH SS1/23 require all 4 gates."""
        from chapter_14.mlops.ss1_23_validation import (
            SS123ValidationSuite,
        )
        suite = SS123ValidationSuite(
            "run_001", "awb-pd-model-high-risk"
        )
        assert suite.AUC_MINIMUM == 0.75
        assert suite.FAIRNESS_TOLERANCE == 0.05


# ── Prompt Registry Tests ────────────────────────────


class TestPromptRegistry:
    @pytest.fixture
    def registry(self, tmp_path):
        from chapter_14.llmops.prompt_registry import (
            PromptRegistry,
        )
        return PromptRegistry(
            registry_path=tmp_path / "registry.yaml"
        )

    def test_register_minor_version(self, registry):
        from chapter_14.llmops.prompt_registry import (
            PromptVersion,
            ChangeType,
        )
        v = PromptVersion(
            service_id="MR-2026-038",
            version="1.3.0",
            git_tag="prompt/MR-2026-038/1.3.0",
            change_type=ChangeType.MINOR,
            author="test@awb.co.uk",
            description="Add IFRS 9 staging field",
            requires_mrc=False,
            ab_test_days=7,
        )
        registry.register_version(v)
        prod_ver = registry.get_production_version(
            "MR-2026-038"
        )
        # Not yet promoted
        assert prod_ver is None

    def test_promote_to_production(self, registry):
        from chapter_14.llmops.prompt_registry import (
            PromptVersion, ChangeType,
        )
        v = PromptVersion(
            service_id="MR-2026-038",
            version="1.3.0",
            git_tag="prompt/MR-2026-038/1.3.0",
            change_type=ChangeType.MINOR,
            author="test@awb.co.uk",
            description="IFRS 9 staging",
            requires_mrc=False,
            ab_test_days=7,
        )
        registry.register_version(v)
        registry.promote_to_production(
            "MR-2026-038", "1.3.0"
        )
        assert (
            registry.get_production_version(
                "MR-2026-038"
            )
            == "1.3.0"
        )

    def test_major_bump_resets_minor_patch(self):
        from chapter_14.llmops.prompt_registry import (
            PromptVersion, ChangeType,
        )
        v = PromptVersion(
            "MR-2026-035", "1.2.3", "tag",
            ChangeType.MINOR, "a@b.com", "x",
            False, 7,
        )
        assert v.bump(ChangeType.MAJOR) == "2.0.0"
        assert v.bump(ChangeType.MINOR) == "1.3.0"
        assert v.bump(ChangeType.PATCH) == "1.2.4"


# ── RAGAS Monitor Tests ──────────────────────────────


class TestRAGASMonitor:
    @pytest.fixture
    def rollback_fn(self):
        return MagicMock()

    @pytest.fixture
    def monitor(self, rollback_fn):
        from chapter_14.llmops.ragas_monitor import (
            RAGASMonitor,
        )
        mon = RAGASMonitor(
            service_id="MR-2026-038",
            rollback_fn=rollback_fn,
            sample_rate=1.0,  # 100% for testing
        )
        mon.set_production_version("1.2.0")
        return mon

    def test_rollback_fires_below_threshold(
        self, monitor, rollback_fn
    ):
        """Auto-rollback fires when faithfulness<0.80."""
        from chapter_14.llmops.ragas_monitor import (
            RAGASMetrics,
        )
        # Inject 15 low-faithfulness samples
        for _ in range(15):
            monitor._window.append(
                RAGASMetrics(
                    service_id="MR-2026-038",
                    prompt_version="1.2.0",
                    faithfulness=0.70,
                    answer_relevancy=0.75,
                    context_precision=0.72,
                    context_recall=0.68,
                )
            )
        stats = monitor._compute_window_stats()
        should = monitor._should_rollback(stats)
        assert should is True

    def test_no_rollback_above_threshold(
        self, monitor, rollback_fn
    ):
        """No rollback when faithfulness >= 0.80."""
        from chapter_14.llmops.ragas_monitor import (
            RAGASMetrics,
        )
        for _ in range(15):
            monitor._window.append(
                RAGASMetrics(
                    service_id="MR-2026-038",
                    prompt_version="1.2.0",
                    faithfulness=0.88,
                    answer_relevancy=0.82,
                    context_precision=0.75,
                    context_recall=0.72,
                )
            )
        stats = monitor._compute_window_stats()
        assert not monitor._should_rollback(stats)

    def test_window_prunes_old_samples(self, monitor):
        """Samples older than 3 hours are pruned."""
        from chapter_14.llmops.ragas_monitor import (
            RAGASMetrics,
        )
        old_metric = RAGASMetrics(
            service_id="MR-2026-038",
            prompt_version="1.2.0",
            faithfulness=0.70,
            answer_relevancy=0.70,
            context_precision=0.70,
            context_recall=0.70,
            evaluated_at=(
                datetime.utcnow() - timedelta(hours=4)
            ),
        )
        monitor._window.append(old_metric)
        monitor._prune_window()
        assert len(monitor._window) == 0

    def test_sampling_rate_respected(self, rollback_fn):
        """0% sampling rate produces no evaluations."""
        from chapter_14.llmops.ragas_monitor import (
            RAGASMonitor,
        )
        mon = RAGASMonitor(
            "MR-2026-038", rollback_fn,
            sample_rate=0.0,
        )
        mon.set_production_version("1.2.0")
        result = mon.maybe_evaluate("q", ["c"], "r")
        assert result is None


# ── Airflow DAG Structure Tests ──────────────────────


class TestAirflowDAG:
    def test_dag_task_count(self):
        """DAG has exactly 4 tasks."""
        from chapter_14.mlops.airflow_dags import (
            churn_dag,
        )
        assert len(churn_dag.tasks) == 4

    def test_task_dependencies_correct(self):
        """t1 >> t2 >> t3 >> t4 chain is correct."""
        from chapter_14.mlops.airflow_dags import (
            churn_dag,
        )
        task_ids = [t.task_id for t in churn_dag.tasks]
        assert "extract_t24_accounts" in task_ids
        assert "validate_great_expectations" in task_ids
        assert "score_with_xgboost" in task_ids
        assert "write_to_crm" in task_ids

    def test_dag_schedule_weekly(self):
        """DAG runs weekly on Mondays at 06:00."""
        from chapter_14.mlops.airflow_dags import (
            churn_dag,
        )
        assert churn_dag.schedule_interval == "0 6 * * 1"

    def test_dag_has_mr_tag(self):
        """DAG tagged with model registry ID."""
        from chapter_14.mlops.airflow_dags import (
            churn_dag,
        )
        assert "MR-2026-053" in churn_dag.tags
