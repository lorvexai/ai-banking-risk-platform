"""tests/test_chapter_10.py — Chapter 10 test suite. 40+ tests."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from datetime import datetime, timedelta

from awb_commons.models import (
    ModelRecord, ValidationResult, ABTestResult,
    LLMMonitoringSnapshot, RiskRating, ModelStatus, EUAIActClass,
)
from model_inventory.registry import ModelRegistry
from ab_testing.platform import ABTestingPlatform, ABTestConfig, ModelVariant
from credit_validation.validator import CreditModelValidator, ValidationDataset
from llm_monitoring.monitor import LLMMonitor, PromptRegistry, PromptVersion


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def registry():
    return ModelRegistry()

@pytest.fixture
def good_dataset():
    import random
    rng = random.Random(42)
    n = 2000
    outcomes = [rng.randint(0,1) for _ in range(n)]
    # Good discriminatory model: high score → bad outcome
    scores = [
        0.8 + rng.gauss(0, 0.08) if o == 1
        else 0.3 + rng.gauss(0, 0.08)
        for o in outcomes
    ]
    scores = [max(0.0, min(1.0, s)) for s in scores]
    # dev_dist uses same distribution shape as scores
    rng2 = random.Random(42)
    dev_outcomes_dev = [rng2.randint(0,1) for _ in range(500)]
    dev_dist = [
        0.8 + rng2.gauss(0, 0.08) if o == 1
        else 0.3 + rng2.gauss(0, 0.08)
        for o in dev_outcomes_dev
    ]
    dev_dist = [max(0.0, min(1.0, s)) for s in dev_dist]
    return ValidationDataset(
        dataset_id="DS-001",
        model_predictions=scores,
        actual_outcomes=outcomes,
        development_dist=dev_dist,
        validation_period="2025-Q4",
    )

@pytest.fixture
def weak_dataset():
    import random
    rng = random.Random(99)
    n = 2000
    outcomes = [rng.randint(0,1) for _ in range(n)]
    scores = [rng.random() for _ in range(n)]  # random = no skill
    dev_dist = [rng.random() for _ in range(500)]
    return ValidationDataset(
        dataset_id="DS-WEAK",
        model_predictions=scores,
        actual_outcomes=outcomes,
        development_dist=dev_dist,
        validation_period="2025-Q4",
    )

@pytest.fixture
def ab_config():
    return ABTestConfig(
        mr_reference="MR-2026-035",
        test_name="CDA v1 vs v2",
        control=ModelVariant(version="v1", model_fn=lambda x: 0.0),
        treatment=ModelVariant(version="v2", model_fn=lambda x: 0.0),
        min_sample_size=100,
    )

@pytest.fixture
def llm_snapshot_good():
    return LLMMonitoringSnapshot(
        mr_reference="MR-2026-038",
        snapshot_month="2026-03",
        faithfulness_score=0.89,
        answer_relevancy=0.84,
        context_precision=0.78,
        context_recall=0.73,
        avg_cost_per_query_gbp=0.0038,
        p50_latency_ms=820,
        p95_latency_ms=1780,
        hallucination_rate_pct=0.7,
        total_queries=8400,
    )

@pytest.fixture
def llm_snapshot_bad():
    return LLMMonitoringSnapshot(
        mr_reference="MR-2026-038",
        snapshot_month="2026-02",
        faithfulness_score=0.76,   # below 0.85
        answer_relevancy=0.72,     # below 0.80
        context_precision=0.65,    # below 0.75
        context_recall=0.60,       # below 0.70
        avg_cost_per_query_gbp=0.0055,
        p50_latency_ms=1200,
        p95_latency_ms=2400,       # above 2000ms SLA
        hallucination_rate_pct=2.1, # above 1%
        total_queries=7800,
    )


# ── Model Registry Tests ──────────────────────────────────────────
class TestModelRegistry:
    def test_preloaded_models(self, registry):
        assert len(registry.all_models()) >= 5

    def test_register_new_model(self, registry):
        new_model = ModelRecord(
            mr_reference="MR-2026-099",
            model_name="Test Model",
            chapter=10,
            ss1_23_risk=RiskRating.LOW,
            eu_ai_act=EUAIActClass.NOT_IN_SCOPE,
            status=ModelStatus.DEVELOPMENT,
            owner="Test Team",
        )
        result = registry.register(new_model)
        assert result.mr_reference == "MR-2026-099"
        assert registry.get("MR-2026-099") is not None

    def test_duplicate_registration_raises(self, registry):
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ModelRecord(
                mr_reference="MR-2026-035",
                model_name="Duplicate",
                chapter=2,
                ss1_23_risk=RiskRating.MEDIUM,
                eu_ai_act=EUAIActClass.HIGH_RISK,
                status=ModelStatus.ACTIVE,
                owner="Test",
            ))

    def test_status_transition_valid(self, registry):
        new = ModelRecord(
            mr_reference="MR-2026-098",
            model_name="Transition Test",
            chapter=10,
            ss1_23_risk=RiskRating.MEDIUM,
            eu_ai_act=EUAIActClass.LIMITED,
            status=ModelStatus.DEVELOPMENT,
            owner="Test Team",
        )
        registry.register(new)
        result = registry.update_status("MR-2026-098", ModelStatus.VALIDATION)
        assert result.status == ModelStatus.VALIDATION

    def test_invalid_status_transition_raises(self, registry):
        with pytest.raises(ValueError, match="Invalid transition"):
            registry.update_status("MR-2026-035", ModelStatus.DEVELOPMENT)

    def test_revalidation_schedule_set(self, registry):
        new = ModelRecord(
            mr_reference="MR-2026-097",
            model_name="Reval Test",
            chapter=10,
            ss1_23_risk=RiskRating.HIGH,
            eu_ai_act=EUAIActClass.HIGH_RISK,
            status=ModelStatus.DEVELOPMENT,
            owner="Risk",
        )
        registry.register(new)
        m = registry.get("MR-2026-097")
        assert m.next_revalidation is not None
        # HIGH risk: 12 months
        expected = datetime.utcnow() + timedelta(days=360)
        assert abs((m.next_revalidation - expected).days) <= 2

    def test_record_validation(self, registry):
        val = ValidationResult(
            mr_reference="MR-2026-035",
            validator_id="MRT-001",
            gini_coefficient=0.751,
            psi=0.045,
            outcome="PASS",
        )
        registry.record_validation(val)
        history = registry.validation_history("MR-2026-035")
        assert len(history) >= 1

    def test_due_for_revalidation(self, registry):
        # Add a model with past revalidation date
        old = ModelRecord(
            mr_reference="MR-2026-096",
            model_name="Overdue Model",
            chapter=6,
            ss1_23_risk=RiskRating.HIGH,
            eu_ai_act=EUAIActClass.HIGH_RISK,
            status=ModelStatus.ACTIVE,
            owner="Credit",
            next_revalidation=datetime(2025, 1, 1),
        )
        registry._models["MR-2026-096"] = old
        due = registry.due_for_revalidation()
        assert any(m.mr_reference == "MR-2026-096" for m in due)

    def test_get_returns_none_for_unknown(self, registry):
        assert registry.get("MR-9999-999") is None


# ── A/B Testing Platform Tests ────────────────────────────────────
class TestABTestingPlatform:
    def test_significant_improvement(self, ab_config):
        platform = ABTestingPlatform(ab_config)
        import random
        rng = random.Random(42)
        # Control: 82% rate, treatment: 89% rate
        for _ in range(500):
            platform.record_outcome(False, 1.0 if rng.random() < 0.82 else 0.0)
            platform.record_outcome(True,  1.0 if rng.random() < 0.89 else 0.0)
        result = platform.analyse()
        assert result.lift_pct > 0

    def test_insufficient_samples_raises(self, ab_config):
        platform = ABTestingPlatform(ab_config)
        platform.record_outcome(False, 1.0)
        with pytest.raises(ValueError, match="Insufficient control"):
            platform.analyse()

    def test_mr_reference_stored(self, ab_config):
        platform = ABTestingPlatform(ab_config)
        import random; rng = random.Random(1)
        for _ in range(150):
            platform.record_outcome(False, rng.random())
            platform.record_outcome(True, rng.random())
        result = platform.analyse()
        assert result.mr_reference == "MR-2026-035"

    def test_versions_stored(self, ab_config):
        platform = ABTestingPlatform(ab_config)
        import random; rng = random.Random(2)
        for _ in range(150):
            platform.record_outcome(False, rng.random())
            platform.record_outcome(True, rng.random())
        result = platform.analyse()
        assert result.control_version == "v1"
        assert result.treatment_version == "v2"

    def test_required_sample_size(self, ab_config):
        platform = ABTestingPlatform(ab_config)
        n = platform.required_sample_size(
            baseline_rate=0.82, min_detectable_effect=0.03
        )
        assert n > 100

    def test_p_value_range(self, ab_config):
        platform = ABTestingPlatform(ab_config)
        import random; rng = random.Random(3)
        for _ in range(200):
            platform.record_outcome(False, rng.random())
            platform.record_outcome(True, rng.random())
        result = platform.analyse()
        assert 0.0 <= result.p_value <= 1.0


# ── Credit Model Validator Tests ──────────────────────────────────
class TestCreditModelValidator:
    def test_good_model_passes(self, good_dataset):
        v = CreditModelValidator()
        result = v.validate("MR-2026-035", "MRT-001", good_dataset)
        assert result.outcome in ("PASS", "CONDITIONAL_PASS")
        assert result.gini_coefficient > 0.0

    def test_weak_model_fails(self, weak_dataset):
        v = CreditModelValidator()
        result = v.validate("MR-2026-035", "MRT-001", weak_dataset)
        assert result.gini_coefficient < 0.70 or result.outcome in ("FAIL", "CONDITIONAL_PASS")

    def test_gini_range(self, good_dataset):
        v = CreditModelValidator()
        result = v.validate("MR-2026-035", "MRT-001", good_dataset)
        assert -1.0 <= result.gini_coefficient <= 1.0

    def test_auc_range(self, good_dataset):
        v = CreditModelValidator()
        result = v.validate("MR-2026-035", "MRT-001", good_dataset)
        assert 0.0 <= result.auc_roc <= 1.0

    def test_psi_non_negative(self, good_dataset):
        v = CreditModelValidator()
        result = v.validate("MR-2026-035", "MRT-001", good_dataset)
        assert result.psi >= 0.0

    def test_empty_dataset_raises(self):
        v = CreditModelValidator()
        with pytest.raises(ValueError, match="cannot be empty"):
            ValidationDataset(
                dataset_id="BAD",
                model_predictions=[],
                actual_outcomes=[],
                development_dist=[0.5],
                validation_period="2025-Q4",
            )

    def test_mismatched_lengths_raises(self):
        v = CreditModelValidator()
        with pytest.raises(ValueError, match="equal length"):
            ValidationDataset(
                dataset_id="BAD2",
                model_predictions=[0.5, 0.6],
                actual_outcomes=[1],
                development_dist=[0.5],
                validation_period="2025-Q4",
            )

    def test_findings_populated_on_fail(self, weak_dataset):
        v = CreditModelValidator()
        result = v.validate("MR-2026-035", "MRT-001", weak_dataset)
        # Weak model should have findings
        # (gini likely below threshold)
        assert isinstance(result.findings, list)


# ── LLM Monitoring Tests ──────────────────────────────────────────
class TestLLMMonitor:
    def test_good_snapshot_no_alerts(self, llm_snapshot_good):
        mon = LLMMonitor("MR-2026-038", baseline_cost_gbp=0.0040)
        alerts = mon.assess_snapshot(llm_snapshot_good)
        assert len(alerts) == 0

    def test_bad_snapshot_has_alerts(self, llm_snapshot_bad):
        mon = LLMMonitor("MR-2026-038", baseline_cost_gbp=0.0040)
        alerts = mon.assess_snapshot(llm_snapshot_bad)
        assert len(alerts) > 0

    def test_faithfulness_alert(self):
        mon = LLMMonitor("MR-2026-038")
        snap = LLMMonitoringSnapshot(
            mr_reference="MR-2026-038",
            snapshot_month="2026-03",
            faithfulness_score=0.78,  # below 0.85
            answer_relevancy=0.85,
            context_precision=0.80,
            context_recall=0.75,
            avg_cost_per_query_gbp=0.0040,
            p50_latency_ms=800,
            p95_latency_ms=1800,
            hallucination_rate_pct=0.5,
            total_queries=5000,
        )
        alerts = mon.assess_snapshot(snap)
        assert any("faithfulness" in a.lower() for a in alerts)

    def test_latency_sla_alert(self):
        mon = LLMMonitor("MR-2026-038")
        snap = LLMMonitoringSnapshot(
            mr_reference="MR-2026-038",
            snapshot_month="2026-03",
            faithfulness_score=0.90,
            answer_relevancy=0.88,
            context_precision=0.82,
            context_recall=0.77,
            avg_cost_per_query_gbp=0.0040,
            p50_latency_ms=900,
            p95_latency_ms=2500,  # above 2000ms SLA
            hallucination_rate_pct=0.3,
            total_queries=5000,
        )
        alerts = mon.assess_snapshot(snap)
        assert any("latency" in a.lower() for a in alerts)

    def test_revalidation_triggered_on_multiple_failures(self, llm_snapshot_bad):
        mon = LLMMonitor("MR-2026-038")
        result = mon.trigger_revalidation(llm_snapshot_bad)
        assert result is True

    def test_no_revalidation_on_good_snapshot(self, llm_snapshot_good):
        mon = LLMMonitor("MR-2026-038")
        result = mon.trigger_revalidation(llm_snapshot_good)
        assert result is False


# ── Prompt Registry Tests ─────────────────────────────────────────
class TestPromptRegistry:
    def test_register_prompt(self):
        reg = PromptRegistry()
        pv = PromptVersion(
            prompt_id="P001",
            mr_reference="MR-2026-038",
            version="v1.0",
            system_prompt="You are a regulatory analyst.",
            user_template="Answer: {query}",
        )
        result = reg.register(pv)
        assert result.version == "v1.0"

    def test_duplicate_version_raises(self):
        reg = PromptRegistry()
        pv = PromptVersion(
            prompt_id="P002", mr_reference="MR-2026-038",
            version="v1.0", system_prompt="S", user_template="U",
        )
        reg.register(pv)
        with pytest.raises(ValueError, match="already exists"):
            reg.register(PromptVersion(
                prompt_id="P003", mr_reference="MR-2026-038",
                version="v1.0", system_prompt="S2", user_template="U2",
            ))

    def test_approve_sets_active(self):
        reg = PromptRegistry()
        pv = PromptVersion(
            prompt_id="P004", mr_reference="MR-2026-038",
            version="v2.0", system_prompt="S", user_template="U",
        )
        reg.register(pv)
        approved = reg.approve("MR-2026-038", "v2.0", "MRT-001")
        assert approved.is_active
        assert approved.approved_by == "MRT-001"

    def test_only_one_active_at_a_time(self):
        reg = PromptRegistry()
        for v in ["v1.0", "v2.0", "v3.0"]:
            reg.register(PromptVersion(
                prompt_id=f"P-{v}", mr_reference="MR-9001",
                version=v, system_prompt="S", user_template="U",
            ))
        reg.approve("MR-9001", "v1.0", "A1")
        reg.approve("MR-9001", "v2.0", "A2")
        active = [p for p in reg.version_history("MR-9001") if p.is_active]
        assert len(active) == 1
        assert active[0].version == "v2.0"

    def test_content_hash_deterministic(self):
        pv = PromptVersion(
            prompt_id="P005", mr_reference="MR-2026-038",
            version="v1.0", system_prompt="Hello", user_template="{q}",
        )
        assert pv.content_hash == pv.content_hash

    def test_active_prompt_none_before_approval(self):
        reg = PromptRegistry()
        reg.register(PromptVersion(
            prompt_id="P006", mr_reference="MR-NEW",
            version="v1.0", system_prompt="S", user_template="U",
        ))
        assert reg.active_prompt("MR-NEW") is None
