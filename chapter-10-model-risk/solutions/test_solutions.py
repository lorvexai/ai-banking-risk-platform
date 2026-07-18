"""Verify both reference solutions pass all exercise assertions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Exercise 10.1 solution tests ─────────────────────────────
from exercises.ex_10_1_validate import (
    SCORES, OUTCOMES, REF_PROPS, _bin_proportions
)
from solutions.sol_10_1_validate import SolCreditModelValidator

def _sol1():
    return SolCreditModelValidator(SCORES, OUTCOMES, REF_PROPS)

def test_sol1_auc_roc_above_threshold():
    assert _sol1().run().auc_roc >= 0.750

def test_sol1_gini_above_threshold():
    assert _sol1().run().gini >= 0.700

def test_sol1_ks_above_threshold():
    assert _sol1().run().ks_statistic >= 0.300

def test_sol1_psi_below_warning():
    assert _sol1().run().psi < 0.100

def test_sol1_all_pass():
    assert _sol1().run().all_pass()

def test_sol1_gini_equals_2_auc_minus_1():
    r = _sol1().run()
    assert abs(r.gini - round(2 * r.auc_roc - 1, 4)) < 0.001

# ── Exercise 10.2 solution tests ─────────────────────────────
from exercises.ex_10_2_registry import (
    ModelRecord, ModelStatus, RiskRating, DeploymentDecision
)
from solutions.sol_10_2_registry import SolModelRegistry
from datetime import datetime

MR_REF = "MR-2026-099"

def _make_model():
    return ModelRecord(
        mr_reference=MR_REF,
        model_name="AWB Test Model",
        ss1_23_risk=RiskRating.MEDIUM,
        status=ModelStatus.DEVELOPMENT,
        owner="credit.team@awb.co.uk",
    )

def test_sol2_registration_succeeds():
    r = SolModelRegistry()
    m = r.register(_make_model())
    assert m.mr_reference == MR_REF

def test_sol2_duplicate_raises():
    r = SolModelRegistry()
    r.register(_make_model())
    try:
        r.register(_make_model())
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

def test_sol2_revalidation_set():
    r = SolModelRegistry()
    m = r.register(_make_model())
    months = (m.next_revalidation - datetime.utcnow()).days / 30
    assert 11 <= months <= 13

def test_sol2_gate_rejects_before_validation():
    r = SolModelRegistry()
    r.register(_make_model())
    d = r.deployment_gate(MR_REF)
    assert not d.approved

def test_sol2_validation_pass_sets_active():
    r = SolModelRegistry()
    r.register(_make_model())
    rec = r.record_validation(MR_REF, "PASS", "validator@awb")
    assert rec.status == ModelStatus.ACTIVE

def test_sol2_gate_approves_after_pass():
    r = SolModelRegistry()
    r.register(_make_model())
    r.record_validation(MR_REF, "PASS", "validator@awb")
    d = r.deployment_gate(MR_REF)
    assert d.approved

def test_sol2_unknown_ref_rejected():
    r = SolModelRegistry()
    d = r.deployment_gate("MR-2026-UNKNOWN")
    assert not d.approved
