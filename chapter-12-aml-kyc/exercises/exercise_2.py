"""
Exercise 12.2: Build an End-to-End AML Pipeline with Tipping-Off Guarantee
Difficulty: 4/5 | Estimated time: 45 minutes

Task: Wire together AMLTransactionScorer (MR-2026-061), AMLNetworkAnalyser,
TippingOffGuardrail, and a mock NCA SubmitSAR endpoint.

Process 20 synthetic transactions including one structuring ring.
Verify: (1) ring is detected, (2) SAR draft generated,
(3) credit agent receives only BLOCKED regardless of SAR vs KYC block.

Starter: This file
Solution: github.com/lorvenio/ai-banking-risk-platform/chapter_12/solutions/
"""
from __future__ import annotations

from typing import Optional


# ── Mock classes for the exercise ───────────────────────────────────────
class MockAMLScorer:
    """Stand-in for AMLTransactionScorer (MR-2026-061)."""

    def score_transaction(
        self,
        transaction: dict,
    ) -> float:
        """Return a mock AML score 0.0-1.0."""
        # TODO: Return high score for suspicious amounts
        return 0.1


class MockTippingOffGuardrail:
    """Stand-in for TippingOffGuardrail (POCA 2002 s.333A)."""

    @staticmethod
    def get_safe_credit_status(
        sar_filed: bool,
        kyc_result_status: str,
    ) -> str:
        """Return credit gate status without leaking SAR information.

        Args:
            sar_filed: Whether a SAR has been filed for this entity.
            kyc_result_status: The underlying KYC result.

        Returns:
            Safe status string for credit agent (never reveals SAR filing).
        """
        # TODO: Implement POCA s.333A guarantee
        # If sar_filed is True, return "BLOCKED" regardless of kyc_result_status
        # The credit agent must not be able to tell WHY it is blocked
        return kyc_result_status


class MockSARSubmitter:
    """Stand-in for NCA SubmitSAR API integration."""

    def submit_sar(self, sar_draft: dict) -> dict:
        """Submit a SAR draft to the mock NCA endpoint."""
        print(f"[MOCK NCA] SAR received: {sar_draft.get('nature_of_suspicion', '')[:60]}...")
        return {"sar_reference": "SAR-2025-TEST-001", "status": "ACCEPTED"}


# ── Pipeline function ────────────────────────────────────────────────────
def run_aml_pipeline(
    transactions: list,
    sar_filed_accounts: Optional[list] = None,
) -> dict:
    """Run the end-to-end AML pipeline.

    Args:
        transactions: List of transaction dicts.
        sar_filed_accounts: Account IDs with active SAR filings.

    Returns:
        Pipeline results dict.
    """
    if sar_filed_accounts is None:
        sar_filed_accounts = []

    scorer = MockAMLScorer()
    guardrail = MockTippingOffGuardrail()
    submitter = MockSARSubmitter()

    results = {
        "alerts_generated": 0,
        "sars_submitted": 0,
        "credit_statuses": {},
    }

    for txn in transactions:
        score = scorer.score_transaction(txn)

        if score >= 0.90:
            results["alerts_generated"] += 1
            # TODO: Generate SAR draft and submit
            # TODO: Update results["sars_submitted"]
            pass

        # TODO: For each account, determine credit gate status
        account_id = txn.get("sender_id")
        sar_active = account_id in sar_filed_accounts
        kyc_status = "CLEARED"  # simplified

        credit_status = guardrail.get_safe_credit_status(sar_active, kyc_status)
        results["credit_statuses"][account_id] = credit_status

    return results


if __name__ == "__main__":
    # Generate 20 synthetic transactions
    transactions = [
        {"sender_id": f"ACC{i:03d}", "receiver_id": "ACC999",
         "amount_gbp": 4500 if i < 5 else 25000}
        for i in range(20)
    ]

    # Accounts 0-4 have SAR filings
    sar_accounts = [f"ACC{i:03d}" for i in range(5)]

    results = run_aml_pipeline(transactions, sar_filed_accounts=sar_accounts)
    print(f"Alerts: {results['alerts_generated']}")
    print(f"SARs submitted: {results['sars_submitted']}")

    # Verify tipping-off guarantee
    for acc_id in sar_accounts:
        status = results["credit_statuses"].get(acc_id, "UNKNOWN")
        assert status == "BLOCKED", (
            f"POCA s.333A VIOLATION: {acc_id} SAR status leaked via {status}"
        )
        print(f"  {acc_id}: {status} (correct — SAR not disclosed)")
    print("All tipping-off guarantee checks passed.")
