"""Solution — Exercise 12.2: end-to-end AML pipeline with tipping-off guarantee.

Wires AMLTransactionScorer -> AMLNetworkAnalyser -> TippingOffGuardrail ->
mock NCA SubmitSAR. The credit agent receives only BLOCKED — it cannot
distinguish a SAR filing from a standard KYC block (POCA s.333A).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Txn:
    txn_id: str
    customer: str
    amount_gbp: float


def make_txns(seed: int = 21) -> list[Txn]:
    rng = random.Random(seed)
    txns = [Txn(f"T-{i:03d}", f"CUST-{rng.randint(1, 8)}",
                rng.uniform(100, 45_000)) for i in range(17)]
    # one structuring ring: CUST-9 splits £27k into three sub-£10k txns
    txns += [Txn(f"T-9{i}", "CUST-9", 9_000 + i * 100) for i in range(3)]
    return txns


class AMLTransactionScorer:  # MR-2026-061
    def score(self, txns: list[Txn]) -> dict[str, float]:
        by_cust: dict[str, list[Txn]] = {}
        for t in txns:
            by_cust.setdefault(t.customer, []).append(t)
        return {
            c: (0.93 if len([t for t in ts if 8_500 < t.amount_gbp < 10_000]) >= 3
                else 0.12)
            for c, ts in by_cust.items()
        }


class AMLNetworkAnalyser:
    def confirm_ring(self, customer: str, txns: list[Txn]) -> bool:
        sub10k = [t for t in txns if t.customer == customer and t.amount_gbp < 10_000]
        return len(sub10k) >= 3


class NCAEndpoint:
    def __init__(self):
        self.sars: list[str] = []

    def submit_sar(self, customer: str, narrative: str) -> None:
        self.sars.append(customer)


class TippingOffGuardrail:
    """POCA s.333A: downstream systems must never learn WHY a block exists."""

    def outward_status(self, customer: str, sar_filed: bool, kyc_block: bool) -> str:
        return "BLOCKED" if (sar_filed or kyc_block) else "CLEAR"


if __name__ == "__main__":
    txns = make_txns()
    scores = AMLTransactionScorer().score(txns)
    analyser, nca, guard = AMLNetworkAnalyser(), NCAEndpoint(), TippingOffGuardrail()
    credit_agent_view: dict[str, str] = {}
    for customer, score in sorted(scores.items()):
        sar = False
        if score > 0.9 and analyser.confirm_ring(customer, txns):
            nca.submit_sar(customer, f"structuring pattern, score {score:.2f}")
            sar = True
        credit_agent_view[customer] = guard.outward_status(customer, sar, kyc_block=False)
    print(f"transactions processed: {len(txns)}")
    print(f"rings detected & SAR drafts filed: {len(nca.sars)} ({nca.sars})")
    print(f"credit agent sees: { {c: s for c, s in credit_agent_view.items() if s != 'CLEAR'} }")
    assert nca.sars == ["CUST-9"]
    assert credit_agent_view["CUST-9"] == "BLOCKED"
    assert "SAR" not in str(credit_agent_view)
    print("Tipping-off guarantee holds: downstream sees only BLOCKED")
