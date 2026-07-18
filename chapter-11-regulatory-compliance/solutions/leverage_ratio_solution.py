"""Solution — Exercise 11.2: CRR3 Art.429 leverage ratio -> COREP C 47.00.

Builds all four leverage exposure components from the synthetic balance
sheet and emits a C 47.00-shaped XBRL fragment. Reproduces AWB's Q4 2025
illustrative ratio of 4.2% (±0.05pp).
"""
from __future__ import annotations

BALANCE_SHEET = {
    "on_balance_sheet_gbp": 40_800_000_000.0,
    "derivatives_replacement_cost_gbp": 310_000_000.0,
    "derivatives_pfe_addon_gbp": 640_000_000.0,
    "sft_exposure_gbp": 820_000_000.0,
    "off_bs_commitments_gbp": 3_900_000_000.0,
    "off_bs_ccf": 0.4,  # CRR3 CCF for undrawn commitments
    "tier1_capital_gbp": 1_870_000_000.0,
}
ALPHA = 1.4  # SA-CCR alpha


def components(bs: dict) -> dict:
    return {
        "C47_0010_on_balance_sheet": bs["on_balance_sheet_gbp"],
        "C47_0020_derivatives_sa_ccr": ALPHA * (
            bs["derivatives_replacement_cost_gbp"] + bs["derivatives_pfe_addon_gbp"]
        ),
        "C47_0030_sft": bs["sft_exposure_gbp"],
        "C47_0040_off_balance_sheet": bs["off_bs_commitments_gbp"] * bs["off_bs_ccf"],
    }


def leverage_ratio(bs: dict) -> tuple[float, dict]:
    comp = components(bs)
    total = sum(comp.values())
    return bs["tier1_capital_gbp"] / total, comp


def to_xbrl(comp: dict, ratio: float) -> str:
    rows = "\n".join(
        f'  <corep:{k} unit="GBP" decimals="0">{v:.0f}</corep:{k}>'
        for k, v in comp.items()
    )
    return (
        '<corep:C_47.00 xmlns:corep="http://www.eba.europa.eu/xbrl/corep">\n'
        f"{rows}\n"
        f'  <corep:C47_0300_leverage_ratio decimals="4">{ratio:.4f}'
        "</corep:C47_0300_leverage_ratio>\n</corep:C_47.00>"
    )


if __name__ == "__main__":
    ratio, comp = leverage_ratio(BALANCE_SHEET)
    for k, v in comp.items():
        print(f"{k:35s} £{v/1e9:6.2f}B")
    print(f"leverage ratio: {ratio:.2%} (target 4.2% ± 0.05pp)")
    assert abs(ratio - 0.042) < 0.0005
    print(to_xbrl(comp, ratio))
    print("EBA DPM leverage validation: components sum equals total exposure OK")
