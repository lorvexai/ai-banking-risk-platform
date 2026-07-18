"""Dataclass schemas for Chapter 11 regulatory reporting.

Model IDs:
    MR-2026-047: HMRC Tax Reporting Engine (LOW risk)
    MR-2026-048: MJRRP - Multi-Jurisdiction Platform (HIGH risk)
    MR-2026-049: Basel Credit Risk Reporting (MEDIUM risk)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List
import logging

log = logging.getLogger(__name__)


@dataclass
class LeverageRatioResult:
    """CRR3 Art. 429 leverage ratio — all 4 exposure components.

    Formula: Tier 1 Capital / Total Leverage Exposure Measure >= 3%
    AWB minimum: 3.0% (not G-SIB so 3.5% add-on not applicable).
    """
    quarter_end: date
    tier1_capital_gbp: Decimal
    on_balance_sheet_gbp: Decimal    # CRR3 Art. 429b
    sa_ccr_derivatives_gbp: Decimal  # CRR3 Art. 429c (SA-CCR)
    sft_exposure_gbp: Decimal        # CRR3 Art. 429d (repos, SBL)
    off_balance_sheet_gbp: Decimal   # CRR3 Arts 429e-429g (CCFs)
    total_exposure_gbp: Decimal = field(init=False)
    leverage_ratio_pct: float = field(init=False)
    breaches_minimum: bool = field(init=False)
    MINIMUM_RATIO_PCT: float = 3.0   # CRR3 Art. 429; 3.5% G-SIBs

    def __post_init__(self) -> None:
        self.total_exposure_gbp = (
            self.on_balance_sheet_gbp
            + self.sa_ccr_derivatives_gbp
            + self.sft_exposure_gbp
            + self.off_balance_sheet_gbp
        )
        if self.total_exposure_gbp <= 0:
            raise ValueError("Total exposure must be positive")
        self.leverage_ratio_pct = float(
            self.tier1_capital_gbp
            / self.total_exposure_gbp * 100
        )
        self.breaches_minimum = (
            self.leverage_ratio_pct < self.MINIMUM_RATIO_PCT
        )


@dataclass
class LCRResult:
    """CRR3 Arts 411-428 Liquidity Coverage Ratio.

    Formula: Adjusted HQLA / Net Cash Outflows (30-day stress) >= 100%
    COREP return: C 72.00 (monthly).
    """
    reporting_date: date
    hqla_level1_gbp: Decimal   # 0% haircut (central bank, govts)
    hqla_level2a_gbp: Decimal  # 15% haircut (covered bonds, IG corps)
    hqla_level2b_gbp: Decimal  # 25-50% haircut (RMBS, equities)
    net_cash_outflows_30d_gbp: Decimal
    lcr_pct: float = field(init=False)
    MINIMUM_PCT: float = 100.0

    def __post_init__(self) -> None:
        adjusted_hqla = (
            self.hqla_level1_gbp
            + self.hqla_level2a_gbp * Decimal("0.85")
            + self.hqla_level2b_gbp * Decimal("0.75")
        )
        if self.net_cash_outflows_30d_gbp <= 0:
            raise ValueError("Net cash outflows must be positive")
        self.lcr_pct = float(
            adjusted_hqla / self.net_cash_outflows_30d_gbp * 100
        )


@dataclass
class NSFRResult:
    """CRR3 Arts 428a-428au Net Stable Funding Ratio.

    Formula: Available Stable Funding / Required Stable Funding >= 100%
    COREP return: C 80.00 (quarterly).
    """
    quarter_end: date
    available_stable_funding_gbp: Decimal  # ASF
    required_stable_funding_gbp: Decimal   # RSF
    nsfr_pct: float = field(init=False)
    MINIMUM_PCT: float = 100.0

    def __post_init__(self) -> None:
        if self.required_stable_funding_gbp <= 0:
            raise ValueError("RSF must be positive")
        self.nsfr_pct = float(
            self.available_stable_funding_gbp
            / self.required_stable_funding_gbp * 100
        )


@dataclass
class RWAResult:
    """CRR3 Arts 112-191 Risk-Weighted Assets by pillar.

    COREP returns: C 02.00 (SA credit), C 08.00 (IRB credit),
    C 18.00 (market FRTB), C 10.00 (operational SMA).
    """
    quarter_end: date
    credit_risk_sa_gbp: Decimal      # CRR3 Arts 112-141
    credit_risk_irb_gbp: Decimal     # CRR3 Arts 142-191
    market_risk_frtb_gbp: Decimal    # FRTB SA approach
    operational_risk_sma_gbp: Decimal # Basel III SMA
    cet1_capital_gbp: Decimal = Decimal("0")
    total_rwa_gbp: Decimal = field(init=False)
    cet1_ratio_pct: float = field(init=False)
    OUTPUT_FLOOR_PCT: float = 72.5   # CRR3 Art. 465 output floor

    def __post_init__(self) -> None:
        self.total_rwa_gbp = (
            self.credit_risk_sa_gbp
            + self.credit_risk_irb_gbp
            + self.market_risk_frtb_gbp
            + self.operational_risk_sma_gbp
        )
        if self.cet1_capital_gbp and self.total_rwa_gbp > 0:
            self.cet1_ratio_pct = float(
                self.cet1_capital_gbp / self.total_rwa_gbp * 100
            )
        else:
            self.cet1_ratio_pct = 0.0


@dataclass
class Section104Pool:
    """Section 104 pool for CGT calculation (TCGA 1992 s104)."""
    asset_name: str
    total_quantity: Decimal = Decimal("0")
    total_qualifying_expenditure: Decimal = Decimal("0")

    @property
    def average_cost_per_unit(self) -> Decimal:
        if self.total_quantity <= 0:
            return Decimal("0")
        return self.total_qualifying_expenditure / self.total_quantity


@dataclass
class CGTDisposal:
    """Capital gains tax disposal — Section 104 pool calculation."""
    disposal_date: date
    asset_name: str
    disposal_proceeds_gbp: Decimal
    allowable_cost_gbp: Decimal       # From Section 104 pool
    gain_or_loss_gbp: Decimal = field(init=False)
    is_gain: bool = field(init=False)
    bed_and_breakfast_matched: bool = False  # 30-day rule applied

    def __post_init__(self) -> None:
        self.gain_or_loss_gbp = (
            self.disposal_proceeds_gbp - self.allowable_cost_gbp
        )
        self.is_gain = self.gain_or_loss_gbp >= 0


@dataclass
class TaxLetter:
    """LLM-generated HMRC client tax summary letter.

    MR-2026-047: requires_adviser_review always True per EU AI Act Art.14.
    FCA PS22/9: must be clear, fair, personalised, not misleading.
    """
    client_id: str
    tax_year: str                    # e.g., "2025-26"
    letter_content: str
    total_gain_gbp: Decimal
    annual_cgt_allowance_gbp: Decimal = Decimal("3000")  # HMRC 2025-26
    taxable_gain_gbp: Decimal = field(init=False)
    tax_liability_gbp: Decimal = Decimal("0")
    requires_adviser_review: bool = True
    generated_by: str = "Gemini 3.5 Flash (AI-ASSISTED DRAFT)"
    model_id: str = "MR-2026-047"

    def __post_init__(self) -> None:
        self.taxable_gain_gbp = max(
            Decimal("0"),
            self.total_gain_gbp - self.annual_cgt_allowance_gbp
        )


@dataclass
class COREPReturn:
    """Filed COREP regulatory return with audit trail."""
    return_code: str          # e.g., "C 47.00"
    reporting_period: date
    xbrl_instance_xml: str
    model_id: str = "MR-2026-048"
    filed_at: Optional[datetime] = None
    filing_reference: Optional[str] = None
    validation_passed: bool = False
    pra_gabriel_response: Optional[str] = None


@dataclass
class XBRLInstance:
    """EBA XBRL Taxonomy 4.0 instance document."""
    return_code: str
    taxonomy_version: str = "4.0"    # EBA ITS effective Q1 2025
    entity_id: str = "AWB"
    xml_content: str = ""
    validation_errors: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0
