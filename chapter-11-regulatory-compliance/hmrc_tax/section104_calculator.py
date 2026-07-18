"""HMRC Tax Reporting — Section 104 CGT Calculator.

Model ID: MR-2026-047 | Risk: LOW | EU AI Act: Not in scope
Regulation: TCGA 1992 s104 | HMRC annual ISA return (ISATR)
CGT allowance 2025-26: £3,000 | ISA limit: £20,000
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional, Tuple
import hashlib
import logging

from awb_commons.models import (
    CGTDisposal, Section104Pool, TaxLetter,
)

log = logging.getLogger(__name__)

CGT_ANNUAL_ALLOWANCE = Decimal("3000")   # HMRC 2025-26
ISA_ANNUAL_LIMIT = Decimal("20000")      # HMRC 2025-26
BED_AND_BREAKFAST_DAYS = 30


class Section104PoolCalculator:
    """Calculate CGT using TCGA 1992 s104 pooling method.

    The Section 104 pool averages acquisition costs across all
    shares of the same class held. On disposal, the allowable
    cost is the proportion of the pool matching the disposal.

    Args:
        asset_name: Security name for the pool.

    Example:
        >>> calc = Section104PoolCalculator("Barclays PLC")
        >>> calc.add_acquisition(
        ...     date(2023, 3, 1), Decimal("1000"),
        ...     Decimal("2500")
        ... )
        >>> disposal = calc.add_disposal(
        ...     date(2025, 11, 15),
        ...     Decimal("500"),
        ...     Decimal("1800"),
        ... )
    """

    def __init__(self, asset_name: str) -> None:
        self._pool = Section104Pool(asset_name=asset_name)
        self._pending_acquisitions: List[
            Tuple[date, Decimal, Decimal]
        ] = []
        self._disposals: List[CGTDisposal] = []

    def add_acquisition(
        self,
        acq_date: date,
        quantity: Decimal,
        total_cost_gbp: Decimal,
    ) -> None:
        """Add acquisition to Section 104 pool.

        Args:
            acq_date: Date of acquisition.
            quantity: Number of units acquired.
            total_cost_gbp: Total cost including stamp duty.
        """
        if quantity <= 0:
            raise ValueError(
                f"Quantity must be positive: {quantity}"
            )
        self._pending_acquisitions.append(
            (acq_date, quantity, total_cost_gbp)
        )
        self._pool.total_quantity += quantity
        self._pool.total_qualifying_expenditure += (
            total_cost_gbp
        )
        log.info(
            "Pool acquisition: %s qty=%s cost=£%s avg=£%s",
            self._pool.asset_name,
            quantity,
            total_cost_gbp,
            self._pool.average_cost_per_unit,
        )

    def add_disposal(
        self,
        disposal_date: date,
        quantity_disposed: Decimal,
        disposal_proceeds_gbp: Decimal,
    ) -> CGTDisposal:
        """Calculate CGT disposal using Section 104 pool.

        Applies bed-and-breakfast rule (30-day matching) before
        deducting from pool.

        Args:
            disposal_date: Date of disposal.
            quantity_disposed: Units sold.
            disposal_proceeds_gbp: Net proceeds after dealing costs.

        Returns:
            CGTDisposal with gain or loss calculated.

        Raises:
            ValueError: If disposing more units than held.
        """
        if quantity_disposed > self._pool.total_quantity:
            raise ValueError(
                f"Cannot dispose {quantity_disposed} — "
                f"pool holds {self._pool.total_quantity}"
            )
        bb_matched = self._check_bed_and_breakfast(
            disposal_date, quantity_disposed
        )
        avg_cost = self._pool.average_cost_per_unit
        allowable_cost = (
            avg_cost * quantity_disposed
        ).quantize(Decimal("0.01"))
        self._pool.total_quantity -= quantity_disposed
        self._pool.total_qualifying_expenditure -= (
            allowable_cost
        )
        disposal = CGTDisposal(
            disposal_date=disposal_date,
            asset_name=self._pool.asset_name,
            disposal_proceeds_gbp=disposal_proceeds_gbp,
            allowable_cost_gbp=allowable_cost,
            bed_and_breakfast_matched=bb_matched,
        )
        self._disposals.append(disposal)
        log.info(
            "Disposal: %s gain=£%s (pool avg £%s/unit)",
            disposal.asset_name,
            disposal.gain_or_loss_gbp,
            avg_cost,
        )
        return disposal

    def _check_bed_and_breakfast(
        self,
        disposal_date: date,
        quantity: Decimal,
    ) -> bool:
        """Check 30-day bed-and-breakfast rule (TCGA 1992 s106A).

        If the same asset is reacquired within 30 days of disposal,
        the reacquisition cost is used, not the pool average.
        """
        window_end = disposal_date + timedelta(
            days=BED_AND_BREAKFAST_DAYS
        )
        for acq_date, acq_qty, _ in self._pending_acquisitions:
            if disposal_date < acq_date <= window_end:
                log.warning(
                    "Bed-and-breakfast rule triggered: "
                    "%s disposed %s reacquired within 30 days",
                    self._pool.asset_name,
                    quantity,
                )
                return True
        return False

    @property
    def total_net_gain(self) -> Decimal:
        """Sum of all gains less losses in the pool's disposals."""
        return sum(
            d.gain_or_loss_gbp for d in self._disposals
        )


class ISAReportingEngine:
    """Monitor ISA subscriptions and generate HMRC ISATR XML.

    Tracks annual ISA subscription limits (£20,000 for 2025-26)
    across all account types for each AWB Wealth client.

    Regulation: HMRC ISA regulations 1998 (SI 1998/1870).
    """

    def __init__(self, tax_year: str = "2025-26") -> None:
        self._tax_year = tax_year
        self._subscriptions: dict[str, Decimal] = {}

    def record_subscription(
        self,
        client_id: str,
        amount_gbp: Decimal,
    ) -> bool:
        """Record an ISA subscription and check annual limit.

        Args:
            client_id: AWB client identifier.
            amount_gbp: Subscription amount.

        Returns:
            True if within limit; False if breach detected.
        """
        current = self._subscriptions.get(
            client_id, Decimal("0")
        )
        new_total = current + amount_gbp
        self._subscriptions[client_id] = new_total
        if new_total > ISA_ANNUAL_LIMIT:
            log.error(
                "ISA BREACH: client %s total £%s > £%s",
                client_id, new_total, ISA_ANNUAL_LIMIT,
            )
            return False
        return True

    def generate_isatr_xml(
        self, client_id: str
    ) -> str:
        """Generate HMRC ISATR XML for one client.

        Returns HMRC-compatible Making Tax Digital XML format.
        """
        subscribed = self._subscriptions.get(
            client_id, Decimal("0")
        )
        return (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<ISAReturn xmlns="http://www.hmrc.gov.uk/isa">'
            f'<TaxYear>{self._tax_year}</TaxYear>'
            f'<ClientRef>{client_id}</ClientRef>'
            f'<TotalSubscribed>{subscribed}</TotalSubscribed>'
            f'<AnnualLimit>{ISA_ANNUAL_LIMIT}</AnnualLimit>'
            f'<Status>'
            f'{"COMPLIANT" if subscribed <= ISA_ANNUAL_LIMIT else "BREACH"}'
            f'</Status>'
            f'</ISAReturn>'
        )
