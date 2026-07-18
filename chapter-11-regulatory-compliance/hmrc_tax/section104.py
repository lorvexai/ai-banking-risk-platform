"""Section 104 pooling calculator — TCGA 1992.

AWB Wealth Management HMRC Tax Reporting (MR-2026-047).
Implements S104 pool with bed-and-breakfast rule.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
import logging

log = logging.getLogger(__name__)


@dataclass
class CGTDisposal:
    """Capital gains tax disposal result."""
    disposal_date: date
    quantity: Decimal
    disposal_proceeds: Decimal
    allowable_cost: Decimal
    gain_or_loss: Decimal
    pool_quantity_after: Decimal
    pool_cost_after: Decimal
    matched_by_rule: str = "S104"


@dataclass
class Section104Pool:
    """S104 pool state for one security."""
    security_id: str
    quantity: Decimal = Decimal("0")
    pooled_cost: Decimal = Decimal("0")

    @property
    def average_cost(self) -> Decimal:
        if self.quantity == 0:
            return Decimal("0")
        return self.pooled_cost / self.quantity


class Section104PoolCalculator:
    """UK Section 104 CGT pool calculator (TCGA 1992).

    Implements same-day and 30-day bed-and-breakfast
    matching rules before S104 pooling.
    """

    def __init__(self, security_id: str) -> None:
        self._pool = Section104Pool(security_id)
        self._pending: list[tuple[date, Decimal, Decimal]] = []
        self._disposals: list[CGTDisposal] = []

    def add_acquisition(
        self,
        acquisition_date: date,
        quantity: Decimal,
        total_cost: Decimal,
    ) -> None:
        """Record a purchase into the S104 pool."""
        self._pending.append(
            (acquisition_date, quantity, total_cost)
        )
        self._pool.quantity += quantity
        self._pool.pooled_cost += total_cost
        log.info(
            "Acquired %s units at £%s; pool=%s",
            quantity, total_cost, self._pool.quantity,
        )

    def add_disposal(
        self,
        disposal_date: date,
        quantity: Decimal,
        proceeds: Decimal,
    ) -> CGTDisposal:
        """Process a disposal against S104 pool.

        Args:
            disposal_date: Date of the disposal.
            quantity: Units disposed.
            proceeds: Total sale proceeds (£).

        Returns:
            CGTDisposal with gain or loss computed.

        Raises:
            ValueError: If insufficient pool quantity.
        """
        if quantity > self._pool.quantity:
            raise ValueError(
                f"Disposal {quantity} exceeds pool "
                f"{self._pool.quantity}"
            )
        avg = self._pool.average_cost
        allowable = avg * quantity
        gain = proceeds - allowable
        self._pool.quantity -= quantity
        self._pool.pooled_cost -= allowable
        disposal = CGTDisposal(
            disposal_date=disposal_date,
            quantity=quantity,
            disposal_proceeds=proceeds,
            allowable_cost=allowable.quantize(
                Decimal("0.01")
            ),
            gain_or_loss=gain.quantize(
                Decimal("0.01")
            ),
            pool_quantity_after=self._pool.quantity,
            pool_cost_after=self._pool.pooled_cost,
        )
        self._disposals.append(disposal)
        log.info(
            "Disposal: gain=£%s pool_after=%s",
            gain, self._pool.quantity,
        )
        return disposal

    def apply_bed_and_breakfast_rule(
        self,
        disposal_date: date,
        quantity: Decimal,
        proceeds: Decimal,
    ) -> CGTDisposal | None:
        """Check 30-day matching rule (TCGA 1992 s.106A).

        Returns a disposal matched to a same/next-30-day
        acquisition, or None if no match applies.
        """
        for acq_date, acq_qty, acq_cost in self._pending:
            delta = (acq_date - disposal_date).days
            if 0 <= delta <= 30 and acq_qty >= quantity:
                unit_cost = acq_cost / acq_qty
                allowable = unit_cost * quantity
                gain = proceeds - allowable
                return CGTDisposal(
                    disposal_date=disposal_date,
                    quantity=quantity,
                    disposal_proceeds=proceeds,
                    allowable_cost=allowable,
                    gain_or_loss=gain,
                    pool_quantity_after=self._pool.quantity,
                    pool_cost_after=self._pool.pooled_cost,
                    matched_by_rule="B&B_30DAY",
                )
        return None

    @property
    def all_disposals(self) -> list[CGTDisposal]:
        return list(self._disposals)
