"""Exercise 6.1 — Add an amendment-aware covenant alert to the CIM.

Difficulty: 3/5 | Estimated time: 45 minutes

Extend the Covenant Compliance Analyser so every covenant check resolves
the LATEST amended facility letter before testing thresholds — the failure
mode from this chapter's war story. Your alert must fire only on breaches
under the amended terms and cite the governing letter version.

Success criterion: on the 5-facility test set below, exactly ONE genuine
DSCR breach is flagged and ZERO false breaches from superseded letters.

Solution: solutions/covenant_alert_solution.py
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FacilityLetter:
    facility_id: str
    version: int
    dscr_covenant: float  # minimum DSCR under this letter


@dataclass(frozen=True)
class CovenantTest:
    facility_id: str
    measured_dscr: float


# Superseded and amended letters deliberately mixed, as in production.
LETTERS = [
    FacilityLetter("FAC-1001", 1, 1.40),
    FacilityLetter("FAC-1001", 2, 1.10),  # amended DOWN — v1 is superseded
    FacilityLetter("FAC-1002", 1, 1.25),
    FacilityLetter("FAC-1003", 1, 1.25),
    FacilityLetter("FAC-1003", 2, 1.05),  # amended down
    FacilityLetter("FAC-1004", 1, 1.20),
    FacilityLetter("FAC-1005", 1, 1.30),
    FacilityLetter("FAC-1005", 2, 1.50),  # amended UP — tighter covenant
]

TESTS = [
    CovenantTest("FAC-1001", 1.18),  # breach under v1 (1.40) but NOT v2 (1.10)
    CovenantTest("FAC-1002", 1.31),  # compliant
    CovenantTest("FAC-1003", 1.12),  # breach under v1 but NOT v2
    CovenantTest("FAC-1004", 1.27),  # compliant
    CovenantTest("FAC-1005", 1.42),  # GENUINE breach under amended v2 (1.50)
]


@dataclass(frozen=True)
class CovenantAlert:
    facility_id: str
    measured_dscr: float
    covenant: float
    letter_version: int


def latest_letter(facility_id: str) -> FacilityLetter:
    """TODO: return the highest-version letter for the facility."""
    raise NotImplementedError("Exercise 6.1")


def run_covenant_checks() -> list[CovenantAlert]:
    """TODO: test each facility against its LATEST letter only."""
    raise NotImplementedError("Exercise 6.1")


if __name__ == "__main__":
    alerts = run_covenant_checks()
    for a in alerts:
        print(
            f"ALERT {a.facility_id}: DSCR {a.measured_dscr:.2f} < "
            f"{a.covenant:.2f} (letter v{a.letter_version})"
        )
    assert len(alerts) == 1 and alerts[0].facility_id == "FAC-1005"
    print("Success criterion met: 1 genuine breach, 0 false breaches")
