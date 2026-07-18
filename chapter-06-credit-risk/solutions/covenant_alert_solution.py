"""Solution — Exercise 6.1: amendment-aware covenant alert."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from exercises.covenant_alert import (  # noqa: E402
    LETTERS,
    TESTS,
    CovenantAlert,
    FacilityLetter,
)


def latest_letter(facility_id: str) -> FacilityLetter:
    """Resolve the governing (highest-version) letter for a facility."""
    candidates = [l for l in LETTERS if l.facility_id == facility_id]
    return max(candidates, key=lambda l: l.version)


def run_covenant_checks() -> list[CovenantAlert]:
    """Test each facility against its LATEST amended letter only."""
    alerts: list[CovenantAlert] = []
    for test in TESTS:
        letter = latest_letter(test.facility_id)
        if test.measured_dscr < letter.dscr_covenant:
            alerts.append(
                CovenantAlert(
                    facility_id=test.facility_id,
                    measured_dscr=test.measured_dscr,
                    covenant=letter.dscr_covenant,
                    letter_version=letter.version,
                )
            )
    return alerts


if __name__ == "__main__":
    alerts = run_covenant_checks()
    for a in alerts:
        print(
            f"ALERT {a.facility_id}: DSCR {a.measured_dscr:.2f} < "
            f"{a.covenant:.2f} (letter v{a.letter_version})"
        )
    assert len(alerts) == 1 and alerts[0].facility_id == "FAC-1005"
    print("Success criterion met: 1 genuine breach, 0 false breaches")
