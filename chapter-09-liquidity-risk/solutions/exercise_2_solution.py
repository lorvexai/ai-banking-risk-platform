"""Solution — Exercise 9.2: quarter-end intraday liquidity stress simulation.

Connects the Cash Flow Forecaster to the Intraday Liquidity Monitor for a
simulated quarter-end settlement day with a 15:30 RED-threshold peak.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

LIMIT_GBP = 5_470_000_000.0  # intraday facility
THRESHOLDS = [(0.95, "CRITICAL"), (0.85, "RED"), (0.70, "AMBER")]


@dataclass(frozen=True)
class Snapshot:
    time: str
    net_position_gbp: float

    @property
    def utilisation(self) -> float:
        return self.net_position_gbp / LIMIT_GBP


SNAPSHOTS = [
    Snapshot("10:00", 3_100_000_000.0),   # 56% — normal
    Snapshot("13:00", 4_100_000_000.0),   # 75% — AMBER
    Snapshot("15:30", 4_700_000_000.0),   # 85% — RED
]


def forecast_30d(seed: int = 9) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(250e6, 120e6) for _ in range(30)]


def buffer_breaches(forecast: list[float], buffer_gbp: float = 400e6) -> list[int]:
    running, breaches = 0.0, []
    for day, flow in enumerate(forecast, 1):
        running += flow
        if running < -buffer_gbp:
            breaches.append(day)
    return breaches


def classify(snapshot: Snapshot) -> str | None:
    for level, name in THRESHOLDS:
        if snapshot.utilisation >= level:
            return name
    return None


def run(snapshots: list[Snapshot]) -> None:
    print("30-day cash flow forecast generated "
          f"({len(buffer_breaches(forecast_30d()))} buffer breach days)")
    for s in snapshots:
        alert = classify(s)
        if alert:
            action = {"AMBER": "notify treasury desk",
                      "RED": "activate contingency funding plan",
                      "CRITICAL": "draw committed facility + notify PRA"}[alert]
            print(f"{s.time}  £{s.net_position_gbp/1e9:.1f}B "
                  f"({s.utilisation:.0%})  {alert}: {action}")


if __name__ == "__main__":
    run(SNAPSHOTS)
    print("\nBonus — pushing 15:30 peak to £5.2B (95%):")
    run([Snapshot("15:30", 5_200_000_000.0)])
