"""Exercise 3.1 — Add a cost circuit breaker to the Treasury Agent.

Difficulty: 3/5 | Estimated time: 30 minutes

Task: implement AgentRunBudget with max_cost_gbp=5.00 on the
TreasuryAgentOrchestrator below. The mock T24BalanceTool simulates
exponential cost increases; your implementation must stop the agent
BEFORE it exceeds the budget and log the threshold breach.

Success criterion: the agent halts within 3 tool calls and prints
    "AgentBudgetExceeded: cost £X.XX exceeds limit £5.00"

Solution: solutions/circuit_breaker_solution.py
"""
from __future__ import annotations

from dataclasses import dataclass, field


class AgentBudgetExceeded(Exception):
    """Raised when a run's cumulative cost would exceed its budget."""


@dataclass
class T24BalanceTool:
    """Mock T24 core-banking balance tool with exponentially growing cost."""

    calls: int = 0

    def run(self, account_id: str) -> dict:
        self.calls += 1
        cost_gbp = 0.80 * (2 ** (self.calls - 1))  # 0.80, 1.60, 3.20, 6.40 ...
        return {
            "account_id": account_id,
            "balance_gbp": 1_250_000.00,
            "call_cost_gbp": cost_gbp,
        }


@dataclass
class AgentRunBudget:
    """TODO: enforce a hard cost cap for a single agent run.

    Implement:
      * record(cost_gbp)  – accumulate spend
      * check(next_cost_gbp) – raise AgentBudgetExceeded if the NEXT
        call would take the total over max_cost_gbp
    """

    max_cost_gbp: float = 5.00
    spent_gbp: float = 0.0

    def record(self, cost_gbp: float) -> None:
        raise NotImplementedError("Exercise 3.1")

    def check(self, next_cost_gbp: float) -> None:
        raise NotImplementedError("Exercise 3.1")


@dataclass
class TreasuryAgentOrchestrator:
    """Simplified treasury agent loop. Wire the budget into run()."""

    tool: T24BalanceTool = field(default_factory=T24BalanceTool)
    budget: AgentRunBudget = field(default_factory=AgentRunBudget)

    def run(self, account_id: str, max_steps: int = 10) -> None:
        for _step in range(max_steps):
            # TODO: estimate the next call cost (mirror the tool's cost
            # schedule), call self.budget.check(...) BEFORE invoking the
            # tool, then self.budget.record(...) after.
            result = self.tool.run(account_id)
            print(f"step cost £{result['call_cost_gbp']:.2f}")


if __name__ == "__main__":
    agent = TreasuryAgentOrchestrator()
    try:
        agent.run("GB29AWBK60161331926819")
    except AgentBudgetExceeded as exc:
        print(exc)
