"""Solution — Exercise 3.1: cost circuit breaker for the Treasury Agent."""
from __future__ import annotations

from dataclasses import dataclass, field


class AgentBudgetExceeded(Exception):
    pass


@dataclass
class T24BalanceTool:
    calls: int = 0

    def run(self, account_id: str) -> dict:
        self.calls += 1
        cost_gbp = 0.80 * (2 ** (self.calls - 1))
        return {
            "account_id": account_id,
            "balance_gbp": 1_250_000.00,
            "call_cost_gbp": cost_gbp,
        }

    def next_cost(self) -> float:
        return 0.80 * (2 ** self.calls)


@dataclass
class AgentRunBudget:
    max_cost_gbp: float = 5.00
    spent_gbp: float = 0.0

    def record(self, cost_gbp: float) -> None:
        self.spent_gbp += cost_gbp

    def check(self, next_cost_gbp: float) -> None:
        projected = self.spent_gbp + next_cost_gbp
        if projected > self.max_cost_gbp:
            raise AgentBudgetExceeded(
                f"AgentBudgetExceeded: cost £{projected:.2f} "
                f"exceeds limit £{self.max_cost_gbp:.2f}"
            )


@dataclass
class TreasuryAgentOrchestrator:
    tool: T24BalanceTool = field(default_factory=T24BalanceTool)
    budget: AgentRunBudget = field(default_factory=AgentRunBudget)

    def run(self, account_id: str, max_steps: int = 10) -> None:
        for _step in range(max_steps):
            self.budget.check(self.tool.next_cost())
            result = self.tool.run(account_id)
            self.budget.record(result["call_cost_gbp"])
            print(
                f"step {self.tool.calls}: cost £{result['call_cost_gbp']:.2f} "
                f"(total £{self.budget.spent_gbp:.2f})"
            )


if __name__ == "__main__":
    agent = TreasuryAgentOrchestrator()
    try:
        agent.run("GB29AWBK60161331926819")
    except AgentBudgetExceeded as exc:
        print(exc)  # halts on call 3: 0.80 + 1.60 + projected 3.20 = £5.60
