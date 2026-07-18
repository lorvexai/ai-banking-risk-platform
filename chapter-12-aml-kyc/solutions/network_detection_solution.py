"""Solution — Exercise 12.1: transaction network + Louvain ring detection.

Builds a directed graph from 500 synthetic transactions with 3 planted
structuring rings and surfaces them via community detection. Uses NetworkX
when available; falls back to a connected-component heuristic otherwise.
"""
from __future__ import annotations

import random
from collections import defaultdict

RING_SPECS = [  # (members, txn count) planted structuring rings
    ([f"R1-{i}" for i in range(6)], 40),
    ([f"R2-{i}" for i in range(5)], 32),
    ([f"R3-{i}" for i in range(7)], 45),
]


def make_transactions(seed: int = 12) -> list[tuple[str, str, float]]:
    rng = random.Random(seed)
    txns = []
    for members, count in RING_SPECS:  # dense in-ring flows under £10k
        for _ in range(count):
            a, b = rng.sample(members, 2)
            txns.append((a, b, rng.uniform(8_000, 9_900)))
    others = [f"C-{i}" for i in range(120)]
    while len(txns) < 500:  # sparse background noise
        a, b = rng.sample(others, 2)
        txns.append((a, b, rng.uniform(50, 250_000)))
    rng.shuffle(txns)
    return txns


class AMLNetworkAnalyser:
    def __init__(self, txns: list[tuple[str, str, float]]):
        self.txns = txns

    def identify_structuring_pattern(self) -> list[set[str]]:
        try:
            import networkx as nx

            g = nx.DiGraph()
            for a, b, amt in self.txns:
                if amt < 10_000:  # structuring band
                    g.add_edge(a, b, weight=amt)
            comms = nx.community.louvain_communities(g.to_undirected(), seed=12)
            mod = nx.community.modularity(g.to_undirected(), comms)
            print(f"louvain modularity: {mod:.3f} (target > 0.4)")
            return [c for c in comms if len(c) >= 4]
        except ImportError:  # heuristic fallback
            adj = defaultdict(set)
            for a, b, amt in self.txns:
                if amt < 10_000:
                    adj[a].add(b), adj[b].add(a)
            seen, rings = set(), []
            for node in adj:
                if node in seen:
                    continue
                stack, comp = [node], set()
                while stack:
                    n = stack.pop()
                    if n in comp:
                        continue
                    comp.add(n), seen.add(n)
                    stack.extend(adj[n] - comp)
                if len(comp) >= 4:
                    rings.append(comp)
            print("networkx unavailable — connected-component fallback used")
            return rings


if __name__ == "__main__":
    rings = AMLNetworkAnalyser(make_transactions()).identify_structuring_pattern()
    planted = [set(m) for m, _ in RING_SPECS]
    found = sum(1 for p in planted if any(p <= r for r in rings))
    print(f"rings surfaced: {len(rings)} | planted rings recovered: {found}/3")
    assert found == 3
    print("Success criterion met: all 3 structuring rings detected")
