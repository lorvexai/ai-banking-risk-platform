"""
Exercise 12.1: Build a Transaction Network and Run Louvain Detection
Difficulty: 4/5 | Estimated time: 45 minutes

Task: Using NetworkX on 500 synthetic transactions, construct a directed
transaction graph and run Louvain community detection. Identify 3 structuring
rings planted in the synthetic data.

Success criterion: all 3 rings have modularity > 0.4 and are surfaced by
AMLNetworkAnalyser.identify_structuring_pattern().

Starter: This file
Solution: github.com/lorvenio/ai-banking-risk-platform/chapter_12/solutions/
"""
from __future__ import annotations

import random
from decimal import Decimal
from typing import List, Tuple

import networkx as nx


def generate_synthetic_transactions(
    num_accounts: int = 100,
    num_transactions: int = 500,
    num_rings: int = 3,
    seed: int = 42,
) -> Tuple[List[dict], List[List[str]]]:
    """Generate synthetic transactions with planted structuring rings.

    Args:
        num_accounts: Total number of customer accounts.
        num_transactions: Total number of transactions to generate.
        num_rings: Number of structuring rings to plant.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (transactions, ring_account_groups).
    """
    random.seed(seed)
    accounts = [f"ACC{i:04d}" for i in range(num_accounts)]

    transactions: List[dict] = []
    ring_groups: List[List[str]] = []

    # TODO: Plant structuring rings
    # Each ring should have 5-10 accounts all sending sub-£5,000
    # amounts to a single beneficiary account
    # Hint: Use accounts[0:10] as ring 1 beneficiary etc.

    # TODO: Generate background legitimate transactions

    return transactions, ring_groups


def build_transaction_graph(
    transactions: List[dict],
) -> nx.DiGraph:
    """Build a directed NetworkX graph from transactions.

    Args:
        transactions: List of transaction dicts with keys:
            sender_id, receiver_id, amount_gbp.

    Returns:
        Directed graph with edge weights = total amount.
    """
    graph = nx.DiGraph()
    # TODO: Add nodes and weighted edges
    return graph


def detect_structuring_rings(
    graph: nx.DiGraph,
    amount_threshold: Decimal = Decimal("5000"),
    min_contributors: int = 3,
) -> List[dict]:
    """Detect structuring rings using Louvain community detection.

    Args:
        graph: Transaction graph from build_transaction_graph().
        amount_threshold: Maximum individual amount for structuring flag.
        min_contributors: Minimum accounts feeding a single beneficiary.

    Returns:
        List of detected ring summaries.
    """
    # TODO: Run Louvain on undirected projection of graph
    # import community as community_louvain
    # partition = community_louvain.best_partition(graph.to_undirected())
    # TODO: For each community, check for structuring pattern
    rings: List[dict] = []
    return rings


if __name__ == "__main__":
    transactions, planted_rings = generate_synthetic_transactions()
    print(f"Generated {len(transactions)} transactions with {len(planted_rings)} planted rings")

    graph = build_transaction_graph(transactions)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    detected = detect_structuring_rings(graph)
    print(f"Detected {len(detected)} structuring rings")
    for ring in detected:
        print(f"  Ring: {ring}")
