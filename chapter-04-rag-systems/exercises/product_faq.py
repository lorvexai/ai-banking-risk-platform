"""
exercises/product_faq.py
Exercise 4.1: Build a ChromaDB-backed Product FAQ

Exercise: Build a ChromaDB-backed FAQ for a fictional savings product range.
Difficulty: ★★★☆☆ | Estimated time: 30 minutes

Task:
  1. Ingest 10 fictional savings product documents into ChromaDB
  2. Implement semantic search with metadata eligibility filtering
  3. Enforce eligibility constraints BEFORE generation (not after)
  4. Verify: top-3 results always relevant (faithfulness >= 0.80)

Starter code:
  - SAMPLE_PRODUCTS list provided below — use as your document corpus
  - Complete the TODO sections to build the pipeline
  - Run the test queries at the bottom to verify your implementation

Success criterion: top-3 results always relevant for the 20 test queries
in exercises/test_queries.json (faithfulness >= 0.80 on each).

Solution: github.com/lorvenio/ai-banking-risk-platform/chapter_04/solutions/

Regulatory note:
  FCA PS22/9 (Consumer Duty): product recommendations must evidence
  good customer outcomes. Eligibility must be enforced BEFORE generation —
  a metadata filter on risk_category and min_investment is mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# ── Sample product documents ──────────────────────────────────────────────────

@dataclass
class AWBProduct:
    """A fictional AWB savings product."""
    product_id:   str
    name:         str
    description:  str
    risk_category: str   # "LOW" | "MEDIUM" | "HIGH"
    min_investment: int  # GBP minimum
    annual_rate:  float  # % AER


SAMPLE_PRODUCTS: List[AWBProduct] = [
    AWBProduct("AWB-ISA-001", "AWB Instant Access ISA",
               "Tax-free instant access savings. Withdraw anytime. "
               "FSCS protected up to £85,000.",
               risk_category="LOW", min_investment=1, annual_rate=4.25),
    AWBProduct("AWB-FIX-001", "AWB 1-Year Fixed Rate Bond",
               "Lock in your rate for 12 months. Higher return than "
               "instant access. FSCS protected.",
               risk_category="LOW", min_investment=1000, annual_rate=4.85),
    AWBProduct("AWB-FIX-003", "AWB 3-Year Fixed Rate Bond",
               "Fixed rate for 36 months. Ideal for medium-term savings "
               "goals. FSCS protected.",
               risk_category="LOW", min_investment=5000, annual_rate=5.10),
    AWBProduct("AWB-JISA-001", "AWB Junior ISA",
               "Tax-free savings for under-18s. Parent/guardian opens "
               "account. Funds locked until child turns 18.",
               risk_category="LOW", min_investment=1, annual_rate=4.50),
    AWBProduct("AWB-REG-001", "AWB Regular Saver",
               "Save between £25 and £500 per month. Fixed rate for "
               "12 months. Direct debit setup required.",
               risk_category="LOW", min_investment=25, annual_rate=5.50),
    AWBProduct("AWB-STK-001", "AWB Stocks and Shares ISA",
               "Invest in global equity funds within your ISA allowance. "
               "Capital at risk. Not FSCS protected for investment losses.",
               risk_category="MEDIUM", min_investment=500, annual_rate=0.0),
    AWBProduct("AWB-GIA-001", "AWB General Investment Account",
               "Invest with no annual limit. Suitable for investors who "
               "have used their ISA allowance. Capital at risk.",
               risk_category="MEDIUM", min_investment=1000, annual_rate=0.0),
    AWBProduct("AWB-SIPP-001", "AWB Self-Invested Personal Pension",
               "Invest for retirement with tax relief on contributions. "
               "Wide fund choice. Capital at risk.",
               risk_category="MEDIUM", min_investment=5000, annual_rate=0.0),
    AWBProduct("AWB-BOND-001", "AWB Corporate Bond Fund",
               "Fixed income exposure via diversified UK corporate bonds. "
               "Moderate credit risk. Monthly income distribution.",
               risk_category="MEDIUM", min_investment=10000, annual_rate=0.0),
    AWBProduct("AWB-ALT-001", "AWB Alternative Assets Fund",
               "Exposure to infrastructure, private equity, and real assets. "
               "Suitable only for sophisticated or high-net-worth investors.",
               risk_category="HIGH", min_investment=25000, annual_rate=0.0),
]


# ── Customer profile ──────────────────────────────────────────────────────────

@dataclass
class CustomerProfile:
    """A fictional AWB customer seeking product recommendations."""
    customer_id:       str
    risk_tolerance:    str   # "LOW" | "MEDIUM" | "HIGH"
    available_capital: int   # GBP available to invest


# ── TODO: Implement the following functions ───────────────────────────────────

def build_product_corpus(
    products: List[AWBProduct],
    collection_name: str = "awb_product_faq",
) -> object:
    """
    TODO: Ingest product documents into ChromaDB.

    Steps:
      1. Initialise a ChromaDB client (in-memory for exercise)
      2. Create a collection named collection_name
      3. For each product, add:
         - id:        product.product_id
         - document:  product.name + " " + product.description
         - metadata:  {
               "risk_category":  product.risk_category,
               "min_investment": product.min_investment,
               "annual_rate":    product.annual_rate,
           }
      4. Return the collection object

    Hint:
      import chromadb
      client = chromadb.Client()   # in-memory
      collection = client.create_collection(collection_name)
      collection.add(ids=[...], documents=[...], metadatas=[...])
    """
    raise NotImplementedError("TODO: implement build_product_corpus()")


def recommend_products(
    query:      str,
    customer:   CustomerProfile,
    collection: object,
    top_k:      int = 3,
) -> List[dict]:
    """
    TODO: Retrieve relevant products for a customer query.

    IMPORTANT: Eligibility filter MUST be applied BEFORE retrieval —
    not as a post-generation guardrail. This is the FCA PS22/9 pattern.

    Steps:
      1. Build eligibility filter:
         - risk_category: only equal to or lower risk than customer
         - min_investment: only products the customer can afford
      2. Query collection with semantic search + eligibility filter
      3. Return top_k results as list of dicts with product details

    Hint:
      results = collection.query(
          query_texts=[query],
          n_results=top_k,
          where={
              "risk_category": customer.risk_tolerance,  # simplify: exact match
              "min_investment": {"$lte": customer.available_capital},
          }
      )
    """
    raise NotImplementedError("TODO: implement recommend_products()")


# ── Test runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Chapter 4 — Exercise 4.1: Product FAQ")
    print("=" * 50)

    # Build corpus
    try:
        collection = build_product_corpus(SAMPLE_PRODUCTS)
        print(f"✅ Ingested {len(SAMPLE_PRODUCTS)} products into ChromaDB")
    except NotImplementedError:
        print("❌ build_product_corpus() not yet implemented")
        exit(1)

    # Test queries
    test_cases = [
        ("I want a safe place to save £500", CustomerProfile("C001", "LOW", 500)),
        ("Best rate for £10,000 I won't need for 3 years",
         CustomerProfile("C002", "LOW", 10000)),
        ("I want to invest for retirement", CustomerProfile("C003", "MEDIUM", 5000)),
    ]

    for query, customer in test_cases:
        print(f"\nQuery: '{query}'")
        print(f"Customer: risk={customer.risk_tolerance}, "
              f"capital=£{customer.available_capital:,}")
        try:
            results = recommend_products(query, customer, collection)
            print(f"Top {len(results)} recommendations:")
            for r in results:
                print(f"  • {r}")
        except NotImplementedError:
            print("  ❌ recommend_products() not yet implemented")
            break

    print("\nComplete! Compare your results with the solution at:")
    print("github.com/lorvenio/ai-banking-risk-platform/chapter_04/solutions/")
