"""
data/generate_sample_credit_pack.py
AWB Credit Document Analyser — Sample Credit Pack Generator

Generates realistic UK SME credit pack text for testing and development.
All company names, financials, and people are entirely fictional.

Includes deliberate edge cases:
  - ABC Manufacturing Ltd: healthy company (baseline)
  - Riverside Retail Holdings: revenue decline + weak interest cover
  - Northgate Properties: negative equity + covenant breach
  - Summit Digital Services: missing EBITDA (incomplete accounts)

Run: python data/generate_sample_credit_pack.py
Output: data/*.txt files

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
"""

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Pack 1: ABC Manufacturing Ltd — healthy baseline
# ---------------------------------------------------------------------------

ABC_MANUFACTURING = """
[PAGE 1]

CREDIT PACK — CONFIDENTIAL
Avon & Wessex Bank plc — Corporate Lending Division
Prepared by: AWB Relationship Management Team
Date: 15 January 2026

BORROWER: ABC Manufacturing Ltd
Company Registration: 04521987
Registered Address: 14 Industrial Park, Peterborough, PE1 2AA
Sector: Manufacturing — Precision Engineering Components

FACILITY REQUEST
Revolving Credit Facility: £5,000,000
Term Loan: £3,000,000
Total Facility: £8,000,000
Purpose: Refinance existing facilities and fund capital investment programme

CREDIT ANALYST: J. Thompson
RELATIONSHIP MANAGER: S. Patel

[PAGE 2]

FINANCIAL SUMMARY — ABC MANUFACTURING LTD

Year ended 31 December 2024 (Audited)
Prior Year: Year ended 31 December 2023 (Audited)

PROFIT AND LOSS ACCOUNT                     2024        2023
                                          (£000s)    (£000s)
Revenue                                    42,350      38,900
Cost of Sales                            (28,350)    (26,100)
Gross Profit                               14,000      12,800
Administrative Expenses                    (6,200)     (5,800)
EBITDA                                      7,800       7,000
Depreciation and Amortisation              (1,200)     (1,100)
EBIT                                        6,600       5,900
Net Finance Costs (Interest Expense)         (650)       (580)
Profit Before Tax                           5,950       5,320
Taxation                                    (893)       (798)
Profit After Tax                            5,057       4,522

EBITDA Margin: 18.4%
Year-on-Year Revenue Growth: 8.9%

[PAGE 3]

BALANCE SHEET — ABC MANUFACTURING LTD

As at 31 December 2024                    2024        2023
                                        (£000s)     (£000s)
FIXED ASSETS
Property, Plant and Equipment             8,200       7,400
Intangible Assets                           500         600
Total Fixed Assets                        8,700       8,000

CURRENT ASSETS
Inventory                                 4,100       3,800
Trade Debtors                             5,200       4,700
Cash and Cash Equivalents                 1,850       1,200
Other Current Assets                        450         350
Total Current Assets                     11,600      10,050

TOTAL ASSETS                             20,300      18,050

CURRENT LIABILITIES
Trade Creditors                           3,200       2,900
Bank Overdraft                              200         300
Current Portion of Term Loan              1,000       1,000
Other Current Liabilities                   850         750
Total Current Liabilities                 5,250       4,950

NON-CURRENT LIABILITIES
Term Loan (non-current)                   4,000       5,000
Finance Lease Obligations                   800         950
Total Non-Current Liabilities             4,800       5,950

TOTAL LIABILITIES                        10,050      10,900

NET ASSETS / EQUITY                      10,250       7,150

KEY FINANCIAL RATIOS
Leverage Ratio (Net Debt/EBITDA):          0.54x      0.86x
Net Debt: £4,200k (Term Loan £5,000k + Finance Leases £800k - Cash £1,850k + Overdraft £200k)
Wait — Net Debt = Gross Debt minus Cash = (5,000 + 800 + 200) - 1,850 = 4,150
Revised Net Debt: £4,150k
Leverage Ratio: 4,150 / 7,800 = 0.53x

Interest Cover (EBITDA/Net Finance Costs): 12.0x
Current Ratio (Current Assets/Current Liabilities): 2.21x

COVENANT COMPLIANCE (existing RCF)
Maximum Leverage: 3.0x — COMPLIANT (actual 0.53x)
Minimum Interest Cover: 3.5x — COMPLIANT (actual 12.0x)
"""

# ---------------------------------------------------------------------------
# Pack 2: Riverside Retail Holdings — weak interest cover, revenue decline
# ---------------------------------------------------------------------------

RIVERSIDE_RETAIL = """
[PAGE 1]

CREDIT PACK — CONFIDENTIAL
Avon & Wessex Bank plc — Corporate Lending Division
Date: 20 January 2026

BORROWER: Riverside Retail Holdings Ltd
Company Registration: 07834512
Registered Address: Unit 5, Riverside Business Park, Ely, CB7 4DT
Sector: Retail — Mid-market fashion and homewares

FACILITY REQUEST
Revolving Credit Facility renewal: £3,500,000
Purpose: Working capital and seasonal inventory financing

[PAGE 2]

FINANCIAL SUMMARY — RIVERSIDE RETAIL HOLDINGS LTD

Year ended 30 September 2024 (Audited)

PROFIT AND LOSS                           2024        2023
                                        (£000s)     (£000s)
Revenue                                  18,200      23,500
Cost of Sales                           (13,600)    (17,200)
Gross Profit                              4,600       6,300
Operating Expenses                       (3,950)     (4,100)
EBITDA                                      650       2,200
Depreciation                               (450)       (480)
EBIT                                        200       1,720
Interest Expense                           (620)       (510)
Profit/(Loss) Before Tax                   (420)      1,210

EBITDA Margin: 3.6%
Revenue Decline YoY: -22.6%

[PAGE 3]

BALANCE SHEET — RIVERSIDE RETAIL HOLDINGS

As at 30 September 2024               2024        2023
                                    (£000s)     (£000s)
Fixed Assets                          3,200       3,600
Current Assets
  Inventory                           4,100       3,900
  Trade Debtors                         850         920
  Cash                                  280         640
Total Current Assets                  5,230       5,460

Total Assets                          8,430       9,060

Current Liabilities
  Trade Creditors                     3,800       2,900
  Bank Overdraft                      1,200         600
  Current Loans                         800         800
Total Current Liabilities             5,800       4,300

Non-Current Loans                     2,800       3,600

Total Liabilities                     8,600       7,900
Net Assets / (Deficit)                 (170)       1,160

RATIOS
Net Debt: Overdraft £1,200k + Loans £3,600k - Cash £280k = £4,520k
Leverage Ratio: 4,520 / 650 = 6.95x
Interest Cover: 650 / 620 = 1.05x
Current Ratio: 5,230 / 5,800 = 0.90x

COVENANT STATUS
Maximum Leverage 4.0x: BREACH (actual 6.95x)
Minimum Interest Cover 2.0x: BREACH (actual 1.05x)
"""

# ---------------------------------------------------------------------------
# Pack 3: Summit Digital Services — missing EBITDA (incomplete accounts)
# ---------------------------------------------------------------------------

SUMMIT_DIGITAL = """
[PAGE 1]

CREDIT PACK — DRAFT (Management Accounts — Unaudited)
Avon & Wessex Bank plc
Date: 5 June 2026

BORROWER: Summit Digital Services Ltd
Company Registration: 10234567
Registered Address: 12 Science Park, Cambridge, CB4 0WA
Sector: Technology — SaaS and digital transformation consulting

FACILITY REQUEST
Term Loan: £1,500,000
Purpose: Product development and headcount investment

[PAGE 2]

FINANCIAL SUMMARY — SUMMIT DIGITAL SERVICES

Period: 9 months ended 30 September 2025 (Management Accounts — unaudited)
Note: Full-year audited accounts not yet available (audit in progress).

INCOME STATEMENT (9 months)             £000s
Revenue                                 4,200
Direct Costs                           (2,800)
Gross Profit                            1,400

Note: EBITDA not separately disclosed in management accounts.
Operating costs include £180k depreciation (estimated) and £240k
share-based payments. Management indicated adjusted EBITDA of
approximately £350k-£450k but formal calculation not provided.

BALANCE SHEET (as at 30 Sep 2025)      £000s
Fixed Assets (net)                        920
Current Assets
  Debtors                               1,100
  Cash                                    450
Total Current Assets                    1,550

Bank Loan                               1,200
Trade Creditors                           680
Other Creditors                           320
Total Liabilities                       2,200

Net Assets/(Deficit)                      270

Note: No formal leverage or interest cover ratios provided.
Cash burn rate approximately £85k/month (management estimate).
"""

# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------

def generate_all() -> None:
    packs = {
        "abc_manufacturing_credit_pack.txt": ABC_MANUFACTURING,
        "riverside_retail_credit_pack.txt": RIVERSIDE_RETAIL,
        "summit_digital_credit_pack.txt": SUMMIT_DIGITAL,
    }

    for filename, content in packs.items():
        path = OUTPUT_DIR / filename
        path.write_text(content.strip(), encoding="utf-8")
        print(f"Generated: {path} ({len(content):,} chars)")

    print(f"\n{len(packs)} credit packs generated in {OUTPUT_DIR}")
    print("\nEdge cases:")
    print("  abc_manufacturing: healthy baseline — all ratios within policy")
    print("  riverside_retail:  revenue decline 22.6%, leverage 6.95x, interest cover 1.05x (all P1/P2 flags)")
    print("  summit_digital:    missing EBITDA — tests handling of incomplete accounts")


if __name__ == "__main__":
    generate_all()
