"""
document_analyser/prompt_patterns.py
AWB Credit Document Analyser — Prompt Engineering Patterns

Implements the four core LLM prompt patterns used throughout the AWB
AI programme. Each pattern is a callable that returns a fully-formed
system or user prompt string ready for use with any Gemini 3 model.

These patterns are referenced in Chapter 2 Section 2.6.1 of:
  "AI for Financial Risk, Compliance and Regulatory Reporting"

Pattern taxonomy:
  PATTERN 1 — Role-Based System Prompt
    Sets the model's context, regulatory obligations, and behavioural
    boundaries before any user content is seen. Establishes the 4-component
    AWB prompt architecture: ROLE, REGULATORY CONSTRAINTS, OUTPUT FORMAT,
    EXPLICIT LIMITATIONS.

  PATTERN 2 — Chain-of-Thought (CoT) for Financial Ratio Analysis
    Forces the model to reason step-by-step through ratio calculations
    before committing to an output. Reduces silent hallucination on
    derived metrics (leverage, interest cover, DSCR).

  PATTERN 3 — Structured Output Contract (Pydantic-aligned)
    Instructs the model to return a JSON object matching an explicit
    schema. Combined with response_mime_type="application/json" in the
    Gemini API call, this eliminates markdown-wrapped responses and
    enables direct Pydantic validation.

  PATTERN 4 — Few-Shot Examples for Edge Cases
    Provides 2-3 worked examples in the prompt to anchor model behaviour
    on non-standard formats (foreign-currency accounts, consolidated vs
    standalone, abbreviated period-end reports).

Regulatory compliance:
  PRA SS1/23: Prompt templates are versioned and registered as part of
    model documentation for MR-2026-035. Changes to templates require
    re-validation against the 200-document benchmark set.
  EU AI Act Art. 13 (Transparency): Prompts are disclosed to regulators
    on request as part of HIGH-RISK system technical documentation.

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 2 — LLM Foundations for Financial Document Analysis
Version: 1.2.0  (June 2026)
"""
from __future__ import annotations

from datetime import date
from typing import Optional


# ---------------------------------------------------------------------------
# Version registry — required by PRA SS1/23 model documentation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_VERSION = "1.2.0"
PROMPT_TEMPLATE_MODEL_ID = "MR-2026-035"


# ---------------------------------------------------------------------------
# PATTERN 1 — Role-Based System Prompt
# ---------------------------------------------------------------------------

def build_role_system_prompt(
    institution_name: str = "Avon & Wessex Bank plc",
    model_id: str = PROMPT_TEMPLATE_MODEL_ID,
    today: Optional[str] = None,
) -> str:
    """
    Build a 4-component role-based system prompt for financial document
    extraction tasks.

    The 4-component structure:
      Component 1 — ROLE AND CONTEXT: Who the model is, what institution
        it serves, the current date, and what its output is used for.
      Component 2 — REGULATORY CONSTRAINTS: The specific obligations
        (PRA SS1/23, EU AI Act, UK GDPR) that govern every extraction.
      Component 3 — OUTPUT FORMAT: Exact JSON schema the model must follow.
        Paired with response_mime_type="application/json" at the API level.
      Component 4 — EXPLICIT LIMITATIONS: What the model cannot do and must
        not attempt. Prevents hallucination on data outside the document.

    Why this pattern works:
      Placing regulatory obligations in Component 2 (before any user content)
      means the model processes constraints at maximum attention weight.
      Empirically, this reduces fabricated figures by ~34% vs. embedding
      constraints only in the user turn (AWB evaluation set, January 2026).

    Args:
        institution_name: Bank name for role context.
        model_id:         PRA SS1/23 model registration ID.
        today:            ISO date string; defaults to current date.

    Returns:
        Formatted system prompt string for use as system_instruction
        in genai.GenerativeModel().
    """
    today = today or date.today().isoformat()

    return f"""COMPONENT 1 — ROLE AND CONTEXT:
You are a financial data extraction specialist for {institution_name},
a UK PRA/FCA-regulated bank. Today: {today}.
You extract structured financial data from corporate credit packs submitted
by borrowers seeking credit facilities. Your output feeds directly into the
credit assessment process and is subject to regulatory audit.

COMPONENT 2 — REGULATORY CONSTRAINTS:
- PRA SS1/23 Model {model_id}: every extraction is logged. Your outputs
  are audited by the Model Risk team. Accuracy is measured against ground
  truth on a 200-document validation set.
- EU AI Act 2024/1689 Annex III §5b HIGH-RISK: human oversight is mandatory
  before any output is used in a credit decision. Your role is extraction,
  not decision-making.
- UK GDPR Art. 6(1)(f): extract only the financial fields specified.
  Do not extract personal data (director names, addresses, NI numbers).
- EBA Guidelines on IRB: apply margin of conservatism. When uncertain
  between two values, use the more conservative interpretation
  (higher debt / lower income / lower coverage).
- Never fabricate values. If a figure is not present in the document,
  return null. A null is always preferable to an invented number.

COMPONENT 3 — OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown, no explanation, no preamble.
Schema is specified in the user message for each extraction task.

COMPONENT 4 — EXPLICIT LIMITATIONS:
- You do not have access to Companies House, HMRC, or Bloomberg data.
- You cannot verify figures against external databases.
- Ratios not stated in the document must be calculated from stated figures.
  Do not estimate ratios from industry benchmarks.
- If revenue is stated in a non-GBP currency, flag it — do not silently
  convert using assumed exchange rates.
- All monetary values must be in GBP thousands (£000s). If stated in
  millions, multiply by 1,000. If stated in billions, multiply by 1,000,000.
- Prompt template version: {PROMPT_TEMPLATE_VERSION}"""


# ---------------------------------------------------------------------------
# PATTERN 2 — Chain-of-Thought for Financial Ratio Analysis
# ---------------------------------------------------------------------------

def build_cot_ratio_prompt(
    revenue: Optional[float],
    ebitda: Optional[float],
    net_debt: Optional[float],
    interest_expense: Optional[float],
    current_assets: Optional[float],
    current_liabilities: Optional[float],
) -> str:
    """
    Build a chain-of-thought prompt for computing derived financial ratios.

    Rather than asking the model to output ratios directly (which leads to
    silent errors when source figures are in mixed units), this pattern
    makes the model show each calculation step before committing to a result.

    The CoT structure enforces:
      Step 1 — Unit normalisation: confirm all inputs in £000s
      Step 2 — Formula statement: write out the formula before computing
      Step 3 — Substitution: substitute actual values
      Step 4 — Result with confidence: state result and flag if unusual

    Why this matters for banking:
      Leverage ratios are the most common extraction error. A leverage ratio
      of 2.5x is normal for UK mid-market credit; 25.0x is a data error
      (common when net debt is in £millions and EBITDA in £thousands).
      The CoT step requiring unit confirmation catches this category of error
      before it propagates to downstream credit scoring.

    Args:
        revenue:             Annual revenue (£000s, or None if not extracted).
        ebitda:              EBITDA (£000s, or None).
        net_debt:            Net debt (£000s, or None).
        interest_expense:    Net interest expense (£000s, or None).
        current_assets:      Current assets (£000s, or None).
        current_liabilities: Current liabilities (£000s, or None).

    Returns:
        Formatted user prompt string for ratio derivation.
    """
    inputs = []
    if revenue is not None:
        inputs.append(f"  Revenue:             £{revenue:,.0f}k")
    if ebitda is not None:
        inputs.append(f"  EBITDA:              £{ebitda:,.0f}k")
    if net_debt is not None:
        inputs.append(f"  Net Debt:            £{net_debt:,.0f}k")
    if interest_expense is not None:
        inputs.append(f"  Interest Expense:    £{interest_expense:,.0f}k")
    if current_assets is not None:
        inputs.append(f"  Current Assets:      £{current_assets:,.0f}k")
    if current_liabilities is not None:
        inputs.append(f"  Current Liabilities: £{current_liabilities:,.0f}k")

    inputs_str = "\n".join(inputs) if inputs else "  (No figures extracted yet)"

    return f"""Using the extracted figures below, derive the required financial ratios.
Work through each calculation step by step before stating the result.

EXTRACTED FIGURES (all in £000s):
{inputs_str}

For each ratio, follow this exact structure:
  FORMULA: [state the formula]
  UNIT CHECK: [confirm all inputs are in the same unit — flag any mismatch]
  SUBSTITUTION: [substitute the actual values]
  RESULT: [state the numerical result to 2 decimal places]
  CONFIDENCE: [0.0-1.0 — lower if any input had low confidence or required estimation]
  FLAG: [any concerns — e.g. leverage > 10x warrants a flag even if calculation is correct]

Ratios to derive:
1. EBITDA Margin (%) = EBITDA / Revenue × 100
2. Leverage Ratio (x) = Net Debt / EBITDA
3. Interest Cover (x) = EBITDA / Interest Expense
4. Current Ratio (x) = Current Assets / Current Liabilities

If any input figure is None, state that the ratio cannot be computed and set
confidence to 0.0. Do not estimate or interpolate missing values."""


# ---------------------------------------------------------------------------
# PATTERN 3 — Structured Output Contract
# ---------------------------------------------------------------------------

def build_structured_output_prompt(
    schema_description: str,
    task_description: str,
    document_excerpt: str,
    max_excerpt_chars: int = 2000,
) -> str:
    """
    Build a structured output prompt that enforces exact JSON schema compliance.

    This pattern is paired with Gemini's native JSON mode
    (response_mime_type="application/json") to guarantee parseable output.
    The schema description in the prompt is the human-readable equivalent of
    the Pydantic model — it catches schema violations before they hit the
    validation layer.

    Why explicit schema in prompt AND API-level JSON mode?
      API-level JSON mode guarantees valid JSON syntax.
      Schema description in prompt guides the model toward the correct
      field names, types, and required fields. Without it, models often
      produce valid JSON with wrong field names (e.g., "net_debt_thousands"
      instead of "net_debt") that pass JSON parsing but fail Pydantic.

    Args:
        schema_description: Human-readable description of the required schema.
        task_description:   What the model is being asked to extract.
        document_excerpt:   The text to extract from (truncated to max_excerpt_chars).
        max_excerpt_chars:  Safety truncation limit.

    Returns:
        Formatted user prompt string.
    """
    if len(document_excerpt) > max_excerpt_chars:
        document_excerpt = document_excerpt[:max_excerpt_chars] + "\n[... document continues ...]"

    return f"""TASK:
{task_description}

REQUIRED OUTPUT SCHEMA:
{schema_description}

CRITICAL CONSTRAINTS:
- Return ONLY the JSON object. No markdown code fences, no explanation.
- Every required field must be present, even if the value is null.
- String fields: use null, not empty string "", when data is not found.
- Numeric fields: use null, not 0 or -1, when data is not found.
- Confidence scores: must be genuine probabilities 0.0-1.0. Do not
  default all fields to 1.0 — calibrate based on source clarity.
- source_paragraph: quote the exact text from the document (max 100 chars).
  If no source found, use null. Never fabricate a quote.

DOCUMENT TEXT:
{document_excerpt}"""


# ---------------------------------------------------------------------------
# PATTERN 4 — Few-Shot Examples for Edge Cases
# ---------------------------------------------------------------------------

def build_few_shot_edge_case_prompt(
    document_text: str,
    include_foreign_currency: bool = True,
    include_consolidated: bool = True,
    include_abbreviated: bool = True,
) -> str:
    """
    Build a few-shot prompt that anchors model behaviour on non-standard
    document formats encountered in AWB's corporate credit portfolio.

    Edge case categories:
      Foreign currency accounts: UK borrower with USD functional currency
        (common in resources, shipping, and tech sectors). The model must
        flag the currency rather than silently applying an assumed rate.
      Consolidated vs standalone: Group accounts differ from entity accounts.
        AWB's credit policy requires entity-level figures for covenant
        testing. The model must extract from the correct section.
      Abbreviated accounts: Companies Act 2006 s.444 abbreviated accounts
        for small companies omit the P&L. The model must return null for
        income statement fields rather than hallucinating.

    Why few-shot matters here:
      Zero-shot prompts fail consistently on abbreviated accounts,
      returning estimated revenue figures with high confidence.
      Two-shot examples cut this failure rate from 23% to 3% on
      AWB's edge case validation set (November 2025).

    Args:
        document_text:          The actual document to extract from.
        include_foreign_currency: Include the foreign-currency example.
        include_consolidated:   Include the consolidated/standalone example.
        include_abbreviated:    Include the abbreviated accounts example.

    Returns:
        Formatted user prompt string with examples and task.
    """
    examples = []

    if include_foreign_currency:
        examples.append("""EXAMPLE 1 — Foreign Currency Accounts:
Input excerpt: "Revenue for the year ended 31 December 2024: USD 45,200,000"
Correct output fragment:
  "revenue": {
    "value": 45200.0,
    "unit": "USD000s",
    "source_page": 4,
    "source_paragraph": "Revenue for the year ended 31 December 2024: USD 45,200,000",
    "confidence": 0.95,
    "analyst_review_required": true,
    "review_reason": "Non-GBP currency — conversion rate required before ratio analysis"
  }
Key point: Do NOT convert to GBP. Flag for analyst review with review_reason.""")

    if include_consolidated:
        examples.append("""EXAMPLE 2 — Consolidated vs Entity Accounts:
Input excerpt: "Group revenue: £285,400k. Company (entity) revenue: £142,200k."
AWB credit policy: extract entity-level figures for covenant testing.
Correct output fragment:
  "revenue": {
    "value": 142200.0,
    "unit": "£000s",
    "source_page": 6,
    "source_paragraph": "Company (entity) revenue: £142,200k",
    "confidence": 0.92
  }
Key point: Extract entity-level, not group-level. Note the distinction in
source_paragraph so the analyst can verify the policy was applied.""")

    if include_abbreviated:
        examples.append("""EXAMPLE 3 — Abbreviated Accounts (Companies Act 2006 s.444):
Input excerpt: "Balance Sheet as at 31 March 2025: Fixed Assets £1,240k,
Current Assets £890k, Creditors due within one year £340k"
(No profit and loss account included — abbreviated accounts)
Correct output fragment:
  "revenue": {"value": null, "unit": "£000s", "source_page": null,
    "source_paragraph": null, "confidence": 0.0,
    "analyst_review_required": true,
    "review_reason": "Abbreviated accounts — P&L not filed. Request full accounts."},
  "current_assets": {"value": 890.0, "unit": "£000s", "source_page": 1,
    "source_paragraph": "Current Assets £890k", "confidence": 0.98}
Key point: Return null for missing P&L fields. Do not estimate revenue
from balance sheet figures.""")

    examples_str = "\n\n".join(examples)

    return f"""The following examples show correct extraction behaviour for
non-standard document formats. Study them before extracting from the
target document.

{examples_str}

NOW EXTRACT FROM THE TARGET DOCUMENT:
{document_text[:3000]}{"..." if len(document_text) > 3000 else ""}

Apply the same principles shown in the examples. When in doubt, return
null with analyst_review_required=true and a clear review_reason."""


# ---------------------------------------------------------------------------
# Convenience: build full extraction prompt from components
# ---------------------------------------------------------------------------

def build_full_extraction_prompt(
    document_text: str,
    document_id: str,
    use_cot: bool = False,
    use_few_shot: bool = True,
) -> tuple[str, str]:
    """
    Assemble a complete system + user prompt pair for financial extraction.

    Args:
        document_text: Full document text.
        document_id:   Unique document identifier.
        use_cot:       Add chain-of-thought reasoning instruction.
        use_few_shot:  Include few-shot edge case examples.

    Returns:
        (system_prompt, user_prompt) tuple ready for Gemini API call.
    """
    system_prompt = build_role_system_prompt()

    cot_instruction = ""
    if use_cot:
        cot_instruction = (
            "\n\nFor any derived ratios, show your calculation step by step "
            "before stating the final value. Format: "
            "FORMULA → UNIT CHECK → SUBSTITUTION → RESULT."
        )

    if use_few_shot:
        user_prompt = build_few_shot_edge_case_prompt(document_text)
    else:
        user_prompt = (
            f"Extract all financial fields from this credit pack.\n"
            f"Document ID: {document_id}\n{cot_instruction}\n\n"
            f"DOCUMENT TEXT:\n{document_text[:800_000]}"
        )

    return system_prompt, user_prompt
