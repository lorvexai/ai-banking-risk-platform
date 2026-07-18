"""
chapter_01/exercises/ai_landscape.py
AWB AI Landscape Bubble Chart — Starter Code

Exercise 1.2: Map AWB's AI Landscape
Difficulty: ★★★☆☆ | Estimated time: 30 minutes

Task: Create a bubble chart that maps 10 banking AI use cases by:
  x = implementation complexity (1–10)
  y = annual ROI potential (£M)
  bubble size = regulatory risk rating (1–5)

Use the colour palette from WORD_FORMATTING_STANDARDS_v2.2.md.
Save as JPEG at 300 DPI with DPI metadata.

Success criterion: chart is readable in greyscale (hatching applied).

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AWB colour palette (WORD_FORMATTING_STANDARDS_v2.2)
# ---------------------------------------------------------------------------

COLOUR_NAVY = "#1F4E79"
COLOUR_AMBER = "#FF6600"
COLOUR_GREEN = "#006400"
COLOUR_PURPLE = "#800080"
COLOUR_RED = "#CC0000"
COLOUR_TEAL = "#008080"

# ---------------------------------------------------------------------------
# AWB AI use case data (10 use cases for Chapter 1 landscape)
# ---------------------------------------------------------------------------

USE_CASES = [
    # (label, complexity 1-10, roi_gbp_m, reg_risk 1-5, colour)
    (
        "Customer Service\nChatbot",
        3, 6.5, 2, COLOUR_TEAL,
    ),
    (
        "Credit Document\nAnalyser",
        5, 8.2, 4, COLOUR_NAVY,
    ),
    (
        "Regulatory\nReporting",
        6, 5.1, 5, COLOUR_RED,
    ),
    (
        "Fraud\nDetection",
        7, 12.4, 5, COLOUR_AMBER,
    ),
    (
        "AML Screening",
        8, 4.8, 5, COLOUR_RED,
    ),
    (
        "SME Credit\nScoring",
        6, 9.3, 4, COLOUR_NAVY,
    ),
    (
        "Treasury\nOps Agent",
        5, 3.7, 3, COLOUR_GREEN,
    ),
    (
        "Model Risk\nMonitor",
        4, 2.9, 4, COLOUR_PURPLE,
    ),
    (
        "Liquidity\nForecasting",
        7, 6.1, 3, COLOUR_GREEN,
    ),
    (
        "KYC\nAutomation",
        6, 7.4, 5, COLOUR_AMBER,
    ),
]

# Hatching patterns for B&W greyscale print readability
HATCH_PATTERNS = [
    "///", "\\\\\\", "xxx", "...", "ooo",
    "+++", "***", "OOO", "---", "|||",
]


def build_landscape_chart(output_path: str = "ai_landscape.jpg") -> None:
    """Build and save the AWB AI landscape bubble chart.

    Args:
        output_path: Destination file path (.jpg).

    Chart spec:
        - x-axis: implementation complexity (1–10)
        - y-axis: annual ROI potential (£M)
        - bubble area: proportional to regulatory risk (1–5)
        - hatching: applied for B&W print readability
        - canvas: 15 cm × 10 cm at 300 DPI (1772 × 1181 px)
    """
    # TODO: Create figure at correct physical size
    # Hint: figsize in inches = (15/2.54, 10/2.54)
    # fig, ax = plt.subplots(figsize=(...))

    # TODO: For each use case in USE_CASES:
    #   - scatter plot with bubble size proportional to reg_risk
    #   - apply hatching pattern for B&W readability
    #   - label each bubble with the use case name
    # Hint: ax.scatter(...) then ax.annotate(...)
    #
    # Use adjustText to prevent label overlap:
    # from adjustText import adjust_text
    # texts = [ax.text(x, y, label) for ...]
    # adjust_text(texts, arrowprops=dict(
    #     arrowstyle="->", color="#A6A6A6", lw=0.8
    # ))

    # TODO: Add axis labels, title, legend for regulatory risk tiers

    # TODO: Save as JPEG at 300 DPI
    # Hint: fig.savefig(output_path, dpi=300,
    #           bbox_inches="tight", pad_inches=0.1,
    #           facecolor="white", format="jpeg")

    raise NotImplementedError(
        "Implement build_landscape_chart — "
        "see Section 1.4 AI Landscape and WFS v2.2 figure rules"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = Path(__file__).parent / "ai_landscape.jpg"
    build_landscape_chart(str(out))
    logger.info("Saved: %s", out)
