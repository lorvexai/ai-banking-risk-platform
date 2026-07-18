"""
chapter_01/solutions/ai_landscape_solution.py
AWB AI Landscape Bubble Chart — Complete Solution

Exercise 1.2 complete solution.
Reference: github.com/lorvenio/ai-banking-risk-platform/chapter_01/

Author: AWB AI Programme (AWB-AI-2025)
Chapter: 1 — The AI Transformation of Risk and Compliance
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# AWB colour palette (WORD_FORMATTING_STANDARDS_v2.2)
COLOUR_NAVY = "#1F4E79"
COLOUR_AMBER = "#FF6600"
COLOUR_GREEN = "#006400"
COLOUR_PURPLE = "#800080"
COLOUR_RED = "#CC0000"
COLOUR_TEAL = "#008080"

USE_CASES = [
    ("Customer Service\nChatbot", 3, 6.5, 2, COLOUR_TEAL),
    ("Credit Document\nAnalyser", 5, 8.2, 4, COLOUR_NAVY),
    ("Regulatory\nReporting", 6, 5.1, 5, COLOUR_RED),
    ("Fraud\nDetection", 7, 12.4, 5, COLOUR_AMBER),
    ("AML Screening", 8, 4.8, 5, COLOUR_RED),
    ("SME Credit\nScoring", 6, 9.3, 4, COLOUR_NAVY),
    ("Treasury\nOps Agent", 5, 3.7, 3, COLOUR_GREEN),
    ("Model Risk\nMonitor", 4, 2.9, 4, COLOUR_PURPLE),
    ("Liquidity\nForecasting", 7, 6.1, 3, COLOUR_GREEN),
    ("KYC\nAutomation", 6, 7.4, 5, COLOUR_AMBER),
]

HATCH_PATTERNS = [
    "///", "\\\\\\", "xxx", "...", "ooo",
    "+++", "***", "OOO", "---", "|||",
]


def build_landscape_chart(
    output_path: str = "ai_landscape.jpg",
) -> None:
    """Build and save the AWB AI landscape bubble chart.

    Args:
        output_path: Destination file path (.jpg).
    """
    # 15 cm × 10 cm at 300 DPI
    w_in = 15 / 2.54
    h_in = 10 / 2.54
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    try:
        from adjustText import adjust_text
        use_adjust = True
    except ImportError:
        use_adjust = False
        logger.warning(
            "adjustText not installed; labels may overlap. "
            "pip install adjustText"
        )

    text_objects = []

    for idx, (label, cx, cy, risk, colour) in enumerate(USE_CASES):
        size = risk * 400  # bubble area proportional to risk
        hatch = HATCH_PATTERNS[idx % len(HATCH_PATTERNS)]

        # Scatter with hatch for B&W print
        ax.scatter(
            cx, cy,
            s=size,
            c=colour,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.8,
            hatch=hatch,
            zorder=3,
        )

        txt = ax.text(
            cx + 0.15,
            cy + 0.15,
            label,
            fontsize=6,
            fontfamily="DejaVu Sans",
            va="bottom",
            ha="left",
            color="#1A1A1A",
            zorder=4,
        )
        text_objects.append(txt)

    if use_adjust:
        from adjustText import adjust_text
        adjust_text(
            text_objects,
            ax=ax,
            arrowprops=dict(
                arrowstyle="->",
                color="#A6A6A6",
                lw=0.8,
            ),
        )

    ax.set_xlabel(
        "Implementation Complexity  (1 = simple, 10 = complex)",
        fontsize=8,
        fontfamily="DejaVu Sans",
        color="#333333",
    )
    ax.set_ylabel(
        "Annual ROI Potential (£M)",
        fontsize=8,
        fontfamily="DejaVu Sans",
        color="#333333",
    )
    ax.set_title(
        "AWB AI Landscape — Use Cases by Complexity, ROI "
        "and Regulatory Risk\n(bubble size = regulatory risk; "
        "June 2026)",
        fontsize=9,
        fontfamily="DejaVu Sans",
        fontweight="bold",
        color="#1F4E79",
    )

    ax.set_xlim(1, 11)
    ax.set_ylim(0, 15)
    ax.grid(
        True, linestyle="--", linewidth=0.5,
        color="#CCCCCC", alpha=0.7,
    )

    # Risk tier legend
    risk_legend = [
        mpatches.Patch(
            facecolor="#DDDDDD",
            edgecolor="black",
            hatch=HATCH_PATTERNS[i],
            label=f"Risk tier {i + 1}",
        )
        for i in range(5)
    ]
    ax.legend(
        handles=risk_legend,
        title="Regulatory Risk",
        fontsize=6,
        title_fontsize=7,
        loc="lower right",
        framealpha=0.9,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="white",
        format="jpeg",
    )
    plt.close(fig)
    logger.info("Saved chart: %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    out = Path(__file__).parent / "ai_landscape.jpg"
    build_landscape_chart(str(out))
