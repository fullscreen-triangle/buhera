"""Shared plotting utilities for COE panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANEL_DIR = Path(__file__).parent.parent.parent.parent.parent / "long-grass" / "docs" / "computational-operations-equivalence" / "figures"
PANEL_DIR.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
    })


# Greens for COE (distinguishes from UTL's blue)
GREEN_DARK = "#1a5e3a"
GREEN_MID = "#2e8b57"
GREEN_LIGHT = "#7ec48a"
GREEN_PALE = "#c8e6c9"
ACCENT_RED = "#c0392b"
ACCENT_AMBER = "#d39400"
GREY = "#555555"


def save_panel(fig, name: str):
    out = PANEL_DIR / f"{name}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out
