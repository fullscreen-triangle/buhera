"""Shared plotting utilities for UTL panels.

White background, monochromatic-blue palette, sans-serif. Each panel is a 2x2
grid; one subplot in each panel must be a 3D projection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANEL_DIR = Path(__file__).parent.parent.parent.parent.parent / "long-grass" / "docs" / "os-throughput-law" / "figures"
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


# Monochromatic blue family
BLUE_DARK = "#0b3d91"
BLUE_MID = "#2266c4"
BLUE_LIGHT = "#5c98e8"
BLUE_PALE = "#a9c8f2"
ACCENT_RED = "#c0392b"
ACCENT_AMBER = "#d39400"
GREY = "#555555"


def save_panel(fig, name: str) -> Path:
    out = PANEL_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=220, facecolor="white")
    plt.close(fig)
    return out
