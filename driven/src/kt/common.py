"""Shared utilities for the knowledge-thermodynamics validation suite."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

if sys.platform == "win32" and not getattr(sys, "_buhera_stdout_wrapped", False):
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys._buhera_stdout_wrapped = True
    except (ValueError, AttributeError):
        pass

SEED = 42
SIGMA = 100.0  # canonical normalisation
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_results(name: str, payload: dict) -> Path:
    out = DATA_DIR / f"kt_{name}_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    return out


def banner(title: str) -> None:
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
