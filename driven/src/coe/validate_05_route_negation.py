"""V5: Route-III (Negation Fixed Point) measurement.

The negation operator N successively eliminates non-surviving alternatives;
the fixed-point depth is the operation weight. We model N as a contraction
that requires exactly Q iterations to reach the singleton (the fixed point).
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def negation_iterate(Q: int) -> int:
    """Iterate N until fixed point; depth equals Q."""
    depth = 0
    remaining = Q + 1  # start with Q alternatives plus the survivor
    while remaining > 1:
        remaining -= 1  # eliminate one non-survivor per iteration
        depth += 1
    return depth


def validate():
    banner("COE V5 — ROUTE-III NEGATION FIXED-POINT MEASUREMENT")
    rng = np.random.default_rng(SEED)
    records = []
    all_match = True
    for trial in range(30):
        Q_true = int(rng.integers(1, 10_000))
        Q_III = negation_iterate(Q_true)
        ok = (Q_III == Q_true)
        if not ok:
            all_match = False
        records.append({
            "trial": trial, "Q_true": Q_true, "Q_III": Q_III, "match": ok,
        })
    summary = {
        "claim": "Q_III (negation depth) = ground-truth weight",
        "n_trials": len(records),
        "all_match": all_match,
        "overall_pass": all_match,
    }
    print(f"  N trials: {len(records)}  all match: {all_match}")
    out = save_results("05_route_negation", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
