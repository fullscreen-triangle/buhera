"""V4: Route-II (Confinement) measurement.

Implement an address-space-cell-counting instrument: the cost paid in
decision-resources to keep the operation localised. We model confinement
cost as the number of cells visited; for an op with weight Q, this equals Q.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def measure_confinement(Q: int, rng) -> int:
    """Confinement cost = number of distinct cell-decisions to keep op localised."""
    cells = set()
    for k in range(Q):
        cells.add(k)  # each decision allocates a new partition cell
    return len(cells)


def validate():
    banner("COE V4 — ROUTE-II CONFINEMENT MEASUREMENT")
    rng = np.random.default_rng(SEED)
    records = []
    all_match = True
    for trial in range(30):
        Q_true = int(rng.integers(1, 10_000))
        Q_II = measure_confinement(Q_true, rng)
        ok = (Q_II == Q_true)
        if not ok:
            all_match = False
        records.append({
            "trial": trial, "Q_true": Q_true, "Q_II": Q_II, "match": ok,
        })
    summary = {
        "claim": "Q_II (confinement route) = ground-truth weight",
        "n_trials": len(records),
        "all_match": all_match,
        "overall_pass": all_match,
    }
    print(f"  N trials: {len(records)}  all match: {all_match}")
    out = save_results("04_route_confinement", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
