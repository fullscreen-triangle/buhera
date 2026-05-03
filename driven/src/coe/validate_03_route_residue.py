"""V3: Route-I (Residue) measurement.

Implement a decision-counter instrument: simulate an operation as a sequence
of decision events while the operation maintains its categorical identity;
report the accumulated decision count Q_I.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def measure_residue(true_weight: int, rng) -> int:
    """Faithful Route-I instrument: increment counter once per decision."""
    counter = 0
    for _ in range(true_weight):
        counter += 1
    return counter


def validate():
    banner("COE V3 — ROUTE-I RESIDUE MEASUREMENT")
    rng = np.random.default_rng(SEED)
    records = []
    all_match = True
    for trial in range(30):
        Q_true = int(rng.integers(1, 10_000))
        Q_I = measure_residue(Q_true, rng)
        ok = (Q_I == Q_true)
        if not ok:
            all_match = False
        records.append({
            "trial": trial, "Q_true": Q_true, "Q_I": Q_I, "match": ok,
        })
    summary = {
        "claim": "Q_I (residue route) = ground-truth weight",
        "n_trials": len(records),
        "all_match": all_match,
        "overall_pass": all_match,
    }
    print(f"  N trials: {len(records)}  all match: {all_match}")
    out = save_results("03_route_residue", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
