"""V6: Three-Route Equivalence — for each operation, Q_I = Q_II = Q_III to
machine precision.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results
from .validate_03_route_residue import measure_residue
from .validate_04_route_confinement import measure_confinement
from .validate_05_route_negation import negation_iterate


def validate():
    banner("COE V6 — THREE-ROUTE EQUIVALENCE")
    rng = np.random.default_rng(SEED)
    records = []
    all_match = True
    max_disagreement = 0
    for trial in range(50):
        Q_true = int(rng.integers(1, 10_000))
        Q_I = measure_residue(Q_true, rng)
        Q_II = measure_confinement(Q_true, rng)
        Q_III = negation_iterate(Q_true)
        disagreement = max(abs(Q_I - Q_II), abs(Q_II - Q_III), abs(Q_I - Q_III))
        max_disagreement = max(max_disagreement, disagreement)
        ok = (Q_I == Q_II == Q_III == Q_true)
        if not ok:
            all_match = False
        records.append({
            "trial": trial, "Q_true": Q_true,
            "Q_I": Q_I, "Q_II": Q_II, "Q_III": Q_III,
            "all_equal": ok,
        })
    summary = {
        "claim": "Q_I = Q_II = Q_III for every operation",
        "n_trials": len(records),
        "max_disagreement": int(max_disagreement),
        "all_match": all_match,
        "overall_pass": all_match,
    }
    print(f"  N trials: {len(records)}  max disagreement: {max_disagreement}")
    out = save_results("06_three_route_equivalence", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
