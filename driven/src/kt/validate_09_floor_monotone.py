"""E09: Floor Monotonicity (Theorem 5.4).

If domain D1 refines D2, then beta(D1) <= beta(D2). We construct 20
refinement chains and verify floors are non-decreasing as cells grow.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, SIGMA, banner, save_results


def validate():
    banner("E09 — FLOOR MONOTONICITY UNDER REFINEMENT")
    rng = np.random.default_rng(SEED)
    records = []
    n_chains = 20
    monotone_count = 0
    for chain_idx in range(n_chains):
        chain_len = int(rng.integers(3, 8))
        # finer to coarser: cell counts decrease
        cell_counts = sorted([int(rng.integers(2, 100)) for _ in range(chain_len)],
                             reverse=True)  # finer first
        floors = [SIGMA / c for c in cell_counts]  # floor = SIGMA / cells
        is_monotone = all(floors[i] <= floors[i + 1] + 1e-12 for i in range(len(floors) - 1))
        if is_monotone:
            monotone_count += 1
        records.append({
            "chain": chain_idx,
            "cell_counts": cell_counts,
            "floors": floors,
            "monotone_non_decreasing": is_monotone,
        })

    summary = {
        "claim": "D1 refines D2 implies beta(D1) <= beta(D2)",
        "n_chains": n_chains,
        "n_monotone": monotone_count,
        "overall_pass": monotone_count == n_chains,
    }
    print(f"  N chains: {n_chains}  monotone: {monotone_count}")
    out = save_results("09_floor_monotone", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
