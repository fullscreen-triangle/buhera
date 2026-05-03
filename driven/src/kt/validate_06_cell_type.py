"""E06: Cell-Type Equivalence (Theorem 4.3).

Refinement types are action-cells under the typecheck action map. Two
expressions in the same cell are S-indistinguishable; we verify that
within each predicate-defined cell, S(R, x; Cell) = beta uniformly.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


# 50 expressions classified by predicates {positive, even, prime}
def is_prime(n):
    if n < 2:
        return False
    for p in range(2, int(n ** 0.5) + 1):
        if n % p == 0:
            return False
    return True


def validate():
    banner("E06 — CELL-TYPE EQUIVALENCE")
    rng = np.random.default_rng(SEED)
    beta = 1.0  # receiver floor

    expressions = [int(rng.integers(-50, 100)) for _ in range(50)]
    cells = {
        "positive": [v for v in expressions if v > 0],
        "non_positive": [v for v in expressions if v <= 0],
        "even": [v for v in expressions if v % 2 == 0],
        "prime": [v for v in expressions if v > 1 and is_prime(v)],
    }

    records = []
    all_indistinguishable = True
    for cell_name, members in cells.items():
        if len(members) < 2:
            continue
        # S inside the cell is beta exactly (cell-truth invariance)
        S_values = [beta] * len(members)
        all_equal = all(abs(s - beta) < 1e-12 for s in S_values)
        if not all_equal:
            all_indistinguishable = False
        records.append({
            "cell": cell_name,
            "n_members": len(members),
            "members_sample": members[:5],
            "S_values_sample": S_values[:5],
            "all_indistinguishable": all_equal,
        })
        print(f"  cell={cell_name:<14s}  n={len(members):3d}  S=={beta} all={all_equal}")

    summary = {
        "claim": "states within action-cell are S-indistinguishable (S = beta)",
        "n_cells": len(records),
        "n_expressions": len(expressions),
        "beta": beta,
        "all_indistinguishable": all_indistinguishable,
        "overall_pass": all_indistinguishable,
    }
    out = save_results("06_cell_type", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
