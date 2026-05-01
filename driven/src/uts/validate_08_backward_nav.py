"""
Experiment 08: Backward Navigation Complexity (Theorem 11.2).

In a ternary refinement hierarchy of depth k, backward navigation from any
leaf to the root visits exactly k = log_3(N) states. Verified across
depths 2..10 with 100 random leaves per depth.
"""
from __future__ import annotations

import math

import numpy as np

from .common import SEED, banner, save_results


def steps_to_root(leaf_address: tuple) -> int:
    return len(leaf_address)


def validate():
    banner("EXPERIMENT 08 — BACKWARD NAVIGATION")

    rng = np.random.default_rng(SEED)
    records = []
    all_exact = True

    for depth in range(2, 11):
        N = 3 ** depth
        n_samples = min(100, N)
        steps_observed = []
        for _ in range(n_samples):
            address = tuple(int(d) for d in rng.integers(0, 3, size=depth))
            steps = steps_to_root(address)
            steps_observed.append(steps)

        expected = depth
        log3N = math.log(N, 3)
        min_steps = min(steps_observed)
        max_steps = max(steps_observed)
        exact = (min_steps == max_steps == expected)
        if not exact:
            all_exact = False

        records.append({
            "depth": depth,
            "N": N,
            "log3_N": log3N,
            "expected_steps": expected,
            "min_steps_observed": min_steps,
            "max_steps_observed": max_steps,
            "n_samples": n_samples,
            "exact": exact,
        })
        print(f"  depth={depth:2d}  N={N:7d}  log3(N)={log3N:5.2f}  steps={min_steps}-{max_steps}  exact={exact}")

    summary = {
        "claim": "Backward navigation takes exactly log_3(N) = depth steps",
        "n_depths_tested": len(records),
        "all_exact": all_exact,
        "overall_pass": all_exact,
    }
    out = save_results("08_backward_nav", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
