"""
Experiment 09: Virtual Sub-State Existence (Theorem 12.3).

For a global coordinate s in [0, 1], sample sub-coordinate decompositions
(s_1, s_2, s_3) with mean-recovery (s_1+s_2+s_3)/3 = s. Verify that as the
decomposition magnitude M grows, the fraction outside [0, 1]^3 (virtual)
approaches 1.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def sample_decomposition(s: float, M: float, rng) -> tuple:
    """Sample (s_1, s_2, s_3) on the constraint plane (mean = s) with
    coordinates in [-M, M], by drawing s_1 and s_2 uniformly and solving."""
    s1 = float(rng.uniform(-M, M))
    s2 = float(rng.uniform(-M, M))
    s3 = 3 * s - s1 - s2
    return s1, s2, s3


def is_virtual(decomp: tuple) -> bool:
    return any(c < 0 or c > 1 for c in decomp)


def is_physical(decomp: tuple) -> bool:
    return all(0 <= c <= 1 for c in decomp)


def validate():
    banner("EXPERIMENT 09 — VIRTUAL SUB-STATE EXISTENCE")

    rng = np.random.default_rng(SEED)
    n_trials = 1000
    M_grid = [1.0, 2.0, 5.0, 10.0]
    records = []

    for M in M_grid:
        n_virtual = 0
        n_physical = 0
        n_mean_ok = 0
        for _ in range(n_trials):
            s = float(rng.uniform(0.05, 0.95))
            d = sample_decomposition(s, M, rng)
            mean = sum(d) / 3
            if abs(mean - s) < 1e-9:
                n_mean_ok += 1
            if is_virtual(d):
                n_virtual += 1
            elif is_physical(d):
                n_physical += 1
        virtual_frac = n_virtual / n_trials
        records.append({
            "M": M,
            "n_trials": n_trials,
            "n_virtual": n_virtual,
            "n_physical": n_physical,
            "virtual_fraction": virtual_frac,
            "mean_recovery_ok": n_mean_ok == n_trials,
        })
        print(f"  M={M:5.1f}  virtual={virtual_frac:6.4f}  physical={n_physical/n_trials:6.4f}")

    # Predicted: virtual fraction grows monotonically with M
    fractions = [r["virtual_fraction"] for r in records]
    monotone = all(fractions[i+1] >= fractions[i] for i in range(len(fractions)-1))
    saturates = fractions[-1] > 0.85

    summary = {
        "claim": "Virtual sub-state fraction -> 1 as decomposition magnitude grows",
        "M_grid": M_grid,
        "fractions": fractions,
        "monotone": monotone,
        "saturates_above_85_percent": saturates,
        "max_virtual_fraction": fractions[-1],
        "overall_pass": monotone and saturates,
    }
    out = save_results("09_virtual_substates", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
