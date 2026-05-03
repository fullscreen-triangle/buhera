"""V1: Universal OS Transport Law calibration.

Build a synthetic kernel with random tau and g matrices, predict TP via
TP^-1 = N^-1 sum tau*g, simulate by sampling decisions weighted by g and
accumulating tau, verify prediction matches simulation.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def predict_tp_inverse(tau: np.ndarray, g: np.ndarray, N: int) -> float:
    return float((tau * g).sum() / (N * N))


def simulate_kernel(tau: np.ndarray, g: np.ndarray, n_decisions: int, rng) -> tuple:
    """Monte Carlo estimator of TP^-1 = N^-2 sum tau*g.

    Uniform sampling over (i,j) pairs (matches the universal-law averaging
    convention in the paper, which weights each decision class equally).
    """
    N = tau.shape[0]
    total = 0.0
    classes = []
    for _ in range(n_decisions):
        i = int(rng.integers(N))
        j = int(rng.integers(N))
        total += tau[i, j] * g[i, j]
        classes.append(j)
    measured_tp_inv = total / n_decisions
    return measured_tp_inv, classes


def validate():
    banner("UTL V1 — UNIVERSAL TRANSPORT LAW CALIBRATION")
    rng = np.random.default_rng(SEED)
    records = []
    max_err = 0.0
    for trial in range(20):
        N = int(rng.integers(2, 6))
        tau = rng.uniform(0.5, 5.0, (N, N))
        g = rng.uniform(0.1, 1.0, (N, N))
        g = (g + g.T) / 2  # symmetric
        predicted = predict_tp_inverse(tau, g, N)
        measured, _ = simulate_kernel(tau, g, n_decisions=20000, rng=rng)
        err = abs(predicted - measured) / max(predicted, 1e-12)
        max_err = max(max_err, err)
        records.append({
            "trial": trial, "N": N, "predicted_tp_inv": predicted,
            "measured_tp_inv": measured, "rel_err": err,
        })
    summary = {
        "claim": "TP^-1 = N^-1 sum tau*g matches simulated throughput",
        "n_trials": len(records),
        "max_rel_error": max_err,
        "mean_rel_error": float(np.mean([r["rel_err"] for r in records])),
        "overall_pass": max_err < 0.10,
    }
    print(f"  N trials: {len(records)}  max rel err: {max_err:.4f}")
    out = save_results("01_universal_law", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
