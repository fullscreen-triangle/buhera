"""V14: Coupling estimator — empirical g from class-assignment time series
recovers injected coupling within tolerance."""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def estimate_coupling(classes, N):
    """Empirical pairwise coupling from observed transitions."""
    counts = np.zeros((N, N))
    for t in range(len(classes) - 1):
        counts[classes[t], classes[t+1]] += 1
    g = counts / max(counts.sum(), 1)
    # Diagonal extraction (self-coupling) for normalisation
    return g / max(g.max(), 1e-12)


def validate():
    banner("UTL V14 — COUPLING ESTIMATOR")
    rng = np.random.default_rng(SEED)
    records = []
    max_err = 0.0
    for trial in range(20):
        N = int(rng.integers(3, 6))
        # Inject a coupling structure
        injected = rng.uniform(0.1, 1.0, (N, N))
        injected = (injected + injected.T) / 2
        injected = injected / injected.max()

        # Generate class series following the coupling
        P = injected / injected.sum(axis=1, keepdims=True)
        n_dec = 5000
        classes = [0]
        state = 0
        for _ in range(n_dec):
            state = int(rng.choice(N, p=P[state]))
            classes.append(state)

        # Estimate
        estimated = estimate_coupling(classes, N)

        # Compare structures (correlation)
        injected_flat = injected.flatten()
        estimated_flat = estimated.flatten()
        if np.std(injected_flat) > 0 and np.std(estimated_flat) > 0:
            corr = float(np.corrcoef(injected_flat, estimated_flat)[0, 1])
        else:
            corr = 0.0
        err = 1 - corr  # closer to 0 means better recovery
        max_err = max(max_err, err)
        records.append({
            "trial": trial, "N": N, "correlation": corr, "err": err,
        })

    summary = {
        "claim": "Empirical g recovers injected coupling",
        "n_trials": len(records),
        "max_err": max_err,
        "mean_correlation": float(np.mean([r["correlation"] for r in records])),
        "overall_pass": max_err < 0.5,  # weak correlation criterion
    }
    print(f"  N trials: {len(records)}  mean corr: {summary['mean_correlation']:.3f}")
    out = save_results("14_coupling_estimator", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
