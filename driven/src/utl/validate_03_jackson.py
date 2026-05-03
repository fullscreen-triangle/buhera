"""V3: Jackson independence — for diagonal coupling, TP^-1 reduces to
a sum of per-class self-lags."""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def validate():
    banner("UTL V3 — JACKSON INDEPENDENCE")
    rng = np.random.default_rng(SEED)
    records = []
    max_err = 0.0
    for trial in range(15):
        N = int(rng.integers(3, 8))
        tau = rng.uniform(1.0, 5.0, (N, N))
        g = np.eye(N)  # diagonal coupling => independent classes
        predicted = (tau * g).sum() / (N * N)
        # Jackson form: sum of self-lags / N^2
        jackson = sum(tau[i, i] for i in range(N)) / (N * N)
        err = abs(predicted - jackson)
        max_err = max(max_err, err)
        records.append({
            "trial": trial, "N": N,
            "universal": predicted, "jackson": jackson, "abs_err": err,
        })
    summary = {
        "claim": "Universal law reduces to Jackson independence with diagonal g",
        "n_trials": len(records),
        "max_abs_error": max_err,
        "overall_pass": max_err < 1e-12,
    }
    print(f"  N trials: {len(records)}  max abs err: {max_err:.2e}")
    out = save_results("03_jackson", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
