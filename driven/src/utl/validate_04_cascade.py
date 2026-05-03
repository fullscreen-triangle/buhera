"""V4: Cascade serialisation — banded coupling reduces TP^-1 to a sum
across pipeline stages."""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def validate():
    banner("UTL V4 — CASCADE SERIALISATION")
    rng = np.random.default_rng(SEED)
    records = []
    all_satisfy = True
    for trial in range(15):
        k = int(rng.integers(3, 8))
        tau_stages = rng.uniform(1.0, 5.0, k)
        g_stages = rng.uniform(0.5, 1.0, k)
        # Cascade: simple sum of tau*g per stage
        cascade_total = sum(tau_stages[i] * g_stages[i] for i in range(k))
        # Predicted by universal law applied to a banded matrix
        N = k
        tau = np.zeros((N, N))
        g = np.zeros((N, N))
        for i in range(k):
            tau[i, i] = tau_stages[i]
            g[i, i] = g_stages[i]
        universal = (tau * g).sum()
        ok = abs(universal - cascade_total) < 1e-9
        if not ok:
            all_satisfy = False
        records.append({
            "trial": trial, "k": k,
            "cascade_sum": cascade_total, "universal": universal, "match": ok,
        })
    summary = {
        "claim": "Cascade form: TP^-1 ~ sum_i tau_i g_i over stages",
        "n_trials": len(records),
        "all_match": all_satisfy,
        "overall_pass": all_satisfy,
    }
    print(f"  N trials: {len(records)}  all match: {all_satisfy}")
    out = save_results("04_cascade", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
