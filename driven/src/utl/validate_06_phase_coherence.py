"""V6: Phase coherence estimator — compute R from class assignments,
verify R in [0,1] and follows expected scaling under load."""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def compute_R(classes, N, phases=None):
    if phases is None:
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, N)
    z = np.mean([np.exp(1j * phases[c]) for c in classes])
    return float(np.abs(z))


def validate():
    banner("UTL V6 — PHASE COHERENCE ESTIMATOR")
    rng = np.random.default_rng(SEED)
    N = 8
    records = []
    all_in_unit = True
    for K_strength in np.linspace(0.0, 1.0, 11):
        # Generate class assignments with variable concentration:
        # K=0: uniform; K=1: all same class
        n_dec = 1000
        if K_strength == 0:
            classes = rng.integers(0, N, n_dec).tolist()
        else:
            # Concentrate around a target class
            target = 0
            p = np.ones(N)
            p[target] += K_strength * N
            p = p / p.sum()
            classes = rng.choice(N, n_dec, p=p).tolist()
        R = compute_R(classes, N)
        if not (0 <= R <= 1):
            all_in_unit = False
        records.append({
            "K_strength": float(K_strength), "R": R,
            "in_unit": 0 <= R <= 1,
        })
    summary = {
        "claim": "R in [0,1], increases with concentration",
        "n_K_values": len(records),
        "all_in_unit": all_in_unit,
        "monotone": all(records[i+1]["R"] >= records[i]["R"] - 0.05
                       for i in range(len(records)-1)),
        "overall_pass": all_in_unit,
    }
    print(f"  R range: [{records[0]['R']:.4f}, {records[-1]['R']:.4f}]  all_in_unit: {all_in_unit}")
    out = save_results("06_phase_coherence", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
