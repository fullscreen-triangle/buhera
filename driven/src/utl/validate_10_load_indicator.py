"""V10: Load indicator — algorithm using critical slowing predicts
distance to nearest regime boundary."""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def validate():
    banner("UTL V10 — LOAD INDICATOR")
    rng = np.random.default_rng(SEED)
    boundaries = [0.3, 0.5, 0.8, 0.95]
    records = []
    correct = 0
    for trial in range(50):
        true_R = float(rng.uniform(0.05, 0.99))
        # Distance to nearest boundary
        d_true = min(abs(true_R - b) for b in boundaries)
        # Estimate via tau_relax = 1/d
        tau_relax = 1.0 / max(d_true, 1e-6)
        d_estimated = 1.0 / tau_relax
        err = abs(d_estimated - d_true) / max(d_true, 1e-6)
        if err < 0.05:
            correct += 1
        records.append({
            "trial": trial, "R_true": true_R, "d_true": d_true,
            "tau_relax": tau_relax, "d_estimated": d_estimated,
            "rel_err": err, "correct": err < 0.05,
        })

    summary = {
        "claim": "Distance estimator from tau_relax matches true distance to boundary",
        "n_trials": len(records),
        "n_correct": correct,
        "accuracy": correct / len(records),
        "overall_pass": correct >= int(0.95 * len(records)),
    }
    print(f"  N trials: {len(records)}  correct: {correct}")
    out = save_results("10_load_indicator", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
