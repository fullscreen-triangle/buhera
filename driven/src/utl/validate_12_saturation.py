"""V12: Federation saturation — adding kernels drives composite tp_inv -> Sigma
from below; marginal benefit decreases multiplicatively (diminishing returns).

Composition law (from paper):  1 - TP^-1_fed/Sigma = prod_i (1 - TP^-1_i/Sigma)
=> TP^-1_fed monotonically increases toward Sigma; per-kernel marginal gain
decreases by a constant factor each step.
"""
from __future__ import annotations

import numpy as np

from .common import SIGMA, banner, save_results


def validate():
    banner("UTL V12 — FEDERATION SATURATION")
    records = []
    monotone_increasing = True
    diminishing_returns = True
    prev_tp_inv = 0.0
    prev_marginal = float("inf")
    for n in range(1, 21):
        # Each kernel has tp_inv = 30; saturate toward Sigma
        tp_inv_fed = SIGMA * (1 - (1 - 30 / SIGMA) ** n)
        marginal = tp_inv_fed - prev_tp_inv
        if n > 1 and tp_inv_fed < prev_tp_inv - 1e-12:
            monotone_increasing = False
        if n > 1 and marginal > prev_marginal + 1e-12:
            diminishing_returns = False
        records.append({
            "n_kernels": n, "tp_inv_fed": tp_inv_fed, "marginal": marginal,
        })
        prev_tp_inv = tp_inv_fed
        prev_marginal = marginal

    asymptote_gap = SIGMA - records[-1]["tp_inv_fed"]
    summary = {
        "claim": "Composite tp_inv saturates at Sigma; marginal benefit diminishes",
        "n_kernels_max": 20,
        "monotone_increasing": monotone_increasing,
        "diminishing_returns": diminishing_returns,
        "final_tp_inv": records[-1]["tp_inv_fed"],
        "asymptote_gap": float(asymptote_gap),
        "overall_pass": monotone_increasing and diminishing_returns,
    }
    print(f"  Final tp_inv at n=20: {records[-1]['tp_inv_fed']:.6f}")
    out = save_results("12_saturation", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
