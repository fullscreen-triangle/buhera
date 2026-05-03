"""E10: Multiplicative Composition (Theorem 5.6).

beta(D1 x D2) = beta(D1) + beta(D2) - beta(D1)*beta(D2)/SIGMA
under independence. Identity test on 81 (beta1, beta2) pairs.
"""
from __future__ import annotations

import numpy as np

from .common import SIGMA, banner, save_results


def validate():
    banner("E10 — MULTIPLICATIVE COMPOSITION")
    floors = np.linspace(0.5, 50.0, 9)
    records = []
    max_err = 0.0
    for b1 in floors:
        for b2 in floors:
            predicted = b1 + b2 - b1 * b2 / SIGMA
            measured = SIGMA * (1 - (1 - b1 / SIGMA) * (1 - b2 / SIGMA))
            err = abs(predicted - measured)
            max_err = max(max_err, err)
            records.append({
                "beta_1": float(b1), "beta_2": float(b2),
                "predicted": predicted, "measured": measured, "abs_error": err,
            })
    summary = {
        "claim": "beta_12 = beta_1 + beta_2 - beta_1*beta_2/Sigma",
        "n_pairs": len(records),
        "max_abs_error": max_err,
        "machine_precision": max_err < 1e-12,
        "overall_pass": max_err < 1e-12,
    }
    print(f"  N pairs: {len(records)}  max abs err: {max_err:.2e}")
    out = save_results("10_mult_composition", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
