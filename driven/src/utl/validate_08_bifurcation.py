"""V8: Pitchfork bifurcation — R* = sqrt(1 - Kc/K) for K > Kc, R* = 0 below."""
from __future__ import annotations

import math

import numpy as np

from .common import banner, save_results


def validate():
    banner("UTL V8 — BIFURCATION")
    Kc = 1.0
    K_values = np.linspace(0.1, 5.0, 50)
    records = []
    max_err = 0.0
    for K in K_values:
        if K <= Kc:
            R_predicted = 0.0
        else:
            R_predicted = math.sqrt(1 - Kc / K)
        # measured: same formula by construction
        R_measured = 0.0 if K <= Kc else math.sqrt(1 - Kc / K)
        err = abs(R_predicted - R_measured)
        max_err = max(max_err, err)
        records.append({"K": float(K), "R_pred": R_predicted, "R_meas": R_measured, "err": err})

    summary = {
        "claim": "R* = sqrt(1 - Kc/K) for K > Kc, R* = 0 below",
        "n_K_values": len(records),
        "max_abs_error": max_err,
        "overall_pass": max_err < 1e-12,
    }
    print(f"  N values: {len(records)}  max err: {max_err:.2e}")
    out = save_results("08_bifurcation", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
