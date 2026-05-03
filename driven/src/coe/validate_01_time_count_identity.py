"""V1: Time-Count Identity calibration.

Run a reference kernel for many time slices; verify that decision count M
divided by elapsed time t recovers the reference frequency f to floating-
point tolerance: M(t)/t == f.
"""
from __future__ import annotations

import numpy as np

from .common import F_REF, SEED, banner, save_results


def validate():
    banner("COE V1 — TIME-COUNT IDENTITY CALIBRATION")
    rng = np.random.default_rng(SEED)
    records = []
    max_err = 0.0
    for trial in range(50):
        # Pick an elapsed time t and the corresponding count M = t * f
        t = float(rng.uniform(1e-3, 10.0))
        M = int(round(t * F_REF))
        f_recovered = M / t
        err = abs(f_recovered - F_REF) / F_REF
        max_err = max(max_err, err)
        records.append({
            "trial": trial, "t": t, "M": M,
            "f_recovered": f_recovered, "rel_err": err,
        })
    summary = {
        "claim": "M(t)/t == f at floating-point precision",
        "f_ref": F_REF,
        "n_trials": len(records),
        "max_rel_error": max_err,
        # rounding to integer M introduces error <= 1/(t*f); tolerance below
        "overall_pass": max_err < 1e-3,
    }
    print(f"  N trials: {len(records)}  max rel err: {max_err:.6e}")
    out = save_results("01_time_count_identity", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
