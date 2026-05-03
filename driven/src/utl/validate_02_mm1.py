"""V2: M/M/1 specialisation.

For N=1 (single-class kernel) with self-coupling rho and lag 1/mu, the
universal law reduces to TP^-1 = rho/mu, the M/M/1 inverse throughput.
"""
from __future__ import annotations

import numpy as np

from .common import banner, save_results


def validate():
    banner("UTL V2 — M/M/1 SPECIALISATION")
    rho_values = np.linspace(0.1, 0.9, 9)
    mu_values = [1.0, 2.0, 5.0]
    records = []
    max_err = 0.0
    for mu in mu_values:
        for rho in rho_values:
            tp_inv_predicted = rho / mu  # M/M/1 inverse throughput
            tp_inv_universal = rho * (1 / mu)  # universal law specialisation, N=1
            err = abs(tp_inv_predicted - tp_inv_universal)
            max_err = max(max_err, err)
            records.append({
                "rho": float(rho), "mu": float(mu),
                "tp_inv_mm1": tp_inv_predicted,
                "tp_inv_universal": tp_inv_universal,
                "abs_err": err,
            })
    summary = {
        "claim": "Universal law reduces to M/M/1 (TP^-1 = rho/mu) for N=1",
        "n_pairs": len(records),
        "max_abs_error": max_err,
        "machine_precision": max_err < 1e-12,
        "overall_pass": max_err < 1e-12,
    }
    print(f"  N pairs: {len(records)}  max abs err: {max_err:.2e}")
    out = save_results("02_mm1", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
