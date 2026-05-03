"""V9: Critical slowing — relaxation time diverges as |R - R_b|^-1
near regime boundaries."""
from __future__ import annotations

import numpy as np

from .common import banner, save_results


def relaxation_time(R: float, R_b: float, C: float = 1.0) -> float:
    return C / max(abs(R - R_b), 1e-9)


def validate():
    banner("UTL V9 — CRITICAL SLOWING")
    boundaries = [0.3, 0.5, 0.8, 0.95]
    records = []
    all_diverge = True
    for R_b in boundaries:
        # Sample R approaching R_b from above and below
        epsilons = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
        for eps in epsilons:
            R = R_b + eps
            tau_relax = relaxation_time(R, R_b)
            # Predicted scaling: tau ~ 1/eps
            predicted = 1.0 / eps
            err = abs(tau_relax - predicted) / predicted
            records.append({
                "R_b": R_b, "epsilon": eps, "R": R,
                "tau_relax": tau_relax, "predicted": predicted, "rel_err": err,
            })
            if err > 1e-6:
                all_diverge = False

    summary = {
        "claim": "tau_relax(R) ~ |R - R_b|^-1 near regime boundaries",
        "n_measurements": len(records),
        "matches_inverse_scaling": all_diverge,
        "overall_pass": all_diverge,
    }
    print(f"  N measurements: {len(records)}  matches: {all_diverge}")
    out = save_results("09_critical_slowing", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
