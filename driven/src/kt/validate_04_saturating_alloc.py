"""E04: Saturating Allocation (Theorem 3.4).

Among methodologies with sigma_K * sigma_Y = hbar_R, the joint dispersion
sigma = sqrt(sigma_K^2 + sigma_Y^2) is minimised at sigma_K = sigma_Y =
sqrt(hbar_R) (AM-GM saturation).
"""
from __future__ import annotations

import math

import numpy as np

from .common import banner, save_results


def validate():
    banner("E04 — SATURATING ALLOCATION")
    hbar_R = 4.0  # = 2*2 saturation point
    optimal_sigma_K = math.sqrt(hbar_R)
    optimal_joint = math.sqrt(2 * hbar_R)

    ratios = np.geomspace(0.05, 20.0, 25)
    records = []
    measured_min = float("inf")
    measured_argmin = None
    for r in ratios:
        sK = math.sqrt(hbar_R) * math.sqrt(r)
        sY = math.sqrt(hbar_R) / math.sqrt(r)
        # constraint check
        product = sK * sY
        joint = math.sqrt(sK ** 2 + sY ** 2)
        if joint < measured_min:
            measured_min = joint
            measured_argmin = (sK, sY)
        records.append({
            "ratio": float(r), "sigma_K": sK, "sigma_Y": sY,
            "product": product, "joint": joint,
        })

    err = abs(measured_min - optimal_joint) / optimal_joint
    summary = {
        "claim": "min(sigma) over saturating curve at sigma_K = sigma_Y = sqrt(hbar_R)",
        "hbar_R": hbar_R,
        "optimal_sigma_K": optimal_sigma_K,
        "optimal_joint": optimal_joint,
        "measured_min_joint": measured_min,
        "measured_argmin": measured_argmin,
        "rel_error": err,
        "n_grid_points": len(records),
        "overall_pass": err < 0.01,
    }
    print(f"  optimal joint = sqrt(2*hbar_R) = {optimal_joint:.6f}")
    print(f"  measured min  = {measured_min:.6f}  rel_err={err:.2e}")
    out = save_results("04_saturating_alloc", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
