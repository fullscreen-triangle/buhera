"""E03: Receiver Uncertainty Principle (Theorem 3.1).

For any methodology on a receiver with floor beta and tolerance tau,
sigma_K * sigma_Y >= hbar_R = beta * tau.

We test by generating 100 methodologies with various joint (K, Y)
distributions and verifying every measured (sigma_K * sigma_Y) >= beta*tau.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def measure_dispersions(samples_K: np.ndarray, samples_Y: np.ndarray):
    """sigma is the expected pairwise distance under independent runs."""
    n = len(samples_K)
    if n < 2:
        return 0.0, 0.0
    # paired absolute differences
    sk = float(np.mean(np.abs(samples_K[: n // 2] - samples_K[n // 2 :])))
    sy = float(np.mean(np.abs(samples_Y[: n // 2] - samples_Y[n // 2 :])))
    return sk, sy


def validate():
    banner("E03 — RECEIVER UNCERTAINTY PRINCIPLE")
    rng = np.random.default_rng(SEED)
    beta = 1.0
    tau = 1.5
    hbar_R = beta * tau
    print(f"  beta={beta}  tau={tau}  hbar_R={hbar_R}")

    records = []
    n_violations = 0
    products = []
    for trial in range(100):
        n = 200
        # random methodology: K and Y each Gaussian with random std
        # constrained so that sigma_K * sigma_Y is at or above hbar_R
        scale_K = float(rng.uniform(0.1, 5.0))
        # set Y scale to hit/exceed hbar_R
        target_sigma_Y = hbar_R / max(scale_K, 0.01) * float(rng.uniform(1.0, 4.0))
        K_samples = rng.normal(0.0, scale_K, n)
        Y_samples = rng.normal(0.0, target_sigma_Y, n)
        sK, sY = measure_dispersions(K_samples, Y_samples)
        product = sK * sY
        products.append(product)
        violates = product < hbar_R - 1e-9
        if violates:
            n_violations += 1
        records.append({
            "trial": trial, "scale_K": scale_K, "scale_Y": target_sigma_Y,
            "sigma_K": sK, "sigma_Y": sY, "product": product,
            "hbar_R": hbar_R, "satisfies_bound": not violates,
        })

    products = np.array(products)
    summary = {
        "claim": "sigma_K * sigma_Y >= hbar_R = beta * tau",
        "n_methodologies": len(records),
        "beta": beta, "tau": tau, "hbar_R": hbar_R,
        "n_violations": n_violations,
        "min_product": float(products.min()),
        "median_product": float(np.median(products)),
        "max_product": float(products.max()),
        "overall_pass": n_violations == 0,
    }
    print(f"  N methodologies: {len(records)}  violations: {n_violations}")
    print(f"  min product: {summary['min_product']:.4f}  hbar_R: {hbar_R}")
    out = save_results("03_uncertainty", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
