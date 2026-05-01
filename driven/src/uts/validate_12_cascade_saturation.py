"""
Experiment 12: Cascade Saturation (Theorem 13.6).

A cascade (kappa_i) saturates to floor (residual -> 0) iff sum(kappa_i) = inf.
Tested on four cascade types: divergent constant, divergent harmonic,
convergent geometric, and zero.
"""
from __future__ import annotations

import math

from .common import banner, save_results


def cascade_residual(kappas):
    r = 1.0
    for k in kappas:
        r *= (1 - k)
    return r


def validate():
    banner("EXPERIMENT 12 — CASCADE SATURATION")

    # Use n stages large enough that even the slowly-diverging harmonic saturates well.
    n = 10000
    cascades = {
        "divergent_constant_0.1":    ([0.1] * n, True),
        "divergent_harmonic_1/i":    ([1 / (i + 1) for i in range(1, n + 1)], True),
        "convergent_geometric_2^-i": ([2 ** (-(i + 1)) for i in range(1, n + 1)], False),
        "zero":                      ([0.0] * n, False),
    }

    records = []
    for name, (ks, analytic_diverges) in cascades.items():
        sum_k = sum(ks)
        residual = cascade_residual(ks)
        # Two-stage residuals to confirm monotone behaviour
        residual_short = cascade_residual(ks[:max(1, n // 100)])
        # Theorem: residual_n -> 0 iff series diverges. Convergent series have
        # residual bounded away from 0 (bounded below by exp(-sum_k)).
        if analytic_diverges:
            # divergent: residual must be small at large n
            satisfies = residual < 0.1 and residual < residual_short
        else:
            # convergent: residual bounded away from zero
            satisfies = residual > 0.1
        records.append({
            "cascade": name,
            "n_stages": len(ks),
            "sum_kappas": sum_k,
            "analytic_diverges": analytic_diverges,
            "residual_n": residual,
            "residual_n_over_100": residual_short,
            "satisfies_theorem": satisfies,
        })
        print(f"  {name:<32s}  sum={sum_k:9.2f}  residual={residual:.3e}  "
              f"diverges={analytic_diverges}  ok={satisfies}")

    summary = {
        "claim": "Cascade saturates (residual -> 0) iff sum of catalytic powers diverges",
        "n_cascades": len(records),
        "n_match": sum(r["satisfies_theorem"] for r in records),
        "overall_pass": all(r["satisfies_theorem"] for r in records),
    }
    out = save_results("12_cascade_saturation", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
