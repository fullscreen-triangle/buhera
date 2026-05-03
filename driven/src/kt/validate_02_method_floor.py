"""E02: Methodological floor as Banach fixed point (Lemma 1.4).

The recursion s_{n+1} = kappa * s_n + sigma * kappa converges to
sigma * kappa / (1 - kappa) for any kappa < 1.
"""
from __future__ import annotations

from .common import banner, save_results


def validate():
    banner("E02 — METHODOLOGICAL FLOOR (BANACH FIXED POINT)")
    pairs = [(0.1, 0.5), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.9, 0.5),
             (0.5, 0.1), (0.5, 1.0), (0.5, 5.0), (0.7, 2.0)]
    records = []
    max_err = 0.0
    for kappa, sigma in pairs:
        predicted = sigma * kappa / (1 - kappa)
        s = 50.0
        for _ in range(500):
            s = kappa * s + sigma * kappa
        err = abs(s - predicted) / max(predicted, 1e-12)
        max_err = max(max_err, err)
        records.append({
            "kappa": kappa, "sigma": sigma,
            "predicted": predicted, "measured": s, "rel_error": err,
        })
        print(f"  kappa={kappa:.1f} sigma={sigma:.1f}  predicted={predicted:.6f}  measured={s:.6f}  err={err:.2e}")
    summary = {
        "claim": "s_n -> sigma*kappa/(1-kappa) (Banach fixed point)",
        "n_pairs": len(pairs),
        "max_rel_error": max_err,
        "machine_precision": max_err < 1e-12,
        "overall_pass": max_err < 1e-10,
    }
    out = save_results("02_method_floor", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
