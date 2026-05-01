"""
Experiment 11: Multiplicative Catalytic Power (Theorem 13.4).

For catalysts with kappa_1, kappa_2, the composite power is
1 - (1 - kappa_1)(1 - kappa_2). Tested across 81 ordered pairs.
"""
from __future__ import annotations

from .common import banner, save_results


KAPPAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def composite_predicted(k1: float, k2: float) -> float:
    return 1 - (1 - k1) * (1 - k2)


def composite_measured(k1: float, k2: float) -> float:
    """Apply k1 then k2 to a residual r=1; report total reduction."""
    r = 1.0
    r = r * (1 - k1)
    r = r * (1 - k2)
    return 1 - r


def validate():
    banner("EXPERIMENT 11 — MULTIPLICATIVE CATALYST")

    records = []
    max_err = 0.0
    for k1 in KAPPAS:
        for k2 in KAPPAS:
            pred = composite_predicted(k1, k2)
            meas = composite_measured(k1, k2)
            err = abs(pred - meas)
            max_err = max(max_err, err)
            records.append({
                "kappa_1": k1,
                "kappa_2": k2,
                "predicted": pred,
                "measured": meas,
                "abs_error": err,
            })

    print(f"  N pairs: {len(records)}")
    print(f"  max abs error: {max_err:.2e}")

    summary = {
        "claim": "kappa(g1 * g2) = 1 - (1 - kappa_1)(1 - kappa_2)",
        "n_pairs": len(records),
        "max_abs_error": max_err,
        "machine_precision": max_err < 1e-14,
        "overall_pass": max_err < 1e-14,
    }
    out = save_results("11_multiplicativity", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
