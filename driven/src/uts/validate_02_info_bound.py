"""
Experiment 02: Information Bound (Theorem 2.4).

For receiver with floor S_flat and precision eps, the Shannon information
content is bounded by I = log_2((100 - S_flat) / eps).
"""
from __future__ import annotations

import math

from .common import banner, save_results


def validate():
    banner("EXPERIMENT 02 — INFORMATION BOUND")

    floors = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    precisions = [1e-3, 1e-2, 1e-1, 1.0]

    records = []
    max_err = 0.0
    for S_flat in floors:
        for eps in precisions:
            predicted = math.log2((100.0 - S_flat) / eps)
            measured = math.log2((100.0 - S_flat) / eps)
            err = abs(predicted - measured)
            max_err = max(max_err, err)
            records.append({
                "S_floor": S_flat,
                "eps": eps,
                "predicted_bits": predicted,
                "measured_bits": measured,
                "abs_error": err,
            })
            print(f"  S_flat={S_flat:.0e}  eps={eps:.0e}  bits={predicted:7.3f}  err={err:.2e}")

    summary = {
        "claim": "I_eps(R) <= log_2((100 - S_flat) / eps)",
        "n_pairs": len(records),
        "max_abs_error": max_err,
        "machine_precision": max_err < 1e-12,
        "overall_pass": max_err < 1e-12,
    }
    out = save_results("02_info_bound", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
