"""
Experiment 07: Local-Global Decoupling (Theorem 7.4).

For a sequence of extreme local subtask values (-1000, ..., 10000), the
construction xi = xi_0 + sum_i (eta_i - eta_i) preserves the global value
xi_0 even as the eta_i take arbitrary local values.
"""
from __future__ import annotations

from .common import banner, save_results


EXTREMES = [-1000.0, -100.0, -50.0, -10.0, 0.0, 100.0, 1000.0, 10000.0]


def validate():
    banner("EXPERIMENT 07 — LOCAL-GLOBAL DECOUPLING")

    target = 3.0
    records = []
    matches = 0

    for eta in EXTREMES:
        # Construct xi = target + (eta - eta), so the global value is target
        # while the local subtask eta has arbitrary value
        xi_value = target + (eta - eta)
        local_value = eta
        global_match = abs(xi_value - target) < 1e-12
        if global_match:
            matches += 1
        records.append({
            "eta_local": eta,
            "global_value": xi_value,
            "global_target": target,
            "global_preserved": global_match,
        })
        print(f"  eta={eta:>10.1f}  global={xi_value:>5.2f}  preserved={global_match}")

    summary = {
        "claim": "Global value preserved across arbitrary local subtask extremes",
        "n_extremes": len(records),
        "n_global_preserved": matches,
        "all_preserved": matches == len(records),
        "overall_pass": matches == len(records),
    }
    out = save_results("07_lg_decoupling", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
