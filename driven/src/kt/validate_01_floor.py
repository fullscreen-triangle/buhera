"""E01: Floor positivity (Theorem 1.1).

beta(R) > 0 and decreases monotonically with cognitive capacity |K|,
under the canonical model beta = SIGMA / |K|.
"""
from __future__ import annotations

from .common import SIGMA, banner, save_results


def validate():
    banner("E01 — FLOOR POSITIVITY")
    capacities = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    records = []
    monotone = True
    prev = float("inf")
    for K in capacities:
        beta = SIGMA / K
        records.append({"K": K, "beta": beta, "positive": beta > 0})
        if beta >= prev:
            monotone = False
        prev = beta
        print(f"  |K|={K:5d}  beta={beta:9.4f}")
    summary = {
        "claim": "beta(R) > 0 and monotone decreasing in |K|",
        "n_capacities": len(capacities),
        "all_positive": all(r["positive"] for r in records),
        "monotone": monotone,
        "min_beta": records[-1]["beta"],
        "max_beta": records[0]["beta"],
        "overall_pass": all(r["positive"] for r in records) and monotone,
    }
    out = save_results("01_floor", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
