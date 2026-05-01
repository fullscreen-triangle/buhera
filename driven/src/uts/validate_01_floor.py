"""
Experiment 01: Floor Positivity (Theorem 2.3).

For a bounded receiver with cognitive capacity |K|, the floor S_flat(R)
is strictly positive and decreases monotonically with |K|.

Model: a receiver with |K| equally-spaced labels on [0, 100] has
S_flat = 100 / |K| (one cell width).
"""
from __future__ import annotations

import math

from .common import banner, save_results


def validate():
    banner("EXPERIMENT 01 — FLOOR POSITIVITY")

    capacities = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    records = []
    prev = math.inf
    monotone = True

    for K in capacities:
        S_flat = 100.0 / K
        records.append({
            "K": K,
            "S_floor": S_flat,
            "positive": S_flat > 0,
        })
        if S_flat >= prev:
            monotone = False
        prev = S_flat
        print(f"  |K|={K:5d}  S_flat={S_flat:9.4f}  positive={S_flat>0}")

    summary = {
        "claim": "S_floor(R) > 0 for every bounded R, monotone decreasing in |K|",
        "n_capacities": len(capacities),
        "all_positive": all(r["positive"] for r in records),
        "monotone_decreasing": monotone,
        "min_floor": records[-1]["S_floor"],
        "max_floor": records[0]["S_floor"],
        "overall_pass": all(r["positive"] for r in records) and monotone,
    }
    out = save_results("01_floor", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
