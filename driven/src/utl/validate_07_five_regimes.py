"""V7: Five-regime classification at boundaries {0.3, 0.5, 0.8, 0.95}."""
from __future__ import annotations

import numpy as np

from .common import banner, save_results


def classify(R: float) -> str:
    if R < 0.3:
        return "turbulent"
    if R < 0.5:
        return "aperture"
    if R < 0.8:
        return "cascade"
    if R < 0.95:
        return "coherent"
    return "phase_locked"


REGIMES = ["turbulent", "aperture", "cascade", "coherent", "phase_locked"]


def validate():
    banner("UTL V7 — FIVE-REGIME CLASSIFICATION")
    R_values = np.linspace(0.0, 1.0, 1001)
    records = []
    counts = dict.fromkeys(REGIMES, 0)
    for R in R_values:
        regime = classify(float(R))
        counts[regime] += 1
        records.append({"R": float(R), "regime": regime})

    # Verify monotonic regime ordering
    monotone = all(REGIMES.index(records[i+1]["regime"]) >=
                   REGIMES.index(records[i]["regime"])
                   for i in range(len(records)-1))

    summary = {
        "claim": "R partitions [0,1] into 5 regimes at {0.3, 0.5, 0.8, 0.95}",
        "n_R_values": len(records),
        "regime_counts": counts,
        "monotone_ordering": monotone,
        "all_classified": all(r["regime"] in REGIMES for r in records),
        "overall_pass": monotone and all(r["regime"] in REGIMES for r in records),
    }
    print(f"  Distribution: {counts}  monotone: {monotone}")
    out = save_results("07_five_regimes", {"summary": summary})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
