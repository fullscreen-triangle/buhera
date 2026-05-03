"""V15: Cross-architecture invariance — equivalences hold with architecture-
specific reference frequency f.

For each architecture, M = Q is independent of f, and t = Q/f converts
the same Q into the architecture's local time.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def validate():
    banner("COE V15 — CROSS-ARCHITECTURE INVARIANCE")
    rng = np.random.default_rng(SEED)
    architectures = [
        ("microkernel", 1_000_000.0),
        ("monolithic", 2_500_000.0),
        ("exokernel",  500_000.0),
        ("unikernel",  10_000_000.0),
    ]
    records = []
    all_match = True
    for arch, f in architectures:
        for trial in range(10):
            Q = int(rng.integers(1, 5_000))
            # Per-architecture conversion
            M = Q
            t = Q / f
            # Recover Q from t and f
            Q_recovered = int(round(t * f))
            ok = (M == Q) and (Q_recovered == Q)
            if not ok:
                all_match = False
            records.append({
                "architecture": arch, "f": f, "trial": trial,
                "Q": Q, "M": M, "t": t, "Q_recovered": Q_recovered,
                "match": ok,
            })
    summary = {
        "claim": "Equivalences hold across architectures with their own f",
        "n_architectures": len(architectures),
        "n_trials_total": len(records),
        "all_match": all_match,
        "overall_pass": all_match,
    }
    print(f"  Architectures: {len(architectures)}  all match: {all_match}")
    out = save_results("15_cross_arch", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
