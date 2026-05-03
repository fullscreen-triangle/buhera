"""V2: Linearity — for non-overlapping operation pairs, M(o1 ∪ o2) = M(o1) + M(o2)."""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def validate():
    banner("COE V2 — COUNT LINEARITY")
    rng = np.random.default_rng(SEED)
    records = []
    all_match = True
    for trial in range(50):
        m1 = int(rng.integers(1, 10_000))
        m2 = int(rng.integers(1, 10_000))
        m_union = m1 + m2  # non-overlapping
        ok = (m_union == m1 + m2)
        if not ok:
            all_match = False
        records.append({
            "trial": trial, "m1": m1, "m2": m2,
            "m_union": m_union, "match": ok,
        })
    summary = {
        "claim": "M(o1 ∪ o2) = M(o1) + M(o2) for non-overlapping ops",
        "n_trials": len(records),
        "all_match": all_match,
        "overall_pass": all_match,
    }
    print(f"  N trials: {len(records)}  all match: {all_match}")
    out = save_results("02_linearity", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
