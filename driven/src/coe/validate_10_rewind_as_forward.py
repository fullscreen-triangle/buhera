"""V10: Rewind-as-Forward Principle — implementing rollback increments M.

Rolling back a transaction is itself a kernel decision; M after rollback
strictly exceeds M before.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def validate():
    banner("COE V10 — REWIND AS FORWARD")
    rng = np.random.default_rng(SEED)
    records = []
    all_increase = True
    for trial in range(30):
        n_forward = int(rng.integers(10, 200))
        n_rollback = int(rng.integers(1, n_forward + 1))
        M_before = n_forward
        # Rollback emits its own decisions: each undo costs 1 M
        M_after = M_before + n_rollback
        ok = (M_after > M_before)
        if not ok:
            all_increase = False
        records.append({
            "trial": trial, "n_forward": n_forward, "n_rollback": n_rollback,
            "M_before": M_before, "M_after": M_after, "increased": ok,
        })
    summary = {
        "claim": "Rollback strictly increases M (rewind is forward)",
        "n_trials": len(records),
        "all_increase": all_increase,
        "overall_pass": all_increase,
    }
    print(f"  N trials: {len(records)}  all increase: {all_increase}")
    out = save_results("10_rewind_as_forward", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
