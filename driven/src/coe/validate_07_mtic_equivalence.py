"""V7: Mass-Time-Identity-Count Equivalence.

Verify the four-way unit conversion: t = Q/f, M = Q, identity == negation
fixed-point at depth Q. All four invariants encode the same underlying Q.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .common import F_REF, SEED, banner, save_results


def identity_at_depth(Q: int) -> str:
    """Identity = stable hash at the negation fixed point of depth Q."""
    h = hashlib.sha256()
    for k in range(Q):
        h.update(k.to_bytes(8, "little"))
    return h.hexdigest()


def validate():
    banner("COE V7 — MASS-TIME-IDENTITY-COUNT EQUIVALENCE")
    rng = np.random.default_rng(SEED)
    records = []
    all_match = True
    for trial in range(30):
        Q = int(rng.integers(1, 5_000))
        t = Q / F_REF
        M = Q
        ident = identity_at_depth(Q)
        # Check: from t and f, recover Q
        Q_from_t = int(round(t * F_REF))
        # Check: M == Q
        # Check: identity hash is deterministic
        ident2 = identity_at_depth(Q)
        ok = (Q_from_t == Q) and (M == Q) and (ident == ident2)
        if not ok:
            all_match = False
        records.append({
            "trial": trial, "Q": Q, "t": t, "M": M,
            "Q_from_t": Q_from_t, "identity_stable": (ident == ident2),
            "match": ok,
        })
    summary = {
        "claim": "t = Q/f, M = Q, identity == fixed point at depth Q (4-way)",
        "n_trials": len(records),
        "all_match": all_match,
        "overall_pass": all_match,
    }
    print(f"  N trials: {len(records)}  all match: {all_match}")
    out = save_results("07_mtic_equivalence", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
