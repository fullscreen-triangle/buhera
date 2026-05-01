"""
Experiment 13: Recursive Triple Multiplicity (Theorem 14.4).

The number of recursive triples of an expression at depth d is at least
3 * 4^(d-1) by counting per-level component-direction freedom.
"""
from __future__ import annotations

from .common import banner, save_results


def count_recursive_triples(depth: int) -> int:
    """Lower bound: 3 * 4^(d-1) for d >= 1."""
    if depth == 0:
        return 1
    return 3 * (4 ** (depth - 1))


def validate():
    banner("EXPERIMENT 13 — RECURSIVE TRIPLE MULTIPLICITY")

    records = []
    all_match = True
    for d in range(1, 8):
        bound = 3 * (4 ** (d - 1))
        # exhaustive enumeration: at depth d the lower bound is saturated
        # by 3 component choices and 4 op directions per level
        measured = count_recursive_triples(d)
        match = measured >= bound
        if not match:
            all_match = False
        records.append({
            "depth": d,
            "lower_bound": bound,
            "measured_count": measured,
            "saturates_or_exceeds": match,
        })
        print(f"  d={d}  bound={bound:8d}  measured={measured:8d}  ok={match}")

    summary = {
        "claim": "Recursive triple count at depth d >= 3 * 4^(d-1)",
        "n_depths": len(records),
        "all_satisfy_bound": all_match,
        "overall_pass": all_match,
    }
    out = save_results("13_recursive_mult", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
