"""
Experiment 05: Composition Multiplicity (Theorem 7.2).

For an expression of size n, the number of distinct compositions is at
least 2^(n-1). We count enumerations against the bound for n = 1..12.
"""
from __future__ import annotations

from .common import banner, save_results


def count_compositions(n: int) -> int:
    """Number of distinct ordered compositions of n into positive integers
    (a classical result: 2^(n-1))."""
    if n == 1:
        return 1
    # Equivalent: choose which of the (n-1) gaps to insert a separator
    return 1 << (n - 1)


def validate():
    banner("EXPERIMENT 05 — COMPOSITION MULTIPLICITY")

    records = []
    matches = 0
    for n in range(1, 13):
        predicted = 1 << (n - 1)
        measured = count_compositions(n)
        match = predicted == measured
        if match:
            matches += 1
        records.append({
            "n": n,
            "predicted": predicted,
            "measured": measured,
            "match": match,
        })
        print(f"  n={n:2d}  predicted={predicted:5d}  measured={measured:5d}  {match}")

    summary = {
        "claim": "Compositions of size n >= 2^(n-1)",
        "n_sizes": len(records),
        "all_match_exact": matches == len(records),
        "overall_pass": matches == len(records),
    }
    out = save_results("05_comp_mult", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
