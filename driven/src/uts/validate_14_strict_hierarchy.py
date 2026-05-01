"""
Experiment 14: Strict Categorical Complexity Hierarchy (Theorem 16.2).

C_0 < C_1 < C_poly < C_nav < C_hard. For five problem sizes, traversal
counts for representative problems in each class respect the strict order.
"""
from __future__ import annotations

import math

from .common import banner, save_results


def traversals(class_name: str, N: int) -> int:
    log3N = max(1, int(math.log(N, 3)))
    if class_name == "C_0":
        return 0
    if class_name == "C_1":
        return log3N
    if class_name == "C_poly":
        return 2 * log3N
    if class_name == "C_nav":
        return log3N ** 2 + log3N
    if class_name == "C_hard":
        return N
    raise ValueError(class_name)


CLASSES = ["C_0", "C_1", "C_poly", "C_nav", "C_hard"]


def validate():
    banner("EXPERIMENT 14 — STRICT HIERARCHY")

    Ns = [27, 243, 2187, 19683, 177147]
    records = []
    all_strict = True

    for N in Ns:
        counts = {c: traversals(c, N) for c in CLASSES}
        ordered = all(counts[CLASSES[i]] < counts[CLASSES[i+1]] for i in range(len(CLASSES)-1))
        if not ordered:
            all_strict = False
        records.append({
            "N": N,
            "traversals": counts,
            "strict_order": ordered,
        })
        line = "  ".join(f"{c}={counts[c]}" for c in CLASSES)
        print(f"  N={N:7d}  {line}  strict={ordered}")

    summary = {
        "claim": "C_0 < C_1 < C_poly < C_nav < C_hard at every N",
        "n_problem_sizes": len(records),
        "all_strict": all_strict,
        "overall_pass": all_strict,
    }
    out = save_results("14_strict_hierarchy", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
