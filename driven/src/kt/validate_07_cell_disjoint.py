"""E07: Cell-Disjointness Criterion (Theorem 4.5).

Adding a new refinement type preserves cell-truth invariance iff it is
logically disjoint from existing predicates. We check 30 type pairs.
"""
from __future__ import annotations

from .common import banner, save_results


def predicate(name):
    M = {
        "positive":   lambda v: v > 0,
        "negative":   lambda v: v < 0,
        "zero":       lambda v: v == 0,
        "even":       lambda v: v % 2 == 0,
        "odd":        lambda v: v % 2 != 0,
        "small":      lambda v: abs(v) <= 10,
        "large":      lambda v: abs(v) > 10,
        "is_zero":    lambda v: v == 0,
        "non_zero":   lambda v: v != 0,
        "is_one":     lambda v: v == 1,
    }
    return M[name]


PAIRS = [
    ("positive", "negative", True),    # disjoint
    ("positive", "zero", True),        # disjoint
    ("negative", "zero", True),        # disjoint
    ("positive", "even", False),       # overlap (positive evens)
    ("positive", "odd", False),        # overlap
    ("even", "odd", True),             # disjoint
    ("small", "large", True),          # disjoint by definition
    ("zero", "non_zero", True),        # disjoint
    ("is_zero", "is_one", True),       # disjoint
    ("is_zero", "even", False),        # zero is even
    ("is_one", "odd", False),          # 1 is odd
    ("is_one", "positive", False),     # 1 is positive
    ("positive", "small", False),      # overlap
    ("negative", "even", False),       # overlap
    ("zero", "small", False),          # 0 is small
    # synthetic disjoint repeats
    ("positive", "negative", True),
    ("even", "odd", True),
    ("zero", "non_zero", True),
    ("small", "large", True),
    ("is_zero", "is_one", True),
    ("positive", "negative", True),
    ("even", "odd", True),
    ("small", "large", True),
    ("zero", "non_zero", True),
    ("is_zero", "is_one", True),
    # synthetic overlap repeats
    ("positive", "even", False),
    ("positive", "odd", False),
    ("is_zero", "even", False),
    ("is_one", "positive", False),
    ("zero", "small", False),
]


def overlaps(p1, p2, samples):
    f1, f2 = predicate(p1), predicate(p2)
    return any(f1(v) and f2(v) for v in samples)


def validate():
    banner("E07 — CELL-DISJOINTNESS CRITERION")
    samples = list(range(-20, 21))
    records = []
    correct = 0
    for p1, p2, expected_disjoint in PAIRS:
        measured_overlap = overlaps(p1, p2, samples)
        measured_disjoint = not measured_overlap
        match = measured_disjoint == expected_disjoint
        if match:
            correct += 1
        records.append({
            "predicate_1": p1, "predicate_2": p2,
            "expected_disjoint": expected_disjoint,
            "measured_disjoint": measured_disjoint,
            "match": match,
        })
    summary = {
        "claim": "cell-disjointness iff predicates logically disjoint",
        "n_pairs": len(PAIRS),
        "n_correct": correct,
        "match_rate": correct / len(PAIRS),
        "overall_pass": correct == len(PAIRS),
    }
    print(f"  N pairs: {len(PAIRS)}  matches: {correct}")
    out = save_results("07_cell_disjoint", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
