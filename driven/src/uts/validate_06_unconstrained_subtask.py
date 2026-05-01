"""
Experiment 06: Unconstrained Subtask (Theorem 7.3).

Evaluate 19 syntactically distinct expressions all of which evaluate to 3
under standard arithmetic interpretation. Subtasks may be signed,
fractional, mixed-type, or locally infeasible.
"""
from __future__ import annotations

import math

from .common import banner, save_results


EXPRESSIONS = [
    ("1 + 1 + 1",                          lambda: 1 + 1 + 1),
    ("2 + 1",                              lambda: 2 + 1),
    ("1 + 2",                              lambda: 1 + 2),
    ("3",                                  lambda: 3),
    ("4 - 1",                              lambda: 4 - 1),
    ("-1 + 4",                             lambda: -1 + 4),
    ("(11 - 10) + (-1 - (-1)) + 2",        lambda: (11 - 10) + (-1 - (-1)) + 2),
    ("6 / 2",                              lambda: 6 / 2),
    ("3 * 1",                              lambda: 3 * 1),
    ("log_e e^3",                          lambda: math.log(math.e ** 3)),
    ("sin(3*pi/2) + 4",                    lambda: math.sin(3 * math.pi / 2) + 4),
    ("3 * 4 / 4",                          lambda: 3 * 4 / 4),
    ("9 - 6",                              lambda: 9 - 6),
    ("e^ln3",                              lambda: math.exp(math.log(3))),
    ("sqrt(9)",                            lambda: math.sqrt(9)),
    ("1 + sqrt(4)",                        lambda: 1 + math.sqrt(4)),
    ("1000000 - 999997",                   lambda: 1000000 - 999997),
    ("3 + 0",                              lambda: 3 + 0),
    ("0 - (-3)",                           lambda: 0 - -3),
]


def validate():
    banner("EXPERIMENT 06 — UNCONSTRAINED SUBTASK")

    records = []
    matches = 0
    for desc, fn in EXPRESSIONS:
        value = fn()
        match = abs(value - 3.0) < 1e-12
        if match:
            matches += 1
        records.append({"expr": desc, "value": float(value), "matches_target": match})
        print(f"  {desc:<45s} -> {value!r:>10s}  {match}")

    summary = {
        "claim": "All 19 syntactically distinct expressions evaluate to 3",
        "n_expressions": len(records),
        "n_matches": matches,
        "all_match": matches == len(records),
        "overall_pass": matches == len(records),
    }
    out = save_results("06_unconstrained_subtask", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
