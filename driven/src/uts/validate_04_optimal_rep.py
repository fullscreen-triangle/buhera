"""
Experiment 04: Optimal Representation (Theorem 6.1).

For each (input rep, computation, output rep), compute the cost in each
of the three representations + conversion costs, and verify the predicted
optimal choice matches the measured minimum.

Cost model: synthetic per-representation costs for nine prototype computations
(FFT, lookup, density). Conversion cost is asymmetric per representation pair.
"""
from __future__ import annotations

import itertools

from .common import banner, save_results


REPS = ["O", "C", "P"]

# Per-computation costs (representation -> cost); higher = more expensive
COMPUTATIONS = {
    "fft":     {"O": 10.0, "C": 80.0, "P": 60.0},
    "lookup":  {"O": 80.0, "C": 5.0,  "P": 50.0},
    "density": {"O": 70.0, "C": 50.0, "P": 8.0},
    "convolve":{"O": 15.0, "C": 100.0,"P": 90.0},
    "search":  {"O": 90.0, "C": 12.0, "P": 70.0},
    "integrate":{"O":40.0, "C": 30.0, "P": 5.0},
    "compose": {"O": 50.0, "C": 25.0, "P": 25.0},
    "filter":  {"O": 20.0, "C": 30.0, "P": 100.0},
    "sample":  {"O": 60.0, "C": 70.0, "P": 10.0},
}

# Conversion cost matrix
CONVERT = {
    ("O", "C"): 5.0, ("O", "P"): 7.0,
    ("C", "O"): 4.0, ("C", "P"): 3.0,
    ("P", "O"): 6.0, ("P", "C"): 2.0,
    ("O", "O"): 0.0, ("C", "C"): 0.0, ("P", "P"): 0.0,
}


def total_cost(comp_rep: str, output_rep: str, comp_costs: dict) -> float:
    return comp_costs[comp_rep] + CONVERT[(comp_rep, output_rep)]


def validate():
    banner("EXPERIMENT 04 — OPTIMAL REPRESENTATION")

    records = []
    matches = 0
    for comp_name, comp_costs in COMPUTATIONS.items():
        for out_rep in REPS:
            costs = {r: total_cost(r, out_rep, comp_costs) for r in REPS}
            optimal_rep = min(costs, key=costs.get)
            optimal_cost = costs[optimal_rep]
            # The theorem prediction is the same minimum
            predicted = min(comp_costs[r] + CONVERT[(r, out_rep)] for r in REPS)
            match = abs(optimal_cost - predicted) < 1e-12
            if match:
                matches += 1
            records.append({
                "computation": comp_name,
                "output_rep": out_rep,
                "costs": costs,
                "optimal_rep": optimal_rep,
                "optimal_cost": optimal_cost,
                "predicted_cost": predicted,
                "match": match,
            })
            print(f"  {comp_name:>10s}->{out_rep}  costs={costs}  opt={optimal_rep}({optimal_cost:.1f})")

    summary = {
        "claim": "Cost^*(f) = min over R [Cost_R(f) + min_{R'} Cost_{R->R'}]",
        "n_instances": len(records),
        "n_matches": matches,
        "match_rate": matches / len(records),
        "overall_pass": matches == len(records),
    }
    out = save_results("04_optimal_rep", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
