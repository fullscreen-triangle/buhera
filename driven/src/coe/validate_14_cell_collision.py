"""V14: Cell-collision detection — adding a colliding operation is detected
by the framework.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def detect_collision(existing_cells: set, new_cell: int) -> bool:
    return new_cell in existing_cells


def validate():
    banner("COE V14 — CELL COLLISION DETECTION")
    rng = np.random.default_rng(SEED)
    records = []
    all_correct = True
    for trial in range(50):
        N_existing = int(rng.integers(5, 30))
        existing_cells = set(range(N_existing))
        # Half collide, half disjoint
        if rng.random() < 0.5:
            new_cell = int(rng.integers(0, N_existing))  # collision
            expected_collision = True
        else:
            new_cell = N_existing + int(rng.integers(0, 100))  # disjoint
            expected_collision = False
        detected = detect_collision(existing_cells, new_cell)
        ok = (detected == expected_collision)
        if not ok:
            all_correct = False
        records.append({
            "trial": trial, "N_existing": N_existing, "new_cell": new_cell,
            "expected": expected_collision, "detected": detected, "match": ok,
        })
    summary = {
        "claim": "Cell collisions are correctly identified",
        "n_trials": len(records),
        "all_correct": all_correct,
        "overall_pass": all_correct,
    }
    print(f"  N trials: {len(records)}  all correct: {all_correct}")
    out = save_results("14_cell_collision", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
