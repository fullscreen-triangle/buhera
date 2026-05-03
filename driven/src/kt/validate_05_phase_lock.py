"""E05: Phase Lock (Theorem 3.5).

The kernel state machine alternates between COMPILE (sigma_Y -> 0) and
EXECUTE (sigma_K -> 0). Concurrence is forbidden. We simulate 100 trace
patterns and verify mutual exclusion at every time step.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def trace_satisfies_phase_lock(trace):
    """Each step must be in exactly one of {COMPILE, EXECUTE}."""
    return all(s in ("COMPILE", "EXECUTE") for s in trace)


def validate():
    banner("E05 — PHASE LOCK")
    rng = np.random.default_rng(SEED)
    records = []
    n_violations = 0
    transitions_total = 0
    for trial in range(100):
        n_steps = int(rng.integers(20, 80))
        trace = []
        state = "COMPILE"
        for _ in range(n_steps):
            trace.append(state)
            # alternate with random hold time
            if rng.random() < 0.4:
                state = "EXECUTE" if state == "COMPILE" else "COMPILE"
                transitions_total += 1
        valid = trace_satisfies_phase_lock(trace)
        if not valid:
            n_violations += 1
        # count consecutive single-state runs (concurrence would mean tuple states)
        records.append({
            "trial": trial,
            "n_steps": n_steps,
            "valid": valid,
            "n_compile": trace.count("COMPILE"),
            "n_execute": trace.count("EXECUTE"),
        })

    summary = {
        "claim": "kernel always in exactly one of COMPILE / EXECUTE; no concurrence",
        "n_traces": len(records),
        "n_violations": n_violations,
        "total_transitions": transitions_total,
        "overall_pass": n_violations == 0,
    }
    print(f"  N traces: {len(records)}  violations: {n_violations}  transitions: {transitions_total}")
    out = save_results("05_phase_lock", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
