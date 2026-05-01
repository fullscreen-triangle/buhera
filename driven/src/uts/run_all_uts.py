"""
Master runner for the unconstrained-subtasks-trajectory-completion
validation suite. Executes all 14 experiments, persists per-experiment
JSON, and writes an aggregate uts_master_results.json.
"""
from __future__ import annotations

import time

from .common import banner, save_results
from . import (
    validate_01_floor,
    validate_02_info_bound,
    validate_03_triple_equiv,
    validate_04_optimal_rep,
    validate_05_comp_mult,
    validate_06_unconstrained_subtask,
    validate_07_lg_decoupling,
    validate_08_backward_nav,
    validate_09_virtual_substates,
    validate_10_path_opacity,
    validate_11_multiplicativity,
    validate_12_cascade_saturation,
    validate_13_recursive_mult,
    validate_14_strict_hierarchy,
)


EXPERIMENTS = [
    ("01_floor",                validate_01_floor.validate),
    ("02_info_bound",           validate_02_info_bound.validate),
    ("03_triple_equiv",         validate_03_triple_equiv.validate),
    ("04_optimal_rep",          validate_04_optimal_rep.validate),
    ("05_comp_mult",            validate_05_comp_mult.validate),
    ("06_unconstrained_subtask",validate_06_unconstrained_subtask.validate),
    ("07_lg_decoupling",        validate_07_lg_decoupling.validate),
    ("08_backward_nav",         validate_08_backward_nav.validate),
    ("09_virtual_substates",    validate_09_virtual_substates.validate),
    ("10_path_opacity",         validate_10_path_opacity.validate),
    ("11_multiplicativity",     validate_11_multiplicativity.validate),
    ("12_cascade_saturation",   validate_12_cascade_saturation.validate),
    ("13_recursive_mult",       validate_13_recursive_mult.validate),
    ("14_strict_hierarchy",     validate_14_strict_hierarchy.validate),
]


def main():
    banner("UNCONSTRAINED-SUBTASKS-TRAJECTORY-COMPLETION VALIDATION SUITE")
    t0 = time.perf_counter()

    summaries = {}
    pass_flags = {}
    for name, fn in EXPERIMENTS:
        print()
        s = fn()
        summaries[name] = s
        pass_flags[name] = bool(s.get("overall_pass", False))

    elapsed = time.perf_counter() - t0
    n_pass = sum(pass_flags.values())
    overall_pass = n_pass == len(EXPERIMENTS)

    print()
    print("=" * 70)
    print("  AGGREGATE")
    print("=" * 70)
    for name in pass_flags:
        flag = "PASS" if pass_flags[name] else "FAIL"
        print(f"  {name:<28s} {flag}")
    print(f"\n  Pass rate: {n_pass}/{len(EXPERIMENTS)}")
    print(f"  Total time: {elapsed:.3f} s")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")

    master = {
        "suite": "unconstrained-subtasks-trajectory-completion",
        "n_experiments": len(EXPERIMENTS),
        "n_pass": n_pass,
        "overall_pass": overall_pass,
        "elapsed_seconds": elapsed,
        "per_experiment": pass_flags,
        "summaries": summaries,
    }
    out = save_results("master", master)
    print(f"  Saved master: {out}")
    return master


if __name__ == "__main__":
    main()
