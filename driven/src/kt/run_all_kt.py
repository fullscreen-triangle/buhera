"""Master runner for the knowledge-thermodynamics validation suite."""
from __future__ import annotations

import time

from .common import banner, save_results
from . import (
    validate_01_floor,
    validate_02_method_floor,
    validate_03_uncertainty,
    validate_04_saturating_alloc,
    validate_05_phase_lock,
    validate_06_cell_type,
    validate_07_cell_disjoint,
    validate_08_domain_lattice,
    validate_09_floor_monotone,
    validate_10_mult_composition,
    validate_11_cascade_switching,
    validate_12_replication_bifurcation,
    validate_13_know_entropy,
    validate_14_federation,
    validate_15_marginal_reduction,
)

EXPERIMENTS = [
    ("01_floor",                     validate_01_floor.validate),
    ("02_method_floor",              validate_02_method_floor.validate),
    ("03_uncertainty",               validate_03_uncertainty.validate),
    ("04_saturating_alloc",          validate_04_saturating_alloc.validate),
    ("05_phase_lock",                validate_05_phase_lock.validate),
    ("06_cell_type",                 validate_06_cell_type.validate),
    ("07_cell_disjoint",             validate_07_cell_disjoint.validate),
    ("08_domain_lattice",            validate_08_domain_lattice.validate),
    ("09_floor_monotone",            validate_09_floor_monotone.validate),
    ("10_mult_composition",          validate_10_mult_composition.validate),
    ("11_cascade_switching",         validate_11_cascade_switching.validate),
    ("12_replication_bifurcation",   validate_12_replication_bifurcation.validate),
    ("13_know_entropy",              validate_13_know_entropy.validate),
    ("14_federation",                validate_14_federation.validate),
    ("15_marginal_reduction",        validate_15_marginal_reduction.validate),
]


def main():
    banner("KNOWLEDGE-THERMODYNAMICS VALIDATION SUITE")
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
        print(f"  {name:<32s} {flag}")
    print(f"\n  Pass rate: {n_pass}/{len(EXPERIMENTS)}")
    print(f"  Total time: {elapsed:.3f} s")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    master = {
        "suite": "knowledge-thermodynamics",
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
