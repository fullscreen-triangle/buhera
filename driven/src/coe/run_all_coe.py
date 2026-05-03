"""Master runner for the COE validation suite (15 experiments)."""
from __future__ import annotations

import time

from .common import banner, save_results
from . import (
    validate_01_time_count_identity,
    validate_02_linearity,
    validate_03_route_residue,
    validate_04_route_confinement,
    validate_05_route_negation,
    validate_06_three_route_equivalence,
    validate_07_mtic_equivalence,
    validate_08_reproducibility,
    validate_09_sliding_endpoint,
    validate_10_rewind_as_forward,
    validate_11_monotone_log,
    validate_12_api_substitutability,
    validate_13_cell_stability,
    validate_14_cell_collision,
    validate_15_cross_arch,
)

EXPERIMENTS = [
    ("01_time_count_identity",   validate_01_time_count_identity.validate),
    ("02_linearity",             validate_02_linearity.validate),
    ("03_route_residue",         validate_03_route_residue.validate),
    ("04_route_confinement",     validate_04_route_confinement.validate),
    ("05_route_negation",        validate_05_route_negation.validate),
    ("06_three_route_equiv",     validate_06_three_route_equivalence.validate),
    ("07_mtic_equivalence",      validate_07_mtic_equivalence.validate),
    ("08_reproducibility",       validate_08_reproducibility.validate),
    ("09_sliding_endpoint",      validate_09_sliding_endpoint.validate),
    ("10_rewind_as_forward",     validate_10_rewind_as_forward.validate),
    ("11_monotone_log",          validate_11_monotone_log.validate),
    ("12_api_substitutability",  validate_12_api_substitutability.validate),
    ("13_cell_stability",        validate_13_cell_stability.validate),
    ("14_cell_collision",        validate_14_cell_collision.validate),
    ("15_cross_arch",            validate_15_cross_arch.validate),
]


def main():
    banner("COE — COMPUTATIONAL OPERATIONS EQUIVALENCE VALIDATION SUITE")
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
    print("  COE AGGREGATE")
    print("=" * 70)
    for name in pass_flags:
        flag = "PASS" if pass_flags[name] else "FAIL"
        print(f"  {name:<28s} {flag}")
    print(f"\n  Pass rate: {n_pass}/{len(EXPERIMENTS)}")
    print(f"  Total time: {elapsed:.3f} s")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    master = {
        "suite": "computational-operations-equivalence",
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
