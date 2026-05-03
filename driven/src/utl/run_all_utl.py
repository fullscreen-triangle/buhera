"""Master runner for the UTL validation suite (15 experiments)."""
from __future__ import annotations

import time

from .common import banner, save_results
from . import (
    validate_01_universal_law,
    validate_02_mm1,
    validate_03_jackson,
    validate_04_cascade,
    validate_05_cache_extinction,
    validate_06_phase_coherence,
    validate_07_five_regimes,
    validate_08_bifurcation,
    validate_09_critical_slowing,
    validate_10_load_indicator,
    validate_11_federation,
    validate_12_saturation,
    validate_13_lag_estimator,
    validate_14_coupling_estimator,
    validate_15_cross_arch,
)

EXPERIMENTS = [
    ("01_universal_law",     validate_01_universal_law.validate),
    ("02_mm1",               validate_02_mm1.validate),
    ("03_jackson",           validate_03_jackson.validate),
    ("04_cascade",           validate_04_cascade.validate),
    ("05_cache_extinction",  validate_05_cache_extinction.validate),
    ("06_phase_coherence",   validate_06_phase_coherence.validate),
    ("07_five_regimes",      validate_07_five_regimes.validate),
    ("08_bifurcation",       validate_08_bifurcation.validate),
    ("09_critical_slowing",  validate_09_critical_slowing.validate),
    ("10_load_indicator",    validate_10_load_indicator.validate),
    ("11_federation",        validate_11_federation.validate),
    ("12_saturation",        validate_12_saturation.validate),
    ("13_lag_estimator",     validate_13_lag_estimator.validate),
    ("14_coupling_estimator",validate_14_coupling_estimator.validate),
    ("15_cross_arch",        validate_15_cross_arch.validate),
]


def main():
    banner("UTL — UNIVERSAL OS TRANSPORT LAW VALIDATION SUITE")
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
    print("  UTL AGGREGATE")
    print("=" * 70)
    for name in pass_flags:
        flag = "PASS" if pass_flags[name] else "FAIL"
        print(f"  {name:<28s} {flag}")
    print(f"\n  Pass rate: {n_pass}/{len(EXPERIMENTS)}")
    print(f"  Total time: {elapsed:.3f} s")
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    master = {
        "suite": "universal-os-transport-law",
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
