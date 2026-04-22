"""
Master runner: all trajectory-mechanism paper validations + the new
continuous-embedding virtual sub-states tests.

Each individual validator saves its own JSON. This script additionally
writes driven/data/trajectory_master_results.json aggregating a one-line
summary of every test.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32" and not getattr(sys, "_buhera_stdout_wrapped", False):
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys._buhera_stdout_wrapped = True
    except (ValueError, AttributeError):
        pass

from . import validate_triple_equivalence as triple_eq
from . import validate_processor_oscillator as proc_osc
from . import validate_penultimate as penultimate
from . import validate_complexity_hierarchy as hierarchy
from . import validate_zero_cost_sorting as zero_cost
from . import validate_lunar_mechanics as lunar

# continuous-embedding additions
sys.path.insert(0, str(Path(__file__).parent.parent))
from embedding import validate_virtual_substates as virtual_substates  # noqa: E402


def main():
    print("\n" + "=" * 78)
    print("  BUHERA FRAMEWORK — MASTER VALIDATION RUN")
    print("  trajectory-mechanism + continuous-embedding supplements")
    print("=" * 78 + "\n")

    t_start = time.perf_counter()

    all_results = {}

    all_results["triple_equivalence"] = triple_eq.validate()
    all_results["processor_oscillator"] = proc_osc.validate()
    all_results["penultimate_state"] = penultimate.validate()
    all_results["complexity_hierarchy"] = hierarchy.validate()
    all_results["zero_cost_sorting"] = zero_cost.validate()
    all_results["lunar_mechanics"] = lunar.validate()
    all_results["virtual_substates"] = virtual_substates.validate()

    t_total = time.perf_counter() - t_start

    print("\n" + "=" * 78)
    print("  MASTER SUMMARY")
    print("=" * 78)

    summary = {}
    all_pass = True
    for name, r in all_results.items():
        s = r.get("summary", {})
        passed = s.get("overall_pass", s.get("all_passed", s.get("theorem_confirmed",
                                                                  s.get("test_passed", False))))
        summary[name] = {
            "paper": r.get("paper", ""),
            "theorem": r.get("theorem") or r.get("theorems") or r.get("section", ""),
            "summary": s,
            "passed": bool(passed),
        }
        all_pass = all_pass and bool(passed)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {name:30s}  {r.get('theorem', r.get('section', ''))}")

    master = {
        "total_time_seconds": t_total,
        "overall_pass": all_pass,
        "results_summary": summary,
        "result_files": [
            "driven/data/triple_equivalence_results.json",
            "driven/data/processor_oscillator_results.json",
            "driven/data/penultimate_state_results.json",
            "driven/data/complexity_hierarchy_results.json",
            "driven/data/zero_cost_sorting_results.json",
            "driven/data/lunar_mechanics_results.json",
            "driven/data/virtual_substates_results.json",
        ],
    }

    out_path = Path(__file__).parent.parent.parent / "data" / "trajectory_master_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    print(f"\n  Total time: {t_total:.2f}s")
    print(f"  Overall PASS: {all_pass}")
    print(f"  Master saved: {out_path}")
    return master


if __name__ == "__main__":
    main()
