"""Run every validator, in order, and fail loudly if any claim is unmet.

Each validator asserts its claim and writes one figure panel into ../figures/.
This runner is what a reviewer executes to reproduce the paper's empirical
support in one command:

    python validation/run_all.py
"""

from __future__ import annotations

import importlib
import sys
import traceback

VALIDATORS = [
    "trajectory_emergence",
    "run_to_completion",
    "nondeterminism_provenance",
    "resolution_control",
]


def main() -> int:
    failures = []
    for name in VALIDATORS:
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        try:
            mod = importlib.import_module(name)
            mod.main()
        except Exception:  # noqa: BLE001 -- report and continue to next validator
            traceback.print_exc()
            failures.append(name)

    print("\n" + "=" * 66)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"ALL {len(VALIDATORS)} VALIDATORS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
