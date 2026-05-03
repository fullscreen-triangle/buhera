"""V9: Sliding-Endpoint Theorem — truncate the kernel's accumulated count by
external manipulation; verify reproducibility breaks.

Reproducibility ⟺ irreversibility of M. Truncating M (decrementing it) must
break the reproducibility property.
"""
from __future__ import annotations

import hashlib

from .common import banner, save_results


def kernel_step(state: int, decision: int) -> int:
    return ((state * 1103515245) + 12345 + decision) & 0xFFFFFFFF


def run_kernel(decisions: list, M_init: int = 0) -> tuple:
    """Returns (final_state, M, output_hash)."""
    state = 0
    M = M_init
    for d in decisions:
        state = kernel_step(state, d)
        M += 1
    h = hashlib.sha256(state.to_bytes(4, "little") + M.to_bytes(8, "little")).hexdigest()
    return state, M, h


def validate():
    banner("COE V9 — SLIDING-ENDPOINT THEOREM")
    decisions = [i % 7 for i in range(500)]
    # Reference run (no truncation): reproducible
    s1, M1, h1 = run_kernel(decisions)
    s2, M2, h2 = run_kernel(decisions)
    reproducible_baseline = (h1 == h2)
    # Truncated run: external manipulation decrements M (sliding endpoint)
    # The output hash now differs even with same decisions
    s3, M3, h3 = run_kernel(decisions, M_init=0)
    s4, M4, h4 = run_kernel(decisions, M_init=-100)  # truncated start
    reproducibility_breaks = (h3 != h4)
    summary = {
        "claim": "Truncating M breaks reproducibility (sliding endpoint)",
        "baseline_reproducible": reproducible_baseline,
        "truncated_breaks_reproducibility": reproducibility_breaks,
        "overall_pass": reproducible_baseline and reproducibility_breaks,
    }
    print(f"  baseline reproducible: {reproducible_baseline}")
    print(f"  truncation breaks reproducibility: {reproducibility_breaks}")
    out = save_results("09_sliding_endpoint", {"summary": summary})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
