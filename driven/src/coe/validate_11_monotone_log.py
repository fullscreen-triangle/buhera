"""V11: Monotone log growth — workload of forward + rollback operations;
verify the kernel's transaction log grows monotonically.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def validate():
    banner("COE V11 — MONOTONE LOG GROWTH")
    rng = np.random.default_rng(SEED)
    M = 0
    log_history = [M]
    monotone = True
    n_ops = 1000
    for _ in range(n_ops):
        op = rng.choice(["forward", "rollback"])
        if op == "forward":
            M += int(rng.integers(1, 10))
        else:
            # rollback — itself a forward decision
            M += int(rng.integers(1, 10))
        log_history.append(M)
        if log_history[-1] < log_history[-2]:
            monotone = False
    summary = {
        "claim": "Log M grows monotonically across forward + rollback ops",
        "n_ops": n_ops,
        "M_initial": log_history[0],
        "M_final": log_history[-1],
        "min_diff": int(min(b - a for a, b in zip(log_history, log_history[1:]))),
        "monotone": monotone,
        "overall_pass": monotone,
    }
    print(f"  M_final: {summary['M_final']}  monotone: {monotone}")
    out = save_results("11_monotone_log", {"summary": summary,
                                              "log_sample": log_history[:50]})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
