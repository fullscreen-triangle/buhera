"""V8: Reproducibility test — run the same operation 1000 times under the
same conditions; verify identical outputs and identical Q.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .common import SEED, banner, save_results


def run_op(seed: int, Q: int) -> tuple:
    rng = np.random.default_rng(seed)
    accum = 0
    for k in range(Q):
        accum = (accum * 1103515245 + 12345 + int(rng.integers(0, 2**31))) & 0xFFFFFFFF
    h = hashlib.sha256(accum.to_bytes(4, "little")).hexdigest()
    return accum, Q, h


def validate():
    banner("COE V8 — REPRODUCIBILITY")
    Q = 1024
    op_seed = 99
    n_runs = 1000
    outputs = set()
    weights = set()
    hashes = set()
    for _ in range(n_runs):
        out, w, h = run_op(op_seed, Q)
        outputs.add(out)
        weights.add(w)
        hashes.add(h)
    identical_outputs = len(outputs) == 1
    identical_weights = len(weights) == 1
    identical_hashes = len(hashes) == 1
    summary = {
        "claim": "Same input + same conditions => identical output and Q",
        "n_runs": n_runs,
        "Q": Q,
        "identical_outputs": identical_outputs,
        "identical_weights": identical_weights,
        "identical_hashes": identical_hashes,
        "overall_pass": identical_outputs and identical_weights and identical_hashes,
    }
    print(f"  N runs: {n_runs}  identical: {summary['overall_pass']}")
    out = save_results("08_reproducibility", {"summary": summary})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
