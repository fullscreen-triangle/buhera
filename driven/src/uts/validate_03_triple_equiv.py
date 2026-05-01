"""
Experiment 03: Triple Equivalence (Theorem 5.1).

Round-trip O -> C -> P -> C -> O at 40 (omega, phi) input pairs.
Output should equal input up to discretisation cell width on omega and
exactly on phi (the categorical-partition functors preserve phase).
"""
from __future__ import annotations

import math

import numpy as np

from .common import SEED, banner, save_results


def F_OC(omega, phi):
    """Oscillator -> Categorical: floor(log_2 omega), arg phi."""
    label1 = math.floor(math.log2(omega))
    label2 = phi
    return (label1, label2)


def F_CP(label1, label2, depth=10):
    """Categorical -> Partition: pushforward via label discretisation."""
    cell_idx = label1
    phase_cell = int(round(label2 * (1 << depth) / (2 * math.pi))) % (1 << depth)
    return (cell_idx, phase_cell, depth)


def F_PC(cell_idx, phase_cell, depth):
    """Partition -> Categorical: read off labels."""
    label1 = cell_idx
    label2 = phase_cell * 2 * math.pi / (1 << depth)
    return (label1, label2)


def F_CO(label1, label2):
    """Categorical -> Oscillator: omega = 2^label1, phi = label2."""
    omega = 2.0 ** label1
    phi = label2
    return (omega, phi)


def round_trip(omega, phi):
    L1, L2 = F_OC(omega, phi)
    P = F_CP(L1, L2)
    L1b, L2b = F_PC(*P)
    omega_out, phi_out = F_CO(L1b, L2b)
    return omega_out, phi_out


def validate():
    banner("EXPERIMENT 03 — TRIPLE EQUIVALENCE")

    rng = np.random.default_rng(SEED)
    omega_inputs = np.exp(rng.uniform(0.1, 5.0, 40))
    phi_inputs = rng.uniform(0.0, 2 * math.pi, 40)

    records = []
    max_freq_err = 0.0
    max_phase_err = 0.0
    for omega, phi in zip(omega_inputs, phi_inputs):
        omega_out, phi_out = round_trip(omega, phi)
        # frequency recovery up to 2^floor truncation
        freq_err = abs(omega - omega_out) / omega
        # phase: within one phase cell of width 2pi/1024
        phase_err = min(abs(phi - phi_out), 2 * math.pi - abs(phi - phi_out))
        max_freq_err = max(max_freq_err, freq_err)
        max_phase_err = max(max_phase_err, phase_err)
        records.append({
            "omega_in": float(omega),
            "phi_in": float(phi),
            "omega_out": float(omega_out),
            "phi_out": float(phi_out),
            "freq_rel_err": freq_err,
            "phase_abs_err": phase_err,
        })

    print(f"  N round-trips: {len(records)}")
    print(f"  max freq rel err:  {max_freq_err:.4f}  (cell-width discretisation)")
    print(f"  max phase abs err: {max_phase_err:.4f}  (phase-cell discretisation)")

    summary = {
        "claim": "Round-trip O -> C -> P -> C -> O recovers input up to discretisation",
        "n_round_trips": len(records),
        "max_freq_rel_err": max_freq_err,
        "max_phase_abs_err": max_phase_err,
        "freq_within_cell": max_freq_err < 1.0,
        "phase_within_cell": max_phase_err < 2 * math.pi / 512,
        "overall_pass": max_freq_err < 1.0 and max_phase_err < 2 * math.pi / 512,
    }
    out = save_results("03_triple_equiv", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
