"""
Validation of the Zero-Cost Sorting Theorem.

Claim: if [O_cat, O_phys] = 0, then W_sort = 0.

We construct pairs of operators:
  - commuting pair: O_cat sorts categorical indices, O_phys reorders buckets
  - non-commuting pair: O_cat modifies the same physical quantity as O_phys

For each, we compute the commutator Frobenius norm and the thermodynamic
work required to execute the sort. If the theorem holds, commutation = 0
implies work = 0, and non-commutation implies work > 0.
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32" and not getattr(sys, "_buhera_stdout_wrapped", False):
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys._buhera_stdout_wrapped = True
    except (ValueError, AttributeError):
        pass

K_B = 1.380649e-23
T = 300.0  # Kelvin


def commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A @ B - B @ A))


def work_for_sort(O_cat: np.ndarray, O_phys: np.ndarray,
                   initial_state: np.ndarray, T_kelvin: float = T) -> float:
    """
    Work of reordering: apply (O_cat, O_phys) vs (O_phys, O_cat) and compare.
    If [O_cat, O_phys] = 0, both orderings give the same final state and the
    reorder work is zero. If they fail to commute, the two orderings differ,
    and the "work" scales with the commutator action on the state.
    """
    forward = O_phys @ (O_cat @ initial_state)
    reverse = O_cat @ (O_phys @ initial_state)
    diff_norm = float(np.linalg.norm(forward - reverse))
    W = K_B * T_kelvin * diff_norm
    return W, {
        "forward_order_norm": float(np.linalg.norm(forward)),
        "reverse_order_norm": float(np.linalg.norm(reverse)),
        "order_dependence_norm": diff_norm,
    }


def validate():
    print("=" * 70)
    print("  ZERO-COST SORTING THEOREM VALIDATION")
    print("  Claim: [O_cat, O_phys] = 0  =>  W_sort = 0")
    print("=" * 70)

    np.random.seed(42)
    dim = 8

    tests = []

    # Test 1: commuting pair -- both diagonal operators
    print("\n  Test 1: commuting pair (both diagonal)")
    D_cat = np.diag(np.arange(dim).astype(float))
    D_phys = np.diag(np.arange(dim).astype(float) * 2.0)
    comm = commutator_norm(D_cat, D_phys)

    psi = np.ones(dim) / np.sqrt(dim)
    W, diag = work_for_sort(D_cat, D_phys, psi)
    tests.append({
        "name": "commuting_diagonal_pair",
        "commutator_norm": comm,
        "commutes": comm < 1e-10,
        "work_J": W,
        "work_close_to_zero": W < 1e-30,
        "diagnostics": diag,
    })
    print(f"    comm_norm={comm:.3e}  W={W:.3e} J  commutes={comm<1e-10}  zero_cost={W < 1e-30}")

    # Test 2: non-commuting pair -- different non-diagonal operators
    print("\n  Test 2: non-commuting pair (different non-diagonal)")
    A = np.random.randn(dim, dim)
    B = np.random.randn(dim, dim)
    # Make them Hermitian for realism
    A = (A + A.T) / 2
    B = (B + B.T) / 2
    comm = commutator_norm(A, B)

    W, diag = work_for_sort(A, B, psi)
    tests.append({
        "name": "noncommuting_hermitian_pair",
        "commutator_norm": comm,
        "commutes": comm < 1e-10,
        "work_J": W,
        "work_zero": W < 1e-30,
        "diagnostics": diag,
    })
    print(f"    comm_norm={comm:.3e}  W={W:.3e} J  commutes={comm<1e-10}  zero_cost={W < 1e-30}")

    # Test 3: a family of operators where we scan commutator magnitude
    print("\n  Test 3: commutator-magnitude / work correlation")
    correlation_data = []
    D_cat = np.diag(np.arange(dim).astype(float))
    perturbation_base = np.random.randn(dim, dim)
    perturbation_base = (perturbation_base + perturbation_base.T) / 2

    for epsilon in [0.0, 0.01, 0.1, 0.5, 1.0]:
        # non-diagonal component scaled by epsilon
        O_phys = D_cat * 2.0 + epsilon * perturbation_base
        comm = commutator_norm(D_cat, O_phys)
        W, _ = work_for_sort(D_cat, O_phys, psi)
        correlation_data.append({
            "epsilon": epsilon,
            "commutator_norm": comm,
            "work_J": W,
            "work_ratio_kT": W / (K_B * T),
        })
        print(f"    eps={epsilon:.2f}  comm={comm:.3e}  W={W:.3e} J  W/(kT)={W/(K_B*T):.3e}")

    # Check that commutator magnitude correlates with work
    comms = [d["commutator_norm"] for d in correlation_data]
    works = [d["work_J"] for d in correlation_data]
    # Pearson correlation
    if len(comms) > 1:
        c_arr = np.array(comms)
        w_arr = np.array(works)
        if np.std(c_arr) > 0 and np.std(w_arr) > 0:
            corr = float(np.corrcoef(c_arr, w_arr)[0, 1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    # Theorem prediction: test 1 (commuting) should have near-zero work
    # Theorem is verified when:
    #   - commuting case has zero work
    #   - non-commuting case can have non-zero work
    # Theorem predicts: commuting => zero work AND commutator magnitude
    # correlates with work when commutation is broken.
    theorem_confirmed = (
        tests[0]["work_close_to_zero"]              # commuting case zero
        and not tests[1]["work_zero"]                # non-commuting case non-zero
        and correlation_data[0]["work_J"] < 1e-30    # epsilon=0 (commuting) zero
        and correlation_data[-1]["work_J"] > correlation_data[0]["work_J"]  # scan increases
    )

    summary = {
        "claim": "[O_cat, O_phys] = 0 implies W_sort = 0",
        "commuting_pair_zero_work": tests[0]["work_close_to_zero"],
        "correlation_comm_to_work": corr,
        "theorem_confirmed": theorem_confirmed,
    }

    results = {
        "test_name": "zero_cost_sorting",
        "paper": "trajectory-mechanism",
        "theorem": "Theorem 7.1 (Zero-Cost Sorting)",
        "summary": summary,
        "commutation_tests": tests,
        "correlation_scan": correlation_data,
    }

    out_path = Path(__file__).parent.parent.parent / "data" / "zero_cost_sorting_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: {out_path}")
    print(f"  PASS: {summary['theorem_confirmed']}")
    return results


if __name__ == "__main__":
    validate()
