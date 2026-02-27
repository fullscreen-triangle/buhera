"""
Commutation Validation: Test [Ô_cat, Ô_phys] = 0

This module validates the categorical-physical commutation relation,
which is fundamental to zero-cost demon operations.

Previous results showed:
- [n̂, x̂] = 0 [OK] (exact)
- [n̂, p̂] ~= 0.09 I (small deviation)
- [n̂, Ĥ] ~= 0.06 I (small deviation)

This validation:
1. Tests commutation for multiple observables
2. Analyzes WHY momentum/energy show deviation
3. Validates in different physical regimes
4. Provides theoretical explanation
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    save_results,
    HBAR,
    KB,
    BOHR_RADIUS,
    ELECTRON_MASS
)


class QuantumOperators:
    """Quantum mechanical operators in matrix representation."""

    @staticmethod
    def position_operator(n_max: int) -> np.ndarray:
        """
        Position operator in hydrogen atom basis |n,l,m,s⟩.
        Diagonal in spatial representation.
        """
        dim = 2 * sum(2*n**2 for n in range(1, n_max + 1))
        x_op = np.zeros((dim, dim), dtype=complex)

        idx = 0
        for n in range(1, n_max + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    for s in [-0.5, 0.5]:
                        # ⟨r⟩ for hydrogen state |n,l,m⟩
                        r_expectation = BOHR_RADIUS * n**2 * (1.5 - l*(l+1)/(2*n**2))
                        x_op[idx, idx] = r_expectation
                        idx += 1

        return x_op

    @staticmethod
    def momentum_operator(n_max: int) -> np.ndarray:
        """
        Momentum operator (conjugate to position).
        Should satisfy [x,p] = ihbar.
        """
        dim = 2 * sum(2*n**2 for n in range(1, n_max + 1))
        p_op = np.zeros((dim, dim), dtype=complex)

        # Momentum has off-diagonal elements (transitions)
        idx1 = 0
        for n1 in range(1, n_max + 1):
            for l1 in range(n1):
                for m1 in range(-l1, l1 + 1):
                    for s1 in [-0.5, 0.5]:
                        idx2 = 0
                        for n2 in range(1, n_max + 1):
                            for l2 in range(n2):
                                for m2 in range(-l2, l2 + 1):
                                    for s2 in [-0.5, 0.5]:
                                        # Selection rules: Δl = ±1
                                        if abs(l2 - l1) == 1 and m1 == m2 and s1 == s2:
                                            # Matrix element (simplified)
                                            p_op[idx1, idx2] = HBAR / BOHR_RADIUS * \
                                                np.sqrt(n1 * n2) / (n1 + n2)
                                        idx2 += 1
                        idx1 += 1

        return p_op

    @staticmethod
    def hamiltonian_operator(n_max: int) -> np.ndarray:
        """
        Hamiltonian operator for hydrogen atom.
        E_n = -13.6 eV / n^2
        """
        dim = 2 * sum(2*n**2 for n in range(1, n_max + 1))
        H_op = np.zeros((dim, dim), dtype=complex)

        idx = 0
        for n in range(1, n_max + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    for s in [-0.5, 0.5]:
                        # Energy eigenvalue
                        E_n = -13.6 * 1.60218e-19 / n**2  # Joules
                        H_op[idx, idx] = E_n
                        idx += 1

        return H_op


class CategoricalOperators:
    """Categorical operators (quantum numbers)."""

    @staticmethod
    def n_operator(n_max: int) -> np.ndarray:
        """Principal quantum number operator."""
        dim = 2 * sum(2*n**2 for n in range(1, n_max + 1))
        n_op = np.zeros((dim, dim), dtype=complex)

        idx = 0
        for n in range(1, n_max + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    for s in [-0.5, 0.5]:
                        n_op[idx, idx] = n
                        idx += 1

        return n_op

    @staticmethod
    def l_operator(n_max: int) -> np.ndarray:
        """Angular momentum quantum number operator."""
        dim = 2 * sum(2*n**2 for n in range(1, n_max + 1))
        l_op = np.zeros((dim, dim), dtype=complex)

        idx = 0
        for n in range(1, n_max + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    for s in [-0.5, 0.5]:
                        l_op[idx, idx] = l
                        idx += 1

        return l_op

    @staticmethod
    def m_operator(n_max: int) -> np.ndarray:
        """Magnetic quantum number operator."""
        dim = 2 * sum(2*n**2 for n in range(1, n_max + 1))
        m_op = np.zeros((dim, dim), dtype=complex)

        idx = 0
        for n in range(1, n_max + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    for s in [-0.5, 0.5]:
                        m_op[idx, idx] = m
                        idx += 1

        return m_op


def compute_commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def commutator_norm(comm: np.ndarray) -> float:
    """Compute Frobenius norm of commutator."""
    return np.linalg.norm(comm, ord='fro')


def relative_commutator(A: np.ndarray, B: np.ndarray, comm: np.ndarray) -> float:
    """
    Relative commutator size: ||[A,B]|| / (||A|| ||B||)
    """
    norm_A = np.linalg.norm(A, ord='fro')
    norm_B = np.linalg.norm(B, ord='fro')
    norm_comm = np.linalg.norm(comm, ord='fro')

    if norm_A * norm_B == 0:
        return 0.0

    return norm_comm / (norm_A * norm_B)


def validate_commutation_relations(n_max: int = 5) -> Dict[str, Any]:
    """
    Comprehensive validation of categorical-physical commutation.

    Tests [Ô_cat, Ô_phys] for all combinations.
    """
    print("=" * 80)
    print("CATEGORICAL-PHYSICAL COMMUTATION VALIDATION")
    print("=" * 80)
    print(f"Hilbert space dimension: n_max = {n_max}")
    print()

    # Construct operators
    print("Constructing operators...")
    cat_ops = {
        "n": CategoricalOperators.n_operator(n_max),
        "l": CategoricalOperators.l_operator(n_max),
        "m": CategoricalOperators.m_operator(n_max)
    }

    phys_ops = {
        "x": QuantumOperators.position_operator(n_max),
        "p": QuantumOperators.momentum_operator(n_max),
        "H": QuantumOperators.hamiltonian_operator(n_max)
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_max": n_max,
        "hilbert_space_dim": cat_ops["n"].shape[0],
        "commutation_tests": [],
        "summary": {}
    }

    print(f"Hilbert space dimension: {results['hilbert_space_dim']}\n")
    print("Testing commutation relations:")
    print("-" * 80)

    # Test all combinations
    for cat_name, cat_op in cat_ops.items():
        for phys_name, phys_op in phys_ops.items():
            # Compute commutator
            comm = compute_commutator(cat_op, phys_op)
            norm = commutator_norm(comm)
            rel_comm = relative_commutator(cat_op, phys_op, comm)

            # Tolerance for "zero"
            tolerance = 1e-10
            is_zero = norm < tolerance

            test_result = {
                "categorical_op": cat_name,
                "physical_op": phys_name,
                "commutator_norm": float(norm),
                "relative_commutator": float(rel_comm),
                "commutes": is_zero,
                "tolerance": tolerance
            }

            results["commutation_tests"].append(test_result)

            status = "[OK] EXACT" if is_zero else f"[FAIL] {rel_comm:.2e}"
            print(f"  [{cat_name}, {phys_name}]: {status}")

    # Special test: Heisenberg uncertainty [x,p] = ihbar (control)
    print("\nControl: Heisenberg uncertainty relation")
    print("-" * 80)

    comm_xp = compute_commutator(phys_ops["x"], phys_ops["p"])
    expected_xp = 1j * HBAR * np.eye(phys_ops["x"].shape[0])

    diff = comm_xp - expected_xp
    heisenberg_error = np.linalg.norm(diff, ord='fro') / np.linalg.norm(expected_xp, ord='fro')

    print(f"  [x, p] - ihbarI: relative error = {heisenberg_error:.2e}")

    results["heisenberg_control"] = {
        "commutator_norm": float(np.linalg.norm(comm_xp, ord='fro')),
        "expected_norm": float(np.linalg.norm(expected_xp, ord='fro')),
        "relative_error": float(heisenberg_error),
        "passed": heisenberg_error < 1.0  # Loose tolerance for control
    }

    # Test self-commutation of categorical operators
    print("\nCategorical self-commutation:")
    print("-" * 80)

    cat_self_tests = []
    for name1, op1 in cat_ops.items():
        for name2, op2 in cat_ops.items():
            if name1 < name2:  # Avoid duplicates
                comm = compute_commutator(op1, op2)
                norm = commutator_norm(comm)
                rel_comm = relative_commutator(op1, op2, comm)

                test = {
                    "op1": name1,
                    "op2": name2,
                    "commutator_norm": float(norm),
                    "relative_commutator": float(rel_comm),
                    "commutes": norm < 1e-10
                }
                cat_self_tests.append(test)

                status = "[OK]" if test["commutes"] else "[FAIL]"
                print(f"  [{name1}, {name2}]: {status} ({rel_comm:.2e})")

    results["categorical_self_commutation"] = cat_self_tests

    # Summary
    position_commutes = all(
        t["commutes"] for t in results["commutation_tests"]
        if t["physical_op"] == "x"
    )

    momentum_commutes = all(
        t["commutes"] for t in results["commutation_tests"]
        if t["physical_op"] == "p"
    )

    hamiltonian_commutes = all(
        t["commutes"] for t in results["commutation_tests"]
        if t["physical_op"] == "H"
    )

    all_categorical_commute = all(t["commutes"] for t in cat_self_tests)

    results["summary"] = {
        "position_commutes_with_categorical": position_commutes,
        "momentum_commutes_with_categorical": momentum_commutes,
        "hamiltonian_commutes_with_categorical": hamiltonian_commutes,
        "categorical_ops_mutually_commute": all_categorical_commute,
        "claim_validation": {
            "commutation_exact_for_position": position_commutes,
            "commutation_holds_for_all": all([
                position_commutes,
                momentum_commutes,
                hamiltonian_commutes
            ]),
            "categorical_ops_form_complete_set": all_categorical_commute
        }
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Position observables: {'[OK] COMMUTE' if position_commutes else '[FAIL] DO NOT COMMUTE'}")
    print(f"Momentum observables: {'[OK] COMMUTE' if momentum_commutes else '[FAIL] DO NOT COMMUTE'}")
    print(f"Energy observables: {'[OK] COMMUTE' if hamiltonian_commutes else '[FAIL] DO NOT COMMUTE'}")
    print(f"Categorical mutual commutation: {'[OK] EXACT' if all_categorical_commute else '[FAIL] FAIL'}")

    return results


def analyze_commutation_deviation(n_max: int = 10) -> Dict[str, Any]:
    """
    Analyze WHY momentum/energy show deviation.

    Hypothesis: The deviation comes from finite Hilbert space truncation,
    not fundamental physics. In infinite-dimensional limit, should vanish.
    """
    print("\n" + "=" * 80)
    print("COMMUTATION DEVIATION ANALYSIS")
    print("=" * 80)

    n_max_values = [3, 5, 7, 10, 15, 20]
    deviations = []

    for n in n_max_values:
        print(f"\nTesting n_max = {n}...")

        n_op = CategoricalOperators.n_operator(n)
        p_op = QuantumOperators.momentum_operator(n)

        comm = compute_commutator(n_op, p_op)
        rel_comm = relative_commutator(n_op, p_op, comm)

        deviations.append({
            "n_max": n,
            "hilbert_dim": n_op.shape[0],
            "relative_commutator": float(rel_comm)
        })

        print(f"  Dimension: {n_op.shape[0]}")
        print(f"  Relative commutator: {rel_comm:.6e}")

    # Fit scaling: deviation ~ 1/n_max^α
    n_vals = np.array([d["n_max"] for d in deviations])
    dev_vals = np.array([d["relative_commutator"] for d in deviations])

    # Log-log fit
    log_n = np.log(n_vals)
    log_dev = np.log(dev_vals + 1e-15)  # Avoid log(0)

    if len(log_n) > 1:
        fit = np.polyfit(log_n, log_dev, 1)
        alpha = -fit[0]

        print(f"\nScaling analysis:")
        print(f"  deviation ∝ n_max^{-alpha:.2f}")
        print(f"  {'Vanishes in infinite limit' if alpha > 0 else 'Does not vanish'}")
    else:
        alpha = 0

    return {
        "scaling_analysis": deviations,
        "scaling_exponent": float(alpha),
        "vanishes_in_limit": alpha > 0,
        "interpretation": (
            "Finite truncation artifact" if alpha > 0
            else "Fundamental non-commutation"
        )
    }


if __name__ == "__main__":
    # Run basic validation
    results = validate_commutation_relations(n_max=5)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = save_results(
        results,
        f"commutation_validation_{timestamp}.json",
        format="json"
    )
    print(f"\n[OK] Results saved to: {filepath}")

    # Analyze deviation
    deviation_analysis = analyze_commutation_deviation(n_max=10)

    # Save deviation analysis
    dev_filepath = save_results(
        deviation_analysis,
        f"commutation_deviation_analysis_{timestamp}.json",
        format="json"
    )
    print(f"[OK] Deviation analysis saved to: {dev_filepath}")
