"""
Master Validation Runner for Buhera OS

Executes all validation experiments and generates comprehensive report.

Usage:
    python driven/src/run_all_validations.py
    python driven/src/run_all_validations.py --quick    # Fast mode
    python driven/src/run_all_validations.py --full     # Comprehensive mode
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sorting.validate_sorting import validate_sorting_complexity, extrapolate_to_large_n, validate_index_retrieval
from ipc.validate_ipc import validate_ipc_performance
from commutation.validate_commutation import (
    validate_commutation_relations,
    analyze_commutation_deviation
)
from core import save_results


def run_all_validations(mode: str = "standard"):
    """
    Run all Buhera OS validations.

    Args:
        mode: "quick", "standard", or "full"
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "BUHERA OS COMPREHENSIVE VALIDATION")
    print("=" * 80)
    print(f"Mode: {mode.upper()}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_results = {
        "timestamp": timestamp,
        "mode": mode,
        "validations": {}
    }

    # Configure based on mode
    if mode == "quick":
        sort_sizes = [100, 1000, 10000]
        sort_trials = 3
        ipc_sizes = [1024, 102400, 10240000]  # 1KB, 100KB, 10MB
        ipc_trials = 3
        comm_n_max = 3
    elif mode == "full":
        sort_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
        sort_trials = 10
        ipc_sizes = [1024, 10240, 102400, 1024000, 10240000, 51200000, 102400000, 512000000]
        ipc_trials = 10
        comm_n_max = 10
    else:  # standard
        sort_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
        sort_trials = 5
        ipc_sizes = [1024, 10240, 102400, 1024000, 10240000, 51200000, 102400000]
        ipc_trials = 5
        comm_n_max = 5

    # 1a. Index Retrieval Validation (clean O(log_3 N) demonstration)
    print("\n" + "#" * 80)
    print(" VALIDATION 1/4: INDEX RETRIEVAL COMPLEXITY O(log_3 N)")
    print("#" * 80 + "\n")

    try:
        retrieval_results = validate_index_retrieval(
            sizes=[27, 81, 243, 729, 2187, 6561, 19683, 59049],
            queries_per_size=30,
        )
        master_results["validations"]["index_retrieval"] = {
            "status": "SUCCESS",
            "r_squared": retrieval_results["complexity_fit"]["r_squared"],
            "validated": retrieval_results["complexity_fit"]["validated_o_log3_n"],
            "file": f"index_retrieval_validation_{timestamp}.json"
        }
        save_results(retrieval_results, f"index_retrieval_validation_{timestamp}.json")
        print("\n[OK] Index retrieval validation complete")
    except Exception as e:
        print(f"\n[FAIL] Index retrieval FAILED: {e}")
        master_results["validations"]["index_retrieval"] = {"status": "FAILED", "error": str(e)}

    # 1b. Sorting Validation (address navigation benchmark)
    print("\n" + "#" * 80)
    print(" VALIDATION 2/4: SORTING PERFORMANCE")
    print("#" * 80 + "\n")

    try:
        sorting_results = validate_sorting_complexity(
            sizes=sort_sizes,
            n_trials=sort_trials,
            distributions=["random", "reversed", "gaussian"]
        )

        # Extrapolate to large N
        if mode in ["standard", "full"]:
            extrapolation = extrapolate_to_large_n(
                sorting_results,
                target_sizes=[500000, 1000000, 10000000, 100000000]
            )
            sorting_results["extrapolation"] = extrapolation

        master_results["validations"]["sorting"] = {
            "status": "SUCCESS",
            "summary": sorting_results["summary"],
            "file": f"sorting_validation_{timestamp}.json"
        }

        # Save sorting results
        save_results(sorting_results, f"sorting_validation_{timestamp}.json")

        print("\n[OK] Sorting validation complete")

    except Exception as e:
        print(f"\n[FAIL] Sorting validation FAILED: {e}")
        master_results["validations"]["sorting"] = {
            "status": "FAILED",
            "error": str(e)
        }

    # 2. IPC Validation
    print("\n" + "#" * 80)
    print(" VALIDATION 3/4: INTER-PROCESS COMMUNICATION")
    print("#" * 80 + "\n")

    try:
        ipc_results = validate_ipc_performance(
            data_sizes=ipc_sizes,
            n_trials=ipc_trials
        )

        master_results["validations"]["ipc"] = {
            "status": "SUCCESS",
            "summary": ipc_results["summary"],
            "file": f"ipc_validation_{timestamp}.json"
        }

        # Save IPC results
        save_results(ipc_results, f"ipc_validation_{timestamp}.json")

        print("\n[OK] IPC validation complete")

    except Exception as e:
        print(f"\n[FAIL] IPC validation FAILED: {e}")
        master_results["validations"]["ipc"] = {
            "status": "FAILED",
            "error": str(e)
        }

    # 3. Commutation Validation
    print("\n" + "#" * 80)
    print(" VALIDATION 4/4: CATEGORICAL-PHYSICAL COMMUTATION")
    print("#" * 80 + "\n")

    try:
        commutation_results = validate_commutation_relations(n_max=comm_n_max)

        # Deviation analysis (only in standard/full mode)
        if mode in ["standard", "full"]:
            deviation_analysis = analyze_commutation_deviation(n_max=comm_n_max)
            commutation_results["deviation_analysis"] = deviation_analysis

        master_results["validations"]["commutation"] = {
            "status": "SUCCESS",
            "summary": commutation_results["summary"],
            "file": f"commutation_validation_{timestamp}.json"
        }

        # Save commutation results
        save_results(commutation_results, f"commutation_validation_{timestamp}.json")

        print("\n[OK] Commutation validation complete")

    except Exception as e:
        print(f"\n[FAIL] Commutation validation FAILED: {e}")
        master_results["validations"]["commutation"] = {
            "status": "FAILED",
            "error": str(e)
        }

    # Generate master report
    print("\n" + "=" * 80)
    print(" GENERATING MASTER REPORT")
    print("=" * 80 + "\n")

    master_results["completion_time"] = datetime.now().isoformat()
    master_results["overall_status"] = "SUCCESS" if all(
        v.get("status") == "SUCCESS"
        for v in master_results["validations"].values()
    ) else "PARTIAL"

    # Generate summary
    generate_validation_report(master_results)

    # Save master results
    master_filepath = save_results(
        master_results,
        f"master_validation_report_{timestamp}.json"
    )

    print(f"\n[OK] Master report saved to: {master_filepath}")
    print("\n" + "=" * 80)
    print(" VALIDATION COMPLETE")
    print("=" * 80)


def generate_validation_report(results: dict):
    """Generate human-readable validation report."""
    print("\n" + "=" * 80)
    print(" " * 25 + "VALIDATION REPORT")
    print("=" * 80 + "\n")

    # Overall status
    status_symbol = "[OK]" if results["overall_status"] == "SUCCESS" else "[WARN]"
    print(f"{status_symbol} Overall Status: {results['overall_status']}\n")

    # Individual validations
    print("Individual Validations:")
    print("-" * 80)

    for name, validation in results["validations"].items():
        status = validation.get("status", "UNKNOWN")
        symbol = "[OK]" if status == "SUCCESS" else "[FAIL]"

        print(f"\n{symbol} {name.upper()}: {status}")

        if status == "SUCCESS" and "summary" in validation:
            summary = validation["summary"]

            if name == "sorting":
                print(f"  • Mean speedup: {summary.get('overall_mean_speedup', 0):.2f}x")
                print(f"  • Best speedup: {summary.get('best_speedup', 0):.2f}x "
                      f"(N={summary.get('best_speedup_size', 0):,})")
                print(f"  • Complexity validated: "
                      f"{'[OK]' if summary.get('claim_validation', {}).get('categorical_is_log3_n') else '[FAIL]'}")

            elif name == "ipc":
                print(f"  • Best speedup vs pipe: {summary.get('best_speedup_vs_pipe', 0):.2f}x")
                print(f"  • Best speedup vs shared memory: "
                      f"{summary.get('best_speedup_vs_shared_mem', 0):.2f}x")
                print(f"  • 100x speedup achieved: "
                      f"{'[OK]' if summary.get('claim_validation', {}).get('100x_speedup_achieved') else '[FAIL]'}")
                print(f"  • Latency constant: "
                      f"{'[OK]' if summary.get('latency_is_constant') else '[FAIL]'}")

            elif name == "commutation":
                print(f"  • Position commutes: "
                      f"{'[OK]' if summary.get('position_commutes_with_categorical') else '[FAIL]'}")
                print(f"  • Momentum commutes: "
                      f"{'[OK]' if summary.get('momentum_commutes_with_categorical') else '[FAIL]'}")
                print(f"  • Hamiltonian commutes: "
                      f"{'[OK]' if summary.get('hamiltonian_commutes_with_categorical') else '[FAIL]'}")

        elif status == "FAILED":
            print(f"  Error: {validation.get('error', 'Unknown error')}")

    # Claims validation summary
    print("\n" + "=" * 80)
    print(" CLAIMS VALIDATION SUMMARY")
    print("=" * 80 + "\n")

    claims = []

    # Sorting claims
    if "sorting" in results["validations"] and results["validations"]["sorting"].get("status") == "SUCCESS":
        sort_summary = results["validations"]["sorting"]["summary"]
        claims.extend([
            ("Categorical sorting is O(log_3 N)",
             sort_summary.get("claim_validation", {}).get("categorical_is_log3_n", False)),
            ("Speedup increases with N",
             sort_summary.get("claim_validation", {}).get("speedup_increases_with_n", False)),
            ("Energy dramatically lower",
             sort_summary.get("claim_validation", {}).get("energy_dramatically_lower", False)),
        ])

    # IPC claims
    if "ipc" in results["validations"] and results["validations"]["ipc"].get("status") == "SUCCESS":
        ipc_summary = results["validations"]["ipc"]["summary"]
        claims.extend([
            ("100x IPC speedup achieved",
             ipc_summary.get("claim_validation", {}).get("100x_speedup_achieved", False)),
            ("IPC speedup increases with data size",
             ipc_summary.get("claim_validation", {}).get("speedup_increases_with_size", False)),
            ("IPC latency independent of size",
             ipc_summary.get("claim_validation", {}).get("latency_independent_of_size", False)),
        ])

    # Commutation claims
    if "commutation" in results["validations"] and results["validations"]["commutation"].get("status") == "SUCCESS":
        comm_summary = results["validations"]["commutation"]["summary"]
        claims.extend([
            ("Position observables commute exactly",
             comm_summary.get("claim_validation", {}).get("commutation_exact_for_position", False)),
            ("Categorical operators mutually commute",
             comm_summary.get("claim_validation", {}).get("categorical_ops_form_complete_set", False)),
        ])

    for claim, validated in claims:
        symbol = "[OK]" if validated else "[FAIL]"
        print(f"{symbol} {claim}")

    # Overall verdict
    validated_count = sum(1 for _, v in claims if v)
    total_count = len(claims)

    print(f"\n{validated_count}/{total_count} claims validated ({100*validated_count/total_count:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Buhera OS validations")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="standard",
        help="Validation mode: quick, standard, or full"
    )

    # Legacy flags
    parser.add_argument("--quick", action="store_true", help="Quick mode (legacy)")
    parser.add_argument("--full", action="store_true", help="Full mode (legacy)")

    args = parser.parse_args()

    # Resolve mode
    if args.quick:
        mode = "quick"
    elif args.full:
        mode = "full"
    else:
        mode = args.mode

    run_all_validations(mode=mode)
