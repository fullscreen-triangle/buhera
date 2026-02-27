"""
Sorting Validation: Test the 10^6x speedup claim

This module validates:
1. Categorical sorting complexity is O(log_3 N) vs O(N log N)
2. Speedup scales with problem size
3. Energy consumption is dramatically lower
4. Results are identical (correctness)
"""

import numpy as np
from typing import Dict, List, Any
import time
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    CategoricalProcessor,
    generate_test_data,
    save_results,
    KB
)


def validate_sorting_complexity(
    sizes: List[int] = None,
    n_trials: int = 10,
    distributions: List[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive sorting validation across multiple problem sizes.

    Args:
        sizes: List of array sizes to test
        n_trials: Number of trials per size for statistical reliability
        distributions: Data distributions to test

    Returns:
        Complete validation results with speedup, energy, complexity analysis
    """
    if sizes is None:
        sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]

    if distributions is None:
        distributions = ["random", "reversed", "gaussian"]

    print("=" * 80)
    print("BUHERA OS SORTING VALIDATION")
    print("=" * 80)
    print(f"Testing sizes: {sizes}")
    print(f"Trials per size: {n_trials}")
    print(f"Distributions: {distributions}")
    print()

    processor = CategoricalProcessor()
    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "sizes": sizes,
            "n_trials": n_trials,
            "distributions": distributions
        },
        "benchmarks": [],
        "complexity_analysis": {},
        "summary": {}
    }

    # Run benchmarks for each size and distribution
    for size in sizes:
        print(f"\nTesting N = {size:,}")
        print("-" * 40)

        for dist in distributions:
            print(f"  Distribution: {dist}")

            categorical_times = []
            conventional_times = []
            categorical_ops = []
            conventional_ops = []
            categorical_energy = []
            conventional_energy = []
            speedups = []

            for trial in range(n_trials):
                # Generate test data
                data = generate_test_data(size, dist)

                # Categorical sort
                cat_result, cat_metrics = processor.categorical_sort(data.copy())
                categorical_times.append(cat_metrics["time_s"])
                categorical_ops.append(cat_metrics["total_categorical_ops"])
                categorical_energy.append(cat_metrics["total_energy_J"])

                # Conventional sort
                conv_result, conv_metrics = processor.conventional_sort(data.copy())
                conventional_times.append(conv_metrics["time_s"])
                conventional_ops.append(conv_metrics["comparisons"])
                conventional_energy.append(conv_metrics["energy_J"])

                # Verify correctness
                assert np.allclose(cat_result, conv_result), \
                    f"Results don't match! Trial {trial}"

                # Calculate speedup
                speedup = conv_metrics["time_s"] / cat_metrics["time_s"]
                speedups.append(speedup)

            # Compute statistics
            benchmark_result = {
                "size": size,
                "distribution": dist,
                "categorical": {
                    "mean_time_s": float(np.mean(categorical_times)),
                    "std_time_s": float(np.std(categorical_times)),
                    "mean_ops": float(np.mean(categorical_ops)),
                    "std_ops": float(np.std(categorical_ops)),
                    "mean_energy_J": float(np.mean(categorical_energy)),
                    "theoretical_ops": cat_metrics["navigation_steps"] + 1
                },
                "conventional": {
                    "mean_time_s": float(np.mean(conventional_times)),
                    "std_time_s": float(np.std(conventional_times)),
                    "mean_ops": float(np.mean(conventional_ops)),
                    "std_ops": float(np.std(conventional_ops)),
                    "mean_energy_J": float(np.mean(conventional_energy)),
                    "theoretical_ops": size * np.log2(size)
                },
                "speedup": {
                    "mean": float(np.mean(speedups)),
                    "std": float(np.std(speedups)),
                    "min": float(np.min(speedups)),
                    "max": float(np.max(speedups))
                },
                "energy_ratio": float(np.mean(categorical_energy) / np.mean(conventional_energy)),
                "ops_ratio": float(np.mean(categorical_ops) / np.mean(conventional_ops)),
                "correctness": "PASS"
            }

            results["benchmarks"].append(benchmark_result)

            print(f"    Speedup: {benchmark_result['speedup']['mean']:.2f}x "
                  f"(±{benchmark_result['speedup']['std']:.2f})")
            print(f"    Energy ratio: {benchmark_result['energy_ratio']:.2e}")
            print(f"    Ops ratio: {benchmark_result['ops_ratio']:.4f}")

    # Complexity analysis: Fit scaling curves
    print("\n" + "=" * 80)
    print("COMPLEXITY ANALYSIS")
    print("=" * 80)

    sizes_array = np.array(sizes)

    # Extract mean ops for random distribution
    random_benchmarks = [b for b in results["benchmarks"] if b["distribution"] == "random"]
    cat_ops = np.array([b["categorical"]["mean_ops"] for b in random_benchmarks])
    conv_ops = np.array([b["conventional"]["mean_ops"] for b in random_benchmarks])

    # Fit O(log_3 N) for categorical
    log3_n = np.log(sizes_array) / np.log(3)
    cat_fit = np.polyfit(log3_n, cat_ops, 1)  # Linear fit: ops = a*log3(N) + b

    # Fit O(N log N) for conventional
    n_log_n = sizes_array * np.log2(sizes_array)
    conv_fit = np.polyfit(n_log_n, conv_ops, 1)  # Linear fit: ops = a*N*log(N) + b

    results["complexity_analysis"] = {
        "categorical": {
            "complexity": "O(log_3 N)",
            "fit_coefficients": cat_fit.tolist(),
            "fit_quality_r2": float(np.corrcoef(log3_n, cat_ops)[0, 1]**2),
            "measured_ops": cat_ops.tolist(),
            "theoretical_ops": log3_n.tolist()
        },
        "conventional": {
            "complexity": "O(N log N)",
            "fit_coefficients": conv_fit.tolist(),
            "fit_quality_r2": float(np.corrcoef(n_log_n, conv_ops)[0, 1]**2),
            "measured_ops": conv_ops.tolist(),
            "theoretical_ops": n_log_n.tolist()
        }
    }

    print(f"Categorical: ops ~= {cat_fit[0]:.2f} x log_3(N) + {cat_fit[1]:.2f}")
    print(f"  R^2 = {results['complexity_analysis']['categorical']['fit_quality_r2']:.6f}")
    print(f"Conventional: ops ~= {conv_fit[0]:.6f} x N log_2(N) + {conv_fit[1]:.2f}")
    print(f"  R^2 = {results['complexity_analysis']['conventional']['fit_quality_r2']:.6f}")

    # Summary statistics
    all_speedups = [b["speedup"]["mean"] for b in results["benchmarks"]]
    all_energy_ratios = [b["energy_ratio"] for b in results["benchmarks"]]

    # Find best case (largest size)
    largest_size_benchmarks = [b for b in results["benchmarks"] if b["size"] == max(sizes)]
    best_speedup = max(b["speedup"]["mean"] for b in largest_size_benchmarks)
    best_energy_ratio = min(b["energy_ratio"] for b in largest_size_benchmarks)

    results["summary"] = {
        "overall_mean_speedup": float(np.mean(all_speedups)),
        "overall_std_speedup": float(np.std(all_speedups)),
        "best_speedup": float(best_speedup),
        "best_speedup_size": max(sizes),
        "overall_mean_energy_ratio": float(np.mean(all_energy_ratios)),
        "best_energy_ratio": float(best_energy_ratio),
        "speedup_scaling": "INCREASING" if all_speedups[-1] > all_speedups[0] else "CONSTANT",
        "all_tests_passed": all(b["correctness"] == "PASS" for b in results["benchmarks"]),
        "claim_validation": {
            "categorical_is_log3_n": results["complexity_analysis"]["categorical"]["fit_quality_r2"] > 0.95,
            "conventional_is_n_log_n": results["complexity_analysis"]["conventional"]["fit_quality_r2"] > 0.95,
            "speedup_increases_with_n": all_speedups[-1] > all_speedups[0],
            "energy_dramatically_lower": best_energy_ratio < 0.01
        }
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Mean speedup: {results['summary']['overall_mean_speedup']:.2f}x")
    print(f"Best speedup (N={max(sizes):,}): {best_speedup:.2f}x")
    print(f"Best energy ratio: {best_energy_ratio:.2e}")
    print(f"Speedup trend: {results['summary']['speedup_scaling']}")
    print()
    print("Claim Validation:")
    for claim, validated in results["summary"]["claim_validation"].items():
        status = "[OK] PASS" if validated else "[FAIL] FAIL"
        print(f"  {claim}: {status}")

    return results


def extrapolate_to_large_n(results: Dict[str, Any], target_sizes: List[int]) -> Dict[str, Any]:
    """
    Extrapolate performance to larger N based on complexity fits.

    This estimates what speedup we'd get at N=10^6, 10^7, etc.
    """
    print("\n" + "=" * 80)
    print("EXTRAPOLATION TO LARGE N")
    print("=" * 80)

    cat_fit = np.array(results["complexity_analysis"]["categorical"]["fit_coefficients"])
    conv_fit = np.array(results["complexity_analysis"]["conventional"]["fit_coefficients"])

    extrapolations = []

    for n in target_sizes:
        # Categorical ops: a*log3(N) + b
        log3_n = np.log(n) / np.log(3)
        cat_ops = cat_fit[0] * log3_n + cat_fit[1]

        # Conventional ops: a*N*log2(N) + b
        conv_ops = conv_fit[0] * n * np.log2(n) + conv_fit[1]

        # Speedup (assuming time proportional to ops)
        speedup = conv_ops / cat_ops

        extrapolations.append({
            "n": n,
            "categorical_ops": float(cat_ops),
            "conventional_ops": float(conv_ops),
            "predicted_speedup": float(speedup)
        })

        print(f"N = {n:,}:")
        print(f"  Categorical ops: {cat_ops:,.0f}")
        print(f"  Conventional ops: {conv_ops:,.0f}")
        print(f"  Predicted speedup: {speedup:,.0f}x")

    return {"extrapolations": extrapolations}


if __name__ == "__main__":
    # Run comprehensive validation
    results = validate_sorting_complexity(
        sizes=[100, 500, 1000, 5000, 10000, 50000, 100000],
        n_trials=5,
        distributions=["random", "reversed", "gaussian"]
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = save_results(
        results,
        f"sorting_validation_{timestamp}.json",
        format="json"
    )
    print(f"\n[OK] Results saved to: {filepath}")

    # Extrapolate to large N
    extrapolation = extrapolate_to_large_n(
        results,
        target_sizes=[500000, 1000000, 10000000, 100000000]
    )

    # Save extrapolation
    extrap_filepath = save_results(
        extrapolation,
        f"sorting_extrapolation_{timestamp}.json",
        format="json"
    )
    print(f"[OK] Extrapolation saved to: {extrap_filepath}")
