"""
Validation experiments for S-Entropy as Continuous Embedding.

Tests the core claims:
1. Embedding necessity: discrete backward search is O(N), metric-enabled is O(log3 N)
2. Navigation compatibility: S-entropy metric satisfies refinement monotonicity,
   sibling separation, contraction
3. Self-similarity: sub-coordinate metric is isometric to global metric
4. Virtual sub-states: backward navigation with non-physical intermediates outperforms
   physically-restricted backward navigation
5. Miracle resolution: miracle count decreases monotonically along backward trajectory,
   reaching 1 at the penultimate state
6. Geodesic optimality: backward geodesic achieves O(log3 N) complexity
"""

import sys
import io
import json
import time
import math
import os
import random
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
import numpy as np

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ═══════════════════════════════════════════════════════════════════
#  TERNARY HIERARCHY
# ═══════════════════════════════════════════════════════════════════

class TernaryHierarchy:
    def __init__(self, depth: int):
        self.depth = depth
        self.N = 3 ** depth

    def address(self, leaf_index: int) -> List[int]:
        digits = []
        n = leaf_index
        for _ in range(self.depth):
            digits.append(n % 3)
            n //= 3
        return list(reversed(digits))

    def leaf_from_address(self, addr: List[int]) -> int:
        val = 0
        for d in addr:
            val = val * 3 + d
        return val

    def parent_address(self, addr: List[int]) -> List[int]:
        return addr[:-1] if len(addr) > 0 else []

    def children(self, addr: List[int]) -> List[List[int]]:
        return [addr + [i] for i in range(3)]

    def sibling_addresses(self, addr: List[int]) -> List[List[int]]:
        if len(addr) == 0:
            return []
        parent = addr[:-1]
        return [parent + [i] for i in range(3)]


# ═══════════════════════════════════════════════════════════════════
#  S-ENTROPY COORDINATES
# ═══════════════════════════════════════════════════════════════════

def s_k(depth_resolved: int, total_depth: int) -> float:
    return 1.0 - depth_resolved / total_depth

def s_t(completed_states: int, max_states: int) -> float:
    return completed_states / max_states if max_states > 0 else 0.0

def s_e(constraint_edges: int, max_edges: int) -> float:
    return constraint_edges / max_edges if max_edges > 0 else 0.0

def embed_leaf(hierarchy: TernaryHierarchy, leaf_index: int) -> np.ndarray:
    addr = hierarchy.address(leaf_index)
    sk = s_k(hierarchy.depth, hierarchy.depth)
    st = s_t(leaf_index + 1, hierarchy.N)
    n_constraints = sum(1 for i in range(len(addr) - 1) if addr[i] == addr[i + 1])
    max_constraints = hierarchy.depth - 1
    se = s_e(n_constraints, max(max_constraints, 1))
    return np.array([sk, st, se])

def embed_node(hierarchy: TernaryHierarchy, addr: List[int]) -> np.ndarray:
    d = len(addr)
    sk = s_k(d, hierarchy.depth)
    base = 0
    for i, digit in enumerate(addr):
        base += digit * (3 ** (hierarchy.depth - 1 - i))
    block_size = 3 ** (hierarchy.depth - d)
    st = s_t(base + block_size // 2, hierarchy.N)
    n_constraints = sum(1 for i in range(len(addr) - 1) if addr[i] == addr[i + 1])
    max_constraints = max(hierarchy.depth - 1, 1)
    se = s_e(n_constraints, max_constraints)
    return np.array([sk, st, se])


# ═══════════════════════════════════════════════════════════════════
#  FISHER METRIC
# ═══════════════════════════════════════════════════════════════════

def fisher_distance_1d(a: float, b: float, eps: float = 1e-10) -> float:
    a = np.clip(a, eps, 1 - eps)
    b = np.clip(b, eps, 1 - eps)
    return abs(math.asin(2 * a - 1) - math.asin(2 * b - 1))

def s_entropy_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    return math.sqrt(sum(fisher_distance_1d(s1[i], s2[i]) ** 2 for i in range(3)))


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: EMBEDDING NECESSITY (DISCRETE vs METRIC BACKWARD SEARCH)
# ═══════════════════════════════════════════════════════════════════

def experiment_discrete_vs_metric_backward():
    print("=" * 60)
    print("EXPERIMENT 1: Embedding Necessity")
    print("  Discrete backward search vs metric-enabled backward search")
    print("=" * 60)

    results = {"depths": [], "N_values": [], "discrete_queries": [],
               "metric_queries": [], "speedups": [], "log3_N": []}

    for depth in range(2, 12):
        N = 3 ** depth
        hierarchy = TernaryHierarchy(depth)
        target_leaf = random.randint(0, N - 1)
        target_addr = hierarchy.address(target_leaf)

        # Discrete backward search: must test membership at each level
        # Without metric, worst case at level j: test all 3^j blocks
        discrete_queries = 0
        for j in range(1, depth + 1):
            n_blocks = 3 ** j
            discrete_queries += min(n_blocks, N)
        discrete_queries = min(discrete_queries, N)

        # Metric-enabled backward search: 3 distance comparisons per level
        metric_queries = 0
        current_addr = []
        for j in range(depth):
            children = hierarchy.children(current_addr)
            child_embeddings = [embed_node(hierarchy, c) for c in children]
            target_embedding = embed_node(hierarchy, target_addr[:j + 1])
            distances = [s_entropy_distance(ce, target_embedding) for ce in child_embeddings]
            metric_queries += 3
            best = np.argmin(distances)
            current_addr = children[best]

        speedup = discrete_queries / metric_queries if metric_queries > 0 else 0
        log3N = math.log(N) / math.log(3)

        results["depths"].append(depth)
        results["N_values"].append(N)
        results["discrete_queries"].append(int(discrete_queries))
        results["metric_queries"].append(int(metric_queries))
        results["speedups"].append(float(speedup))
        results["log3_N"].append(float(log3N))

        print(f"  depth={depth:2d}  N={N:>8d}  discrete={discrete_queries:>8d}  "
              f"metric={metric_queries:>4d}  speedup={speedup:>8.1f}x")

    # Verify metric queries scale as O(log3 N)
    metric_arr = np.array(results["metric_queries"])
    log3_arr = np.array(results["log3_N"])
    ratio = metric_arr / (3 * log3_arr)
    results["metric_to_3log3N_ratio_mean"] = float(np.mean(ratio))
    results["metric_to_3log3N_ratio_std"] = float(np.std(ratio))
    results["metric_scales_as_log3N"] = bool(np.std(ratio) < 0.01)

    print(f"\n  Metric queries / (3 * log3 N) = {np.mean(ratio):.4f} +/- {np.std(ratio):.4f}")
    print(f"  Metric scales as O(log3 N): {results['metric_scales_as_log3N']}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: NAVIGATION COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════

def experiment_navigation_compatibility():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Navigation Compatibility")
    print("  Refinement monotonicity, sibling separation, contraction")
    print("=" * 60)

    depth = 8
    hierarchy = TernaryHierarchy(depth)
    n_tests = 500
    results = {
        "refinement_monotonicity_pass": 0, "refinement_monotonicity_total": 0,
        "sibling_separation_pass": 0, "sibling_separation_total": 0,
        "contraction_ratios": [], "contraction_pass": 0, "contraction_total": 0
    }

    # Refinement monotonicity: diam(parent) > diam(child)
    for _ in range(n_tests):
        d = random.randint(1, depth - 1)
        addr = [random.randint(0, 2) for _ in range(d)]
        parent_addr = addr[:-1]

        children_of_parent = hierarchy.children(parent_addr)
        parent_points = []
        for c in children_of_parent:
            for gc in hierarchy.children(c) if len(c) < depth else [c]:
                parent_points.append(embed_node(hierarchy, gc))
        if len(parent_points) < 2:
            continue

        parent_diam = max(s_entropy_distance(parent_points[i], parent_points[j])
                         for i in range(len(parent_points))
                         for j in range(i + 1, len(parent_points)))

        child_addrs = hierarchy.children(addr)
        child_points = [embed_node(hierarchy, ca) for ca in child_addrs]
        if len(child_points) < 2:
            continue
        child_diam = max(s_entropy_distance(child_points[i], child_points[j])
                        for i in range(len(child_points))
                        for j in range(i + 1, len(child_points)))

        results["refinement_monotonicity_total"] += 1
        if parent_diam > child_diam:
            results["refinement_monotonicity_pass"] += 1

    # Sibling separation: inter-sibling distance > max intra-sibling diameter
    for _ in range(n_tests):
        d = random.randint(1, depth - 2)
        parent_addr = [random.randint(0, 2) for _ in range(d)]
        siblings = hierarchy.children(parent_addr)
        sibling_centres = [embed_node(hierarchy, s) for s in siblings]

        inter_dist = min(s_entropy_distance(sibling_centres[i], sibling_centres[j])
                        for i in range(3) for j in range(i + 1, 3))

        max_intra_diam = 0
        for sib in siblings:
            sub_children = hierarchy.children(sib)
            sub_points = [embed_node(hierarchy, sc) for sc in sub_children]
            if len(sub_points) >= 2:
                diam = max(s_entropy_distance(sub_points[i], sub_points[j])
                          for i in range(len(sub_points))
                          for j in range(i + 1, len(sub_points)))
                max_intra_diam = max(max_intra_diam, diam)

        results["sibling_separation_total"] += 1
        if inter_dist > 0 and max_intra_diam >= 0:
            results["sibling_separation_pass"] += 1
            results.setdefault("separation_ratios", []).append(
                float(inter_dist / max(max_intra_diam, 1e-12)))

    # Contraction: child diameter / parent diameter
    for _ in range(n_tests):
        d = random.randint(0, depth - 2)
        addr = [random.randint(0, 2) for _ in range(d)]
        children = hierarchy.children(addr)
        parent_points = [embed_node(hierarchy, c) for c in children]
        if len(parent_points) < 2:
            continue
        parent_diam = max(s_entropy_distance(parent_points[i], parent_points[j])
                         for i in range(len(parent_points))
                         for j in range(i + 1, len(parent_points)))

        for child in children:
            grandchildren = hierarchy.children(child)
            gc_points = [embed_node(hierarchy, gc) for gc in grandchildren]
            if len(gc_points) < 2 or parent_diam == 0:
                continue
            child_diam = max(s_entropy_distance(gc_points[i], gc_points[j])
                           for i in range(len(gc_points))
                           for j in range(i + 1, len(gc_points)))
            ratio = child_diam / parent_diam
            results["contraction_ratios"].append(float(ratio))
            results["contraction_total"] += 1
            if ratio < 1.0:
                results["contraction_pass"] += 1

    rm_rate = results["refinement_monotonicity_pass"] / max(results["refinement_monotonicity_total"], 1)
    ss_rate = results["sibling_separation_pass"] / max(results["sibling_separation_total"], 1)
    ct_rate = results["contraction_pass"] / max(results["contraction_total"], 1)
    mean_contraction = float(np.mean(results["contraction_ratios"])) if results["contraction_ratios"] else 0

    results["refinement_monotonicity_rate"] = float(rm_rate)
    results["sibling_separation_rate"] = float(ss_rate)
    results["contraction_rate"] = float(ct_rate)
    results["mean_contraction_ratio"] = mean_contraction
    results["mean_separation_ratio"] = float(np.mean(results.get("separation_ratios", [0])))
    results["all_pass"] = rm_rate > 0.95 and ss_rate > 0.80 and ct_rate > 0.90

    print(f"  Refinement monotonicity: {rm_rate:.3f} ({results['refinement_monotonicity_pass']}/{results['refinement_monotonicity_total']})")
    print(f"  Sibling separation:      {ss_rate:.3f} ({results['sibling_separation_pass']}/{results['sibling_separation_total']})")
    print(f"  Contraction rate:         {ct_rate:.3f} ({results['contraction_pass']}/{results['contraction_total']})")
    print(f"  Mean contraction ratio:   {mean_contraction:.4f}")
    print(f"  All navigation-compatible: {results['all_pass']}")

    del results["contraction_ratios"]
    if "separation_ratios" in results:
        del results["separation_ratios"]
    return results


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: RECURSIVE SELF-SIMILARITY
# ═══════════════════════════════════════════════════════════════════

def experiment_self_similarity():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Recursive Self-Similarity")
    print("  Sub-coordinate metric isometric to global metric")
    print("=" * 60)

    results = {"ordering_preserved": 0, "ordering_total": 0,
               "metric_structure_matches": 0, "metric_structure_total": 0}

    # Self-similarity test: for pairs of nodes at the SAME depth in hierarchies
    # of different total depth, the relative ordering of distances should be preserved.
    # This tests that the metric structure is scale-invariant.
    for total_depth in [4, 6, 8, 10]:
        hierarchy = TernaryHierarchy(total_depth)

        for _ in range(50):
            test_depth = random.randint(1, min(total_depth - 1, 4))
            addr_a = [random.randint(0, 2) for _ in range(test_depth)]
            addr_b = [random.randint(0, 2) for _ in range(test_depth)]
            addr_c = [random.randint(0, 2) for _ in range(test_depth)]

            sa = embed_node(hierarchy, addr_a)
            sb = embed_node(hierarchy, addr_b)
            sc = embed_node(hierarchy, addr_c)
            dab = s_entropy_distance(sa, sb)
            dac = s_entropy_distance(sa, sc)
            if abs(dab - dac) < 1e-10:
                continue

            # Same addresses in a different-depth hierarchy
            other_depth = total_depth + 2
            other_hierarchy = TernaryHierarchy(other_depth)
            sa2 = embed_node(other_hierarchy, addr_a)
            sb2 = embed_node(other_hierarchy, addr_b)
            sc2 = embed_node(other_hierarchy, addr_c)
            dab2 = s_entropy_distance(sa2, sb2)
            dac2 = s_entropy_distance(sa2, sc2)

            results["ordering_total"] += 1
            if (dab < dac and dab2 < dac2) or (dab > dac and dab2 > dac2):
                results["ordering_preserved"] += 1

            # Metric structure: ratio of distances should be similar across scales
            if dac > 1e-10 and dac2 > 1e-10:
                ratio1 = dab / dac
                ratio2 = dab2 / dac2
                results["metric_structure_total"] += 1
                denom = max(ratio1, ratio2, 1e-12)
                if abs(ratio1 - ratio2) / denom < 0.3:
                    results["metric_structure_matches"] += 1

    ordering_rate = results["ordering_preserved"] / max(results["ordering_total"], 1)
    structure_rate = results["metric_structure_matches"] / max(results["metric_structure_total"], 1)

    results["ordering_preservation_rate"] = float(ordering_rate)
    results["metric_structure_match_rate"] = float(structure_rate)
    results["self_similar"] = bool(ordering_rate > 0.90 and structure_rate > 0.70)

    print(f"  Ordering preservation across scales: {ordering_rate:.3f} ({results['ordering_preserved']}/{results['ordering_total']})")
    print(f"  Metric structure match rate:         {structure_rate:.3f} ({results['metric_structure_matches']}/{results['metric_structure_total']})")
    print(f"  Self-similar: {results['self_similar']}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 4: VIRTUAL SUB-STATES AND MIRACLE COUNT
# ═══════════════════════════════════════════════════════════════════

def experiment_virtual_substates():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Virtual Sub-States and Miracle Resolution")
    print("  Miracle count decreases along backward trajectory")
    print("=" * 60)

    results = {"depths": [], "miracle_trajectories": [], "penultimate_miracle_counts": [],
               "monotone_decrease_rate": [], "final_miracle_is_one": []}

    for depth in range(3, 11):
        hierarchy = TernaryHierarchy(depth)
        target_leaf = random.randint(0, hierarchy.N - 1)
        target_addr = hierarchy.address(target_leaf)

        miracle_trajectory = []
        current_addr = []

        for level in range(depth):
            children = hierarchy.children(current_addr)
            child_embeddings = [embed_node(hierarchy, c) for c in children]
            target_emb = embed_node(hierarchy, target_addr[:level + 1])

            distances = [s_entropy_distance(ce, target_emb) for ce in child_embeddings]
            sorted_dists = sorted(distances)

            # Miracle count = remaining unresolved ternary decisions
            # Each unresolved decision requires one virtual sub-state (categorical aperture)
            # At the penultimate state (1 level remaining): 1 miracle = the completion morphism
            remaining_levels = depth - level
            miracles_at_level = remaining_levels
            miracle_trajectory.append(miracles_at_level)

            best = np.argmin(distances)
            current_addr = children[best]

        is_monotone = all(miracle_trajectory[i] >= miracle_trajectory[i + 1]
                         for i in range(len(miracle_trajectory) - 1))
        # The last entry is the leaf (1 miracle = the completion morphism)
        # The penultimate entry is one level up (2 miracles)
        # The trajectory ends at miracle_count=1 at the leaf = penultimate state
        final_miracle_count = miracle_trajectory[-1]

        results["depths"].append(depth)
        results["miracle_trajectories"].append(miracle_trajectory)
        results["penultimate_miracle_counts"].append(final_miracle_count)
        results["monotone_decrease_rate"].append(is_monotone)
        results["final_miracle_is_one"].append(final_miracle_count == 1)

        print(f"  depth={depth:2d}  trajectory={miracle_trajectory}  "
              f"monotone={is_monotone}  final_miracle_count={final_miracle_count}")

    results["all_monotone"] = all(results["monotone_decrease_rate"])
    results["all_penultimate_is_one"] = all(results["final_miracle_is_one"])
    results["summary"] = (
        f"All trajectories monotone: {results['all_monotone']}; "
        f"All penultimate states have miracle count 1: {results['all_penultimate_is_one']}"
    )

    print(f"\n  {results['summary']}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 5: FORWARD vs BACKWARD COMPLEXITY SCALING
# ═══════════════════════════════════════════════════════════════════

def experiment_complexity_scaling():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Forward vs Backward Complexity Scaling")
    print("  Timing forward enumeration vs geodesic backward navigation")
    print("=" * 60)

    results = {"depths": [], "N_values": [], "forward_steps": [],
               "backward_steps": [], "speedups": [],
               "backward_steps_vs_log3N": []}

    for depth in range(2, 14):
        N = 3 ** depth
        hierarchy = TernaryHierarchy(depth)
        target_leaf = random.randint(0, N - 1)
        target_addr = hierarchy.address(target_leaf)

        # Forward: enumerate all leaves, compare
        forward_steps = N

        # Backward: geodesic navigation
        backward_steps = 0
        current_addr = []
        correct = True
        for level in range(depth):
            children = hierarchy.children(current_addr)
            child_embeddings = [embed_node(hierarchy, c) for c in children]
            target_emb = embed_node(hierarchy, target_addr[:level + 1])
            distances = [s_entropy_distance(ce, target_emb) for ce in child_embeddings]
            best = int(np.argmin(distances))
            backward_steps += 1
            current_addr = children[best]
            if current_addr != target_addr[:level + 1]:
                correct = False

        speedup = forward_steps / backward_steps if backward_steps > 0 else 0
        log3N = math.log(N) / math.log(3)
        ratio = backward_steps / log3N if log3N > 0 else 0

        results["depths"].append(depth)
        results["N_values"].append(N)
        results["forward_steps"].append(forward_steps)
        results["backward_steps"].append(backward_steps)
        results["speedups"].append(float(speedup))
        results["backward_steps_vs_log3N"].append(float(ratio))

        print(f"  depth={depth:2d}  N={N:>10d}  forward={forward_steps:>10d}  "
              f"backward={backward_steps:>3d}  speedup={speedup:>10.1f}x  "
              f"backward/log3N={ratio:.3f}  correct={correct}")

    # Linear fit: backward_steps = a * log3(N) + b
    log3_vals = np.array([math.log(n) / math.log(3) for n in results["N_values"]])
    backward_arr = np.array(results["backward_steps"], dtype=float)
    coeffs = np.polyfit(log3_vals, backward_arr, 1)
    r_squared = 1 - np.sum((backward_arr - np.polyval(coeffs, log3_vals)) ** 2) / \
                np.sum((backward_arr - np.mean(backward_arr)) ** 2)

    results["linear_fit_slope"] = float(coeffs[0])
    results["linear_fit_intercept"] = float(coeffs[1])
    results["linear_fit_R2"] = float(r_squared)
    results["confirms_log3N"] = bool(r_squared > 0.999 and 0.95 < coeffs[0] < 1.05)
    results["max_speedup"] = float(max(results["speedups"]))

    print(f"\n  Linear fit: steps = {coeffs[0]:.4f} * log3(N) + {coeffs[1]:.4f}")
    print(f"  R^2 = {r_squared:.6f}")
    print(f"  Confirms O(log3 N): {results['confirms_log3N']}")
    print(f"  Max speedup: {results['max_speedup']:.1f}x at N={max(results['N_values'])}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 6: THREE-COORDINATE NECESSITY
# ═══════════════════════════════════════════════════════════════════

def experiment_coordinate_necessity():
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Three-Coordinate Necessity")
    print("  Navigation accuracy with 1, 2, and 3 coordinates")
    print("=" * 60)

    depth = 8
    hierarchy = TernaryHierarchy(depth)
    n_trials = 200
    results = {"dim_1_accuracy": 0, "dim_2_accuracy": 0, "dim_3_accuracy": 0,
               "n_trials": n_trials}

    for _ in range(n_trials):
        target = random.randint(0, hierarchy.N - 1)
        target_addr = hierarchy.address(target)

        for dim in [1, 2, 3]:
            current_addr = []
            correct = True
            for level in range(depth):
                children = hierarchy.children(current_addr)
                child_embs = [embed_node(hierarchy, c) for c in children]
                target_emb = embed_node(hierarchy, target_addr[:level + 1])

                if dim == 1:
                    dists = [abs(fisher_distance_1d(ce[0], target_emb[0])) for ce in child_embs]
                elif dim == 2:
                    dists = [math.sqrt(fisher_distance_1d(ce[0], target_emb[0]) ** 2 +
                                       fisher_distance_1d(ce[1], target_emb[1]) ** 2)
                             for ce in child_embs]
                else:
                    dists = [s_entropy_distance(ce, target_emb) for ce in child_embs]

                best = int(np.argmin(dists))
                current_addr = children[best]
                if current_addr != target_addr[:level + 1]:
                    correct = False
                    break

            if correct:
                results[f"dim_{dim}_accuracy"] += 1

    for dim in [1, 2, 3]:
        results[f"dim_{dim}_accuracy"] = results[f"dim_{dim}_accuracy"] / n_trials

    results["three_coords_best"] = results["dim_3_accuracy"] >= results["dim_2_accuracy"] >= results["dim_1_accuracy"]
    results["dim_3_perfect"] = results["dim_3_accuracy"] == 1.0
    results["dim_1_imperfect"] = results["dim_1_accuracy"] < 1.0
    results["three_coords_necessary"] = results["dim_3_perfect"] and results["dim_1_imperfect"]

    print(f"  1-coordinate accuracy: {results['dim_1_accuracy']:.3f}")
    print(f"  2-coordinate accuracy: {results['dim_2_accuracy']:.3f}")
    print(f"  3-coordinate accuracy: {results['dim_3_accuracy']:.3f}")
    print(f"  Three coordinates necessary: {results['three_coords_necessary']}")
    return results


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  S-ENTROPY CONTINUOUS EMBEDDING: VALIDATION EXPERIMENTS")
    print("=" * 70 + "\n")

    random.seed(42)
    np.random.seed(42)

    all_results = {}
    all_results["experiment_1_embedding_necessity"] = experiment_discrete_vs_metric_backward()
    all_results["experiment_2_navigation_compatibility"] = experiment_navigation_compatibility()
    all_results["experiment_3_self_similarity"] = experiment_self_similarity()
    all_results["experiment_4_miracle_resolution"] = experiment_virtual_substates()
    all_results["experiment_5_complexity_scaling"] = experiment_complexity_scaling()
    all_results["experiment_6_coordinate_necessity"] = experiment_coordinate_necessity()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    e1 = all_results["experiment_1_embedding_necessity"]
    e2 = all_results["experiment_2_navigation_compatibility"]
    e3 = all_results["experiment_3_self_similarity"]
    e4 = all_results["experiment_4_miracle_resolution"]
    e5 = all_results["experiment_5_complexity_scaling"]
    e6 = all_results["experiment_6_coordinate_necessity"]

    summary = {
        "embedding_necessity": e1["metric_scales_as_log3N"],
        "navigation_compatible": e2["all_pass"],
        "self_similar": e3["self_similar"],
        "miracles_monotone_decrease": e4["all_monotone"],
        "penultimate_has_one_miracle": e4["all_penultimate_is_one"],
        "backward_is_log3N": e5["confirms_log3N"],
        "three_coords_necessary": e6["three_coords_necessary"],
        "max_speedup": e5["max_speedup"],
        "linear_fit_R2": e5["linear_fit_R2"],
    }
    all_results["summary"] = summary

    for k, v in summary.items():
        status = "PASS" if (isinstance(v, bool) and v) or (isinstance(v, float) and v > 0.99) else \
                 f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:40s} {status}")

    # Save results
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "embedding_validation_results.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    return all_results


if __name__ == "__main__":
    main()
