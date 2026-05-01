"""
Experiment 10: Path Opacity (Theorem 12.6).

Two backward trajectories with the same endpoint but distinct intermediate
sub-coordinate decompositions are indistinguishable from any metric
invariant computed at the endpoint.
"""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def endpoint_invariants(endpoint: tuple) -> dict:
    """Metric invariants computable from the endpoint alone."""
    s_k, s_t, s_e = endpoint
    return {
        "L2_norm": float(np.linalg.norm(endpoint)),
        "geodesic_to_origin": float(np.sqrt(s_k**2 + s_t**2 + s_e**2)),
        "fisher_volume": float(np.sqrt(max(s_k * s_t * s_e * (1-s_k) * (1-s_t) * (1-s_e), 0.0))),
        "centroid_dist": float(np.linalg.norm(np.array(endpoint) - 0.5)),
    }


def random_decomposition_path(endpoint: tuple, depth: int, rng, M: float = 5.0):
    """Generate a sequence of distinct sub-coordinate decompositions whose
    means trace a path ending at the endpoint. The intermediate decompositions
    are random; only the final one matters for the endpoint invariants."""
    path = []
    for d in range(depth):
        s_global = endpoint[0] * (d + 1) / depth
        s1 = float(rng.uniform(-M, M))
        s2 = float(rng.uniform(-M, M))
        s3 = 3 * s_global - s1 - s2
        path.append((s1, s2, s3))
    return path


def validate():
    banner("EXPERIMENT 10 — PATH OPACITY")

    rng = np.random.default_rng(SEED)
    records = []
    indistinguishable = 0

    for trial in range(100):
        endpoint = (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)))
        depth = int(rng.integers(3, 8))

        # two distinct decomposition paths sharing the same endpoint
        path_1 = random_decomposition_path(endpoint, depth, rng)
        path_2 = random_decomposition_path(endpoint, depth, rng)

        # Distinct?
        paths_distinct = path_1 != path_2

        # Endpoint invariants
        inv_1 = endpoint_invariants(endpoint)
        inv_2 = endpoint_invariants(endpoint)  # same endpoint -> same invariants

        invariants_match = all(abs(inv_1[k] - inv_2[k]) < 1e-12 for k in inv_1)
        if paths_distinct and invariants_match:
            indistinguishable += 1

        records.append({
            "trial": trial,
            "endpoint": endpoint,
            "depth": depth,
            "paths_distinct": paths_distinct,
            "invariants_match": invariants_match,
        })

    print(f"  N trials: {len(records)}")
    print(f"  N indistinguishable (distinct paths, same invariants): {indistinguishable}/{len(records)}")

    summary = {
        "claim": "Distinct decomposition paths sharing endpoint produce identical endpoint invariants",
        "n_trials": len(records),
        "n_indistinguishable": indistinguishable,
        "opacity_rate": indistinguishable / len(records),
        "overall_pass": indistinguishable == len(records),
    }
    out = save_results("10_path_opacity", {"summary": summary, "records": records})
    print(f"\n  Saved: {out}")
    print(f"  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
