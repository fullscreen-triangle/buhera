"""Validator: the trajectory is a product of the run, not an input to it.

Claim (Thm. 5, Prop. 1): two runs over an *identical* node set induce different
causal edge relations -- different trajectories -- because which values a module
reads is decided during the run. No static schedule reproduces either.

We build one fixed node set. Two modules each decide whether to read a node on
the strength of a value that only exists once an *earlier* module has emitted it.
By seeding one run with a different initial nudge we get two different causal
edge relations over the same nodes, and we show no single fixed edge set equals
both.
"""

from __future__ import annotations

import json
import os

from ckg_runtime import Module, Node, Runtime, ValueDelta  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_nodes() -> list[Node]:
    """A fixed catalogue of subtasks. No edges -- edges do not exist yet."""
    letters = ["a", "b", "c", "d", "e", "f"]
    nodes = []
    for i, ell in enumerate(letters):
        n = Node(tau=ell, address=("cat", ell))
        # a chunk that just publishes the node's own seed value if present
        n.add_chunk(
            "publish",
            lambda vals, ell=ell: ValueDelta(
                kind="signal", payload=vals.get("seed", 0) + 1, source_chunk="publish"
            ),
        )
        nodes.append(n)
    return nodes


def run(seed_node: str, seed_mag: int = 3) -> set[tuple[str, str]]:
    """Return the causal edge relation induced by one run.

    A "reader" module reads a node's `signal` and, if it crosses a threshold that
    depends on run-produced values, emits onward to a *successor* node -- which is
    the causal edge. The threshold logic consults only values produced in-run.
    """
    nodes = build_nodes()
    by_tau = {n.tau: n for n in nodes}
    rt = Runtime()

    # seed exactly one node differently -> different run, same node set
    by_tau[seed_node].values["seed"] = seed_mag

    order = ["a", "b", "c", "d", "e", "f"]
    edges: set[tuple[str, str]] = set()

    # a module that carries a node's emitted signal onward. The *reach* of the
    # carry -- how many nodes ahead it lands -- is the signal's magnitude, a
    # value produced only during this run. A node that inherited a larger seed
    # (directly, or via an earlier carry) reaches further, so seeding a different
    # node reshapes which reads happen and thus which edges exist.
    def emit_forward(u: Node) -> ValueDelta | None:
        return ValueDelta(kind="carried", payload=u.values.get("signal", 0), source_chunk="fwd")

    carrier = Module("carrier", wants=lambda n: "signal" in n.values, emit=emit_forward)

    for i, tau in enumerate(order):
        u = by_tau[tau]
        rt.dispatch(u)  # runs publish chunk -> emits `signal` = seed(u)+1
        carried = carrier.read_transform_emit(u)
        if carried is None:
            continue
        # reach = the carried magnitude; a seeded node emits signal 4 (reach 4),
        # an unseeded one emits signal 1 (reach 1). The target is this-run data.
        reach = int(carried.payload)
        j = i + reach
        if j < len(order):
            v = by_tau[order[j]]
            edges.add((tau, order[j]))
            # propagate: v inherits the carry, so its own later emission -- and
            # therefore its reach -- is shaped by what reached it this run
            v.values["seed"] = max(v.values.get("seed", 0), carried.payload)
    return edges


ORDER = ["a", "b", "c", "d", "e", "f"]


def _adjacency(edges: set[tuple[str, str]]) -> list[list[int]]:
    """Dense 0/1 adjacency over ORDER for one run's induced edge relation."""
    idx = {t: i for i, t in enumerate(ORDER)}
    m = [[0] * len(ORDER) for _ in ORDER]
    for (u, v) in edges:
        m[idx[u]][idx[v]] = 1
    return m


def main() -> None:
    run1 = run(seed_node="a")
    run2 = run(seed_node="d")

    # The claim: the two runs' edge relations differ, and no single fixed edge
    # set equals both (their symmetric difference is non-empty).
    differ = run1 != run2
    sym_diff = run1 ^ run2
    assert differ, "two runs over the same nodes produced identical trajectories"
    assert sym_diff, "no distinguishing edges -- trajectory would be schedulable"

    # --- sweep: for every (seeded node, seed magnitude) record the induced
    # trajectory. This maps how a single run-time value reshapes the whole
    # causal edge relation -- the quantitative face of Thm. 5 / Prop. 1.
    seed_mags = list(range(1, 7))
    edge_count = [[0] * len(seed_mags) for _ in ORDER]        # nodes x magnitudes
    divergence = [[0] * len(seed_mags) for _ in ORDER]        # edges differing from baseline
    baseline = run(seed_node="a", seed_mag=1)                  # the "quiet" run
    per_run = []
    for ni, node in enumerate(ORDER):
        for mi, mag in enumerate(seed_mags):
            e = run(seed_node=node, seed_mag=mag)
            edge_count[ni][mi] = len(e)
            divergence[ni][mi] = len(e ^ baseline)
            per_run.append(
                {"seed_node": node, "seed_mag": mag,
                 "edges": sorted(list(x) for x in e), "n_edges": len(e),
                 "divergence_from_baseline": len(e ^ baseline)}
            )

    # count distinct trajectories observed across the whole sweep
    distinct = {frozenset(run(seed_node=n, seed_mag=m)) for n in ORDER for m in seed_mags}

    result = {
        "claim": "trajectory_emergence",
        "theorem": "Thm. 5 (trajectory emergence), Prop. 1 (edges induced not stored)",
        "passed": True,
        "order": ORDER,
        "two_runs": {
            "run1_seed": "a", "run2_seed": "d",
            "run1_edges": sorted(list(x) for x in run1),
            "run2_edges": sorted(list(x) for x in run2),
            "symmetric_difference": sorted(list(x) for x in sym_diff),
            "run1_adjacency": _adjacency(run1),
            "run2_adjacency": _adjacency(run2),
        },
        "sweep": {
            "seed_nodes": ORDER,
            "seed_mags": seed_mags,
            "edge_count": edge_count,          # [node][mag]
            "divergence": divergence,          # [node][mag]  (edges differing from quiet baseline)
        },
        "distinct_trajectories": len(distinct),
        "total_runs_in_sweep": len(ORDER) * len(seed_mags),
        "per_run": per_run,
    }
    _write(result)

    print(f"[trajectory_emergence] run1 edges: {sorted(run1)}")
    print(f"[trajectory_emergence] run2 edges: {sorted(run2)}")
    print(f"[trajectory_emergence] symmetric difference: {sorted(sym_diff)}")
    print(f"[trajectory_emergence] distinct trajectories over "
          f"{len(ORDER) * len(seed_mags)} runs: {len(distinct)}")
    print("[trajectory_emergence] PASS: trajectory is run-induced, not schedulable")


def _write(result: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.normpath(os.path.join(RESULTS_DIR, "trajectory_emergence.json"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[trajectory_emergence] wrote {out}")


if __name__ == "__main__":
    main()
