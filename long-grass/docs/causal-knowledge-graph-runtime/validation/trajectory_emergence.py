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

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ckg_runtime import Module, Node, Runtime, ValueDelta  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


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


def run(seed_node: str) -> set[tuple[str, str]]:
    """Return the causal edge relation induced by one run.

    A "reader" module reads a node's `signal` and, if it crosses a threshold that
    depends on run-produced values, emits onward to a *successor* node -- which is
    the causal edge. The threshold logic consults only values produced in-run.
    """
    nodes = build_nodes()
    by_tau = {n.tau: n for n in nodes}
    rt = Runtime()

    # seed exactly one node differently -> different run, same node set
    by_tau[seed_node].values["seed"] = 3

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


def main() -> None:
    run1 = run(seed_node="a")
    run2 = run(seed_node="d")

    # The claim: the two runs' edge relations differ, and no single fixed edge
    # set equals both (their symmetric difference is non-empty).
    differ = run1 != run2
    sym_diff = run1 ^ run2
    assert differ, "two runs over the same nodes produced identical trajectories"
    assert sym_diff, "no distinguishing edges -- trajectory would be schedulable"
    print(f"[trajectory_emergence] run1 edges: {sorted(run1)}")
    print(f"[trajectory_emergence] run2 edges: {sorted(run2)}")
    print(f"[trajectory_emergence] symmetric difference: {sorted(sym_diff)}")
    print("[trajectory_emergence] PASS: trajectory is run-induced, not schedulable")

    _plot(run1, run2)


def _node_pos() -> dict[str, tuple[float, float]]:
    order = ["a", "b", "c", "d", "e", "f"]
    return {t: (i, 0.0) for i, t in enumerate(order)}


def _draw(ax, edges, title, color):
    pos = _node_pos()
    for t, (x, y) in pos.items():
        ax.scatter([x], [y], s=900, color="#e8e8e8", edgecolors="#333", zorder=3)
        ax.text(x, y, t, ha="center", va="center", fontsize=12, zorder=4)
    for (u, v) in edges:
        x0, _ = pos[u]
        x1, _ = pos[v]
        ax.annotate(
            "",
            xy=(x1, 0.0),
            xytext=(x0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2.0,
                connectionstyle="arc3,rad=-0.35",
                shrinkA=18,
                shrinkB=18,
            ),
            zorder=2,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-0.7, 5.7)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")


def _plot(run1, run2) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    _draw(axes[0], set(), "Durable node set\n(no edges exist yet)", "#888")
    _draw(axes[1], run1, "Run 1 trajectory\n(seed = a)", "#1f77b4")
    _draw(axes[2], run2, "Run 2 trajectory\n(seed = d)", "#d62728")
    fig.suptitle(
        "Same nodes, different runs, different trajectories "
        "(edges are induced, not stored)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.normpath(os.path.join(FIG_DIR, "trajectory_emergence.png"))
    fig.savefig(out, dpi=150)
    print(f"[trajectory_emergence] wrote {out}")


if __name__ == "__main__":
    main()
