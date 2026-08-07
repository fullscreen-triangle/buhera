"""Validator: absolute control at any resolution (Def. 6, Prop. 7).

Claim: editing at depth k changes only objects whose address shares the
length-k prefix; everything else is byte-identical. A coarse import (short
prefix, touch nothing) and a surgical redesign (full-depth addresses, redesign
individual steps) are the SAME address space traversed to different depths.

We build one addressed node subtree, run it coarsely (no edits), then run it
surgically (edit three full-depth addresses), and assert that exactly the
objects under the edited prefixes changed and nothing else did.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from ckg_runtime import Node, ValueDelta  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def build_setup() -> list[Node]:
    """A standard setup: a depth-3 address tree under ('gel','lane_i','step_j')."""
    nodes = []
    for lane in range(3):
        for step in range(4):
            addr = ("gel", f"lane{lane}", f"step{step}")
            n = Node(tau="/".join(addr), address=addr)
            n.add_chunk(
                "std",
                lambda vals, a=addr: ValueDelta(kind="setting", payload=("standard", a),
                                                source_chunk="std"),
            )
            nodes.append(n)
    return nodes


def realize(nodes: list[Node]) -> dict[tuple[str, ...], object]:
    """Run every node's chunks and collect the resulting `setting` value."""
    from ckg_runtime import Runtime

    rt = Runtime()
    out = {}
    for n in nodes:
        rt.dispatch(n)
        out[n.address] = n.values.get("setting")
    return out


def edit_at(node: Node) -> None:
    """A surgical redesign: replace the chunk at one full-depth address."""
    node.chunks.clear()
    node.add_chunk(
        "surgical",
        lambda vals, a=node.address: ValueDelta(kind="setting", payload=("REDESIGNED", a),
                                                source_chunk="surgical"),
    )


def main() -> None:
    # --- coarse run: import, touch nothing --------------------------------
    coarse_nodes = build_setup()
    coarse = realize(coarse_nodes)

    # --- surgical run: same setup, edit three full-depth addresses --------
    surgical_nodes = build_setup()
    edit_addresses = [
        ("gel", "lane0", "step1"),
        ("gel", "lane1", "step3"),
        ("gel", "lane2", "step0"),
    ]
    by_addr = {n.address: n for n in surgical_nodes}
    for a in edit_addresses:
        edit_at(by_addr[a])
    surgical = realize(surgical_nodes)

    # --- assertions -------------------------------------------------------
    edited = set(edit_addresses)
    changed = {a for a in coarse if coarse[a] != surgical[a]}
    unchanged = {a for a in coarse if coarse[a] == surgical[a]}

    # exactly the edited addresses changed -- prefix containment (Prop. 7)
    assert changed == edited, f"changed set {changed} != edited set {edited}"
    # every other object is byte-identical between coarse and surgical
    assert unchanged == (set(coarse) - edited)
    # the redesigned ones carry the surgical marker; the rest the standard one
    for a in edited:
        assert surgical[a][0] == "REDESIGNED"
    for a in unchanged:
        assert surgical[a][0] == "standard"

    print(f"[resolution_control] total nodes: {len(coarse)}")
    print(f"[resolution_control] edited addresses: {sorted(edited)}")
    print(f"[resolution_control] changed exactly the edited set: {changed == edited}")
    print(f"[resolution_control] untouched objects: {len(unchanged)} (byte-identical)")
    print("[resolution_control] PASS: editing at depth k touches only the k-prefix subtree")

    _plot(coarse, surgical, edited)


def _plot(coarse, surgical, edited) -> None:
    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(13.0, 4.2), sharey=True)

    def draw(ax, realized, title, show_edits):
        for a in realized:
            lane = int(a[1].replace("lane", ""))
            step = int(a[2].replace("step", ""))
            is_edit = show_edits and a in edited
            color = "#d62728" if is_edit else "#bcd4e6"
            ax.add_patch(mpatches.FancyBboxPatch(
                (step - 0.4, lane - 0.4), 0.8, 0.8,
                boxstyle="round,pad=0.02", linewidth=1.2,
                edgecolor="#333", facecolor=color, zorder=2))
            label = "surgical" if is_edit else "std"
            ax.text(step, lane, label, ha="center", va="center", fontsize=8,
                    color="white" if is_edit else "#234", zorder=3)
        ax.set_xlim(-0.7, 3.7)
        ax.set_ylim(-0.7, 2.7)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"step{j}" for j in range(4)])
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"lane{i}" for i in range(3)])
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")

    draw(ax_c, coarse, "Coarse: import standard setup\n(short prefix, touch nothing)", False)
    draw(ax_s, surgical, "Surgical: redesign 3 full-depth addresses\n(everything else byte-identical)", True)

    fig.suptitle("One address space, two resolutions: editing at depth k touches only the k-prefix",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.normpath(os.path.join(FIG_DIR, "resolution_control.png"))
    fig.savefig(out, dpi=150)
    print(f"[resolution_control] wrote {out}")


if __name__ == "__main__":
    main()
