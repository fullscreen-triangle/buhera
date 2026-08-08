"""Validator: absolute control at any resolution (Def. 6, Prop. 7).

Claim: editing at depth k changes only objects whose address shares the
length-k prefix; everything else is byte-identical. A coarse import (short
prefix) and a surgical redesign (full-depth addresses) are the SAME address
space at different depths.

Core check: edit three full-depth addresses in a standard setup and confirm
exactly those objects changed. Sweep: over a uniform address tree of branching
factor b and depth D, edit at each depth k and measure the blast radius -- the
number of leaf objects affected. Prefix containment predicts blast radius
b^(D-k): editing a short prefix (coarse) touches a whole region, editing a full
address (surgical) touches exactly one leaf.
"""

from __future__ import annotations

import json
import os

from ckg_runtime import Node, ValueDelta  # noqa: E402
from ckg_runtime import Runtime  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# --- core scenario (unchanged shape) ----------------------------------------

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
    rt = Runtime()
    out = {}
    for n in nodes:
        rt.dispatch(n)
        out[n.address] = n.values.get("setting")
    return out


def edit_at(node: Node) -> None:
    node.chunks.clear()
    node.add_chunk(
        "surgical",
        lambda vals, a=node.address: ValueDelta(kind="setting", payload=("REDESIGNED", a),
                                                source_chunk="surgical"),
    )


# --- sweep: blast radius vs edit depth on a uniform tree --------------------

def build_tree(branch: int, depth: int) -> list[Node]:
    """All leaves of a uniform tree, addressed root..leaf (length depth+1)."""
    leaves: list[Node] = []

    def rec(prefix: tuple[str, ...], d: int) -> None:
        if d == depth:
            leaves.append(Node(tau="/".join(prefix), address=prefix))
            return
        for i in range(branch):
            rec(prefix + (f"n{d}_{i}",), d + 1)

    rec(("root",), 0)
    for n in leaves:
        n.add_chunk("std", lambda vals, a=n.address:
                    ValueDelta(kind="setting", payload=("standard", a), source_chunk="std"))
    return leaves


def blast_radius(branch: int, depth: int, edit_depth: int) -> int:
    """Edit ONE address prefix of length edit_depth+1; count affected leaves."""
    leaves = build_tree(branch, depth)
    base = realize(leaves)

    # choose the first prefix at the given depth (n0_0/n1_0/...) and redesign
    # every leaf under it
    target_prefix = tuple(["root"] + [f"n{d}_0" for d in range(edit_depth)])
    edited = build_tree(branch, depth)
    for n in edited:
        if n.address[: len(target_prefix)] == target_prefix:
            edit_at(n)
    after = realize(edited)

    changed = sum(1 for a in base if base[a] != after[a])
    # prefix containment predicts branch ** (depth - edit_depth)
    predicted = branch ** (depth - edit_depth)
    assert changed == predicted, f"blast {changed} != predicted {predicted}"
    return changed


def main() -> None:
    # --- core -------------------------------------------------------------
    coarse = realize(build_setup())
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

    edited = set(edit_addresses)
    changed = {a for a in coarse if coarse[a] != surgical[a]}
    unchanged = {a for a in coarse if coarse[a] == surgical[a]}
    assert changed == edited
    assert unchanged == (set(coarse) - edited)

    # --- sweep: blast radius vs edit depth, for several branching factors --
    branches = [2, 3, 4]
    depth = 5
    blast = {}          # branch -> [blast at edit_depth 1..depth]
    predicted = {}
    for b in branches:
        row, prow = [], []
        for k in range(1, depth + 1):
            r = blast_radius(b, depth, k)
            row.append(r)
            prow.append(b ** (depth - k))
        blast[str(b)] = row
        predicted[str(b)] = prow

    result = {
        "claim": "resolution_control",
        "theorem": "Def. 6 (resolution), Prop. 7 (prefix containment)",
        "passed": True,
        "core": {
            "total_nodes": len(coarse),
            "edited_addresses": [list(a) for a in sorted(edited)],
            "changed_exactly_edited": changed == edited,
            "untouched_count": len(unchanged),
        },
        "sweep": {
            "depth": depth,
            "edit_depths": list(range(1, depth + 1)),
            "branches": branches,
            "blast_radius": blast,           # branch -> [leaves affected per edit depth]
            "predicted_b_pow": predicted,    # branch -> [b^(depth-k)]
        },
    }
    _write(result)

    print(f"[resolution_control] core: {len(coarse)} nodes, "
          f"changed exactly the {len(edited)} edited, {len(unchanged)} untouched")
    for b in branches:
        print(f"[resolution_control] branch={b} blast radius by edit depth: {blast[str(b)]}")
    print("[resolution_control] PASS: editing at depth k touches only the k-prefix subtree")


def _write(result: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.normpath(os.path.join(RESULTS_DIR, "resolution_control.json"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[resolution_control] wrote {out}")


if __name__ == "__main__":
    main()
