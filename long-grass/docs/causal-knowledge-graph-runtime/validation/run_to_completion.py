"""Validator: no exit code, run to completion (Thm. 2, Cor. 4).

Claim: a chunk that raises does not halt the run. The error becomes a recorded
value on the node, the runtime emits no verdict, and every subsequent node still
executes. We assemble a report from emitted values -- the report is the output,
not an exit code, and it *accounts for* the anomaly rather than aborting on it.

Also validates multi-chunk execution (Def. 3): a node bearing two chunks from
different modules runs BOTH; both value deltas appear.

Beyond the core scenario we sweep chains of increasing length with a growing
fraction of anomalous nodes, and record how many nodes still execute. A
conventional halt-on-error runtime would reach only up to the first anomaly; this
runtime reaches every node regardless. The gap between the two is the quantity
this validator measures.
"""

from __future__ import annotations

import json
import os

from ckg_runtime import Node, Runtime, ValueDelta  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_core_nodes() -> list[Node]:
    nodes: list[Node] = []

    # node 0: a well-behaved single-chunk node
    n0 = Node(tau="measure", address=("run", "measure"))
    n0.add_chunk(
        "quantify",
        lambda vals: ValueDelta(kind="quantity", payload=42, source_chunk="quantify"),
    )
    nodes.append(n0)

    # node 1: TWO chunks from two different modules on ONE subtask (Def. 3).
    # The runtime must run both; both deltas must appear.
    n1 = Node(tau="derive", address=("run", "derive"))
    n1.add_chunk(
        "shapeshifter_chunk",
        lambda vals: ValueDelta(kind="derived_a", payload="shape", source_chunk="shapeshifter_chunk"),
    )
    n1.add_chunk(
        "graffiti_chunk",
        lambda vals: ValueDelta(kind="derived_b", payload="graff", source_chunk="graffiti_chunk"),
    )
    nodes.append(n1)

    # node 2: a chunk that RAISES. Under a conventional runtime this aborts.
    # Here it must become a recorded value and the run must continue.
    def explode(vals: dict) -> ValueDelta:
        raise RuntimeError("unexpected reading")

    n2 = Node(tau="anomaly", address=("run", "anomaly"))
    n2.add_chunk("explode", explode)
    nodes.append(n2)

    # node 3: proves the run continued past the anomaly
    n3 = Node(tau="after", address=("run", "after"))
    n3.add_chunk(
        "record",
        lambda vals: ValueDelta(kind="tail", payload="reached", source_chunk="record"),
    )
    nodes.append(n3)

    return nodes


def run_core() -> tuple[dict, Runtime]:
    nodes = build_core_nodes()
    rt = Runtime()
    report: dict[str, dict] = {}
    for n in nodes:
        rt.dispatch(n)
        report[n.tau] = dict(n.values)
    return report, rt


def build_chain(length: int, anomaly_positions: set[int]) -> list[Node]:
    """A linear chain; nodes at `anomaly_positions` carry a raising chunk."""
    nodes = []
    for i in range(length):
        n = Node(tau=f"s{i}", address=("chain", f"s{i}"))
        if i in anomaly_positions:
            def explode(vals, i=i):
                raise RuntimeError(f"anomaly at {i}")
            n.add_chunk("explode", explode)
        else:
            n.add_chunk(
                "ok",
                lambda vals, i=i: ValueDelta(kind="ok", payload=i, source_chunk="ok"),
            )
        nodes.append(n)
    return nodes


def reached_counts(length: int, anomaly_positions: set[int]) -> tuple[int, int]:
    """Return (nodes reached by THIS runtime, nodes a halt-on-error runtime
    would reach). This runtime always reaches all `length` nodes; the
    conventional one stops just before the first anomaly."""
    nodes = build_chain(length, anomaly_positions)
    rt = Runtime()
    reached = 0
    for n in nodes:
        rt.dispatch(n)
        reached += 1
    first_anom = min(anomaly_positions) if anomaly_positions else length
    halt_on_error_reached = first_anom  # conventional runtime stops at first anomaly
    return reached, halt_on_error_reached


def main() -> None:
    report, rt = run_core()

    # --- core assertions --------------------------------------------------
    assert report["derive"].get("derived_a") == "shape"
    assert report["derive"].get("derived_b") == "graff"
    assert "error" in report["anomaly"], "anomaly was not recorded as a value"
    assert "unexpected reading" in report["anomaly"]["error"]
    assert report["after"].get("tail") == "reached", "run halted on the anomaly"
    assert not hasattr(rt, "exit_code")
    assert not hasattr(rt, "ok")
    raised = [a for a in rt.audit if a.raised]
    assert len(raised) == 1 and raised[0].tau == "anomaly"

    # --- sweep: chain length x anomaly density ----------------------------
    lengths = [4, 8, 12, 16, 20, 24, 28, 32]
    densities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    grid_runtime = []       # [density][length] nodes reached by this runtime
    grid_halt = []          # [density][length] nodes a halt-on-error runtime reaches
    for d in densities:
        row_rt, row_halt = [], []
        for L in lengths:
            # deterministic anomaly placement: every k-th node, k from density
            if d == 0.0:
                anoms: set[int] = set()
            else:
                step = max(1, round(1 / d))
                anoms = set(range(step, L, step))
            reached, halt = reached_counts(L, anoms)
            assert reached == L, "run did not reach every node"
            row_rt.append(reached)
            row_halt.append(halt)
        grid_runtime.append(row_rt)
        grid_halt.append(row_halt)

    # completion fraction this runtime achieves vs halt-on-error, per density
    completion_runtime = [1.0 for _ in densities]  # always 1.0
    completion_halt = [
        sum(grid_halt[di]) / sum(grid_runtime[di]) for di in range(len(densities))
    ]

    result = {
        "claim": "run_to_completion",
        "theorem": "Thm. 2 (no exit code), Cor. 4 (run to completion), Def. 3 (multi-chunk)",
        "passed": True,
        "core": {
            "report": report,
            "audit": [
                {"act_id": a.act_id, "tau": a.tau, "chunk": a.chunk,
                 "emitted_kind": a.emitted_kind, "raised": a.raised}
                for a in rt.audit
            ],
            "multi_chunk_node": {"tau": "derive",
                                 "chunks_run": ["shapeshifter_chunk", "graffiti_chunk"],
                                 "both_emitted": True},
            "runtime_exposes_verdict": False,
        },
        "sweep": {
            "lengths": lengths,
            "densities": densities,
            "reached_runtime": grid_runtime,   # [density][length]
            "reached_halt_on_error": grid_halt,
            "completion_fraction_runtime": completion_runtime,
            "completion_fraction_halt_on_error": completion_halt,
        },
    }
    _write(result)

    print("[run_to_completion] core report:")
    for tau, vals in report.items():
        print(f"    {tau}: {vals}")
    print(f"[run_to_completion] sweep: {len(lengths)}x{len(densities)} grid, "
          f"this runtime reached 100% in every cell")
    print(f"[run_to_completion] halt-on-error completion fraction by density: "
          f"{[round(x, 3) for x in completion_halt]}")
    print("[run_to_completion] PASS: ran to completion, anomaly recorded, no exit code")


def _write(result: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.normpath(os.path.join(RESULTS_DIR, "run_to_completion.json"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[run_to_completion] wrote {out}")


if __name__ == "__main__":
    main()
