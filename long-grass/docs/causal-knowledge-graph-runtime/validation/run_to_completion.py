"""Validator: no exit code, run to completion (Thm. 2, Cor. 4).

Claim: a chunk that raises does not halt the run. The error becomes a recorded
value on the node, the runtime emits no verdict, and every subsequent node still
executes. We assemble a report from emitted values -- the report is the output,
not an exit code, and it *accounts for* the anomaly rather than aborting on it.

Also validates multi-chunk execution (Def. 3): a node bearing two chunks from
different modules runs BOTH; both value deltas appear.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ckg_runtime import Node, Runtime, ValueDelta  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def build_nodes() -> list[Node]:
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


def main() -> None:
    nodes = build_nodes()
    rt = Runtime()

    report: dict[str, dict] = {}
    for n in nodes:
        rt.dispatch(n)
        report[n.tau] = dict(n.values)

    # --- assertions -------------------------------------------------------
    # multi-chunk: both chunks on `derive` ran and both values are present
    assert report["derive"].get("derived_a") == "shape"
    assert report["derive"].get("derived_b") == "graff"

    # the anomaly is a recorded value, not a halt
    assert "error" in report["anomaly"], "anomaly was not recorded as a value"
    assert "unexpected reading" in report["anomaly"]["error"]

    # the run CONTINUED past the anomaly
    assert report["after"].get("tail") == "reached", "run halted on the anomaly"

    # the runtime exposes no verdict / exit code
    assert not hasattr(rt, "exit_code")
    assert not hasattr(rt, "ok")

    # exactly one audit entry raised; all acts are recorded
    raised = [a for a in rt.audit if a.raised]
    assert len(raised) == 1 and raised[0].tau == "anomaly"

    print("[run_to_completion] report:")
    for tau, vals in report.items():
        print(f"    {tau}: {vals}")
    print("[run_to_completion] PASS: ran to completion, anomaly recorded, no exit code")

    _plot(report, rt)


def _plot(report, rt) -> None:
    fig, (ax_flow, ax_audit) = plt.subplots(1, 2, figsize=(13.0, 3.8))

    order = ["measure", "derive", "anomaly", "after"]
    colors = {
        "measure": "#2ca02c",
        "derive": "#1f77b4",
        "anomaly": "#d62728",
        "after": "#2ca02c",
    }
    for i, tau in enumerate(order):
        ax_flow.scatter([i], [0], s=1500, color=colors[tau], alpha=0.85, zorder=3)
        ax_flow.text(i, 0, tau, ha="center", va="center", color="white", fontsize=10, zorder=4)
        if i < len(order) - 1:
            ax_flow.annotate(
                "",
                xy=(i + 1, 0),
                xytext=(i, 0),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=2, shrinkA=26, shrinkB=26),
            )
    ax_flow.text(2, -0.6, "raises -> recorded,\nnot halted", ha="center", color="#d62728", fontsize=9)
    ax_flow.text(3, 0.6, "run continued", ha="center", color="#2ca02c", fontsize=9)
    ax_flow.set_title("Run to completion: the anomaly does not abort the run", fontsize=12)
    ax_flow.set_xlim(-0.6, 3.6)
    ax_flow.set_ylim(-1.1, 1.1)
    ax_flow.axis("off")

    # audit table: act id, tau, chunk, emitted kind, raised?
    rows = [[a.act_id, a.tau, a.chunk, a.emitted_kind, "yes" if a.raised else ""] for a in rt.audit]
    tbl = ax_audit.table(
        cellText=rows,
        colLabels=["act", "node", "chunk", "emitted", "raised"],
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.35)
    for (r, _c), cell in tbl.get_celld().items():
        if r > 0 and rows[r - 1][4] == "yes":
            cell.set_facecolor("#fde0e0")
    ax_audit.set_title("Audit log: every act recorded, none adjudicated", fontsize=12)
    ax_audit.axis("off")

    fig.tight_layout()
    out = os.path.normpath(os.path.join(FIG_DIR, "run_to_completion.png"))
    fig.savefig(out, dpi=150)
    print(f"[run_to_completion] wrote {out}")


if __name__ == "__main__":
    main()
