"""
vaHera pairing — an ndombolo-shaped AI companion over vaHera execution.

Not a new execution path. vaHera already produces two event streams while
it runs a program against a Kernel:

  - ExecContext.trace   (interpreter.py)     — one line per vaHera statement
  - Kernel.trace() / Kernel.tem.stats()      — one line per subsystem call,
                                                plus a running divergence
                                                sample (TEM) that flags an
                                                alert when delta exceeds the
                                                threshold.

This module reads those two streams after (or during) a run and narrates
them back in the same scientific-statement vocabulary the vaHera grammar
now accepts (see interpreter.py's parser), so a scientist watching a run
sees "observed X" / "ran P on X" / "inconsistency observed" rather than
kernel-internal op names and coordinate tuples. It adds no new kernel
surface: everything here reads data the kernel already computes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..kernel import Kernel
from .interpreter import ExecContext


# ─── op -> scientific-statement narration ─────────────────────────────

_OP_NARRATION = {
    "describe": lambda a: f'observed {a["target"]} as "{a["text"]}"',
    "resolve": lambda a: f'resolved {a["target"]}',
    "spawn": lambda a: f'ran {a["program"]} on {a["target"]}',
    "navigate": lambda a: "navigated toward completion",
    "complete": lambda a: "reached completion",
    "memory_create": lambda a: "allocated a new record",
    "memory_store": lambda a: f'recorded "{a["name"]}"',
    "memory_find": lambda a: f'compared to "{a["query"]}" (k={a["k"]})',
    "demon_sort": lambda a: "ranked records by category",
    "controller_verify": lambda a: "checked consistency",
}


@dataclass
class PairingReport:
    """Narrated account of one vaHera run, in scientific-statement language."""
    narration: list[str] = field(default_factory=list)
    inconsistencies: list[str] = field(default_factory=list)

    def closure_summary(self) -> str:
        """
        Closure-style report (ndombolo's "no further probe reveals a new
        answer-class" framing), read off TEM's alert count rather than a
        confidence threshold: either no inconsistency was found, or the
        specific divergences found are named.
        """
        if not self.inconsistencies:
            return "no inconsistency found — consistent across all checks"
        return f"{len(self.inconsistencies)} inconsistency(ies) observed:\n  " + \
            "\n  ".join(self.inconsistencies)

    def render(self) -> str:
        lines = ["--- pairing narration ---"]
        lines.extend(self.narration)
        lines.append("--- consistency ---")
        lines.append(self.closure_summary())
        return "\n".join(lines)


def narrate(ctx: ExecContext, kernel: Kernel, stmts: list) -> PairingReport:
    """
    Build a PairingReport from one execute_vahera run.

    `stmts` is the parsed statement list (as returned by parse_vahera) —
    passed separately from `ctx` because ExecContext.trace only records a
    subset of ops (describe/resolve/memory_*); this narrates every
    statement that was actually dispatched, in the order it ran.
    """
    report = PairingReport()

    for stmt in stmts:
        fn = _OP_NARRATION.get(stmt.op)
        line = fn(stmt.args) if fn else f"executed {stmt.op}"
        report.narration.append(line)

    for alert in kernel.tem.events():
        report.inconsistencies.append(alert)

    return report


def run_and_narrate(program: str, kernel: Kernel | None = None,
                    molecule_data: dict | None = None
                    ) -> tuple[ExecContext, PairingReport]:
    """Execute a vaHera program and return both the result and its narration."""
    from .interpreter import execute_vahera, parse_vahera

    kernel = kernel or Kernel()
    stmts = parse_vahera(program)  # execute_vahera re-parses internally; both
                                    # are pure/deterministic so this is safe,
                                    # just redundant — cheap for program sizes here
    ctx = execute_vahera(program, kernel=kernel, molecule_data=molecule_data)
    report = narrate(ctx, kernel, stmts)
    return ctx, report
