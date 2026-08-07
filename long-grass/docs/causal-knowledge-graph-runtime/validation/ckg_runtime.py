"""A minimal, faithful model of the causal-knowledge-graph runtime.

This is the substrate the five validators exercise. It mirrors the shape of the
production runtime (``long-grass/src/lib/modules/registry.js``): a module has an
id and an ``execute`` step returning a value-delta plus a completion flag; a
dispatcher runs *every* chunk on a node and appends an audit record WITHOUT
inspecting the returned values; the only run output is a report assembled from
emitted values.

Nothing here judges a result. The runtime is semantically inert by construction:
it moves values by running chunks, and what a value means is left entirely to the
modules that read it. Keeping this file free of any comparison-to-expectation is
the whole point -- the theorems in the paper are properties of *this* inertia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# --- values, chunks, nodes ---------------------------------------------------

@dataclass
class ValueDelta:
    """What a chunk emits. `payload` is opaque to the runtime.

    `kind` lets a downstream *module* (never the runtime) recognise a value it
    cares about -- including an error record, which is just another value.
    """
    kind: str
    payload: Any
    source_chunk: str


# A chunk is a pure function from the node's current values to a value-delta.
# It may raise; the runtime turns that into a recorded value, it does not halt.
Chunk = Callable[[dict[str, Any]], ValueDelta]


@dataclass
class Node:
    """A subtask fused with its realising code (Def. 1 in the paper).

    * `tau`      -- subtask identity; two realisations with the same tau converge.
    * `chunks`   -- a *bag* of realisations; the runtime runs all of them.
    * `values`   -- the time-varying values the node carries.
    * `address`  -- hierarchical address (coarsest..finest); depth = resolution.
    """
    tau: str
    chunks: dict[str, Chunk] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    address: tuple[str, ...] = ()

    def add_chunk(self, name: str, chunk: Chunk) -> None:
        # Convergence merges chunk bags: adding onto an existing tau accretes.
        self.chunks[name] = chunk


# --- the runtime: execute all chunks, judge nothing --------------------------

@dataclass
class AuditEntry:
    act_id: int
    tau: str
    chunk: str
    emitted_kind: str
    raised: bool


class Runtime:
    """Executes nodes. Holds no expectation, issues no exit code.

    `dispatch(node)` runs every chunk in the node's bag (Def. 3). A chunk that
    raises produces an ``error`` value-delta -- emitted like any other value --
    and the run continues (Thm. 2 / Cor. 4). The runtime never branches on the
    *content* of an emission.
    """

    def __init__(self) -> None:
        self.audit: list[AuditEntry] = []
        self._act = 0

    def dispatch(self, node: Node) -> list[ValueDelta]:
        deltas: list[ValueDelta] = []
        for name, chunk in node.chunks.items():
            self._act += 1
            raised = False
            try:
                delta = chunk(node.values)
            except Exception as exc:  # noqa: BLE001 -- error becomes a value, not a halt
                raised = True
                delta = ValueDelta(kind="error", payload=repr(exc), source_chunk=name)
            # emit onto the node; the runtime does not inspect payload meaning
            node.values[delta.kind] = delta.payload
            deltas.append(delta)
            self.audit.append(
                AuditEntry(
                    act_id=self._act,
                    tau=node.tau,
                    chunk=name,
                    emitted_kind=delta.kind,
                    raised=raised,
                )
            )
        return deltas

    # There is deliberately no `exit_code`, no `ok`, no `verdict` method here.


# --- modules: read / transform / emit ---------------------------------------

class Module:
    """A competence that reads node values, transforms internally, and emits.

    Whether this module reads a given node in a run is decided by `wants`, which
    consults values *already present in this run* -- the mechanism behind
    trajectory emergence (Thm. 5): the causal edge relation is a product of the
    run, not an input to it.
    """

    def __init__(
        self,
        mid: str,
        wants: Callable[[Node], bool],
        emit: Callable[[Node], Optional[ValueDelta]],
    ) -> None:
        self.id = mid
        self._wants = wants
        self._emit = emit

    def wants(self, node: Node) -> bool:
        return self._wants(node)

    def read_transform_emit(self, node: Node) -> Optional[ValueDelta]:
        if not self._wants(node):
            return None
        delta = self._emit(node)
        if delta is not None:
            node.values[delta.kind] = delta.payload
        return delta


# --- provenance fingerprint (reproducibility of protocol, not results) -------

def provenance_fingerprint(nodes: list[Node], edit_addresses: list[tuple[str, ...]]) -> str:
    """A stable hash of the *protocol*: imported subtree + chunk bag + edits.

    Deliberately excludes node *values* -- results are not expected to repeat,
    and must not enter the fingerprint (Prop. 8). What repeats is this string.
    """
    import hashlib

    parts: list[str] = []
    for n in sorted(nodes, key=lambda x: x.address):
        parts.append("/".join(n.address) + ":" + tau_shape(n))
    parts.append("EDITS=" + ";".join("/".join(a) for a in sorted(edit_addresses)))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def tau_shape(node: Node) -> str:
    """Structure of a node independent of its (run-varying) values."""
    return node.tau + "{" + ",".join(sorted(node.chunks)) + "}"
