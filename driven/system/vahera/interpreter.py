"""
vaHera — the OS's internal declarative language.

Minimal but real grammar covering the core categorical operations.
The kernel emits vaHera when executing work; this module parses and
executes it by dispatching to kernel subsystems.

Supported statements (one per line):
  resolve <target-name>                         # lookup/compute final S-coord
  describe <target-name> with "<text>"          # bind content to compute coord
  spawn <program-name> from <target-name>       # create a categorical process
  navigate to penultimate                       # backward navigation
  complete trajectory                           # apply completion morphism
  memory create at S(<k>,<t>,<e>)               # allocate an object
  memory store "<name>" = "<text>"              # store text at its content-coord
  memory find nearest "<text>" k=<n>            # categorical retrieval by description
  demon sort                                    # zero-cost sort
  controller verify                             # check triple equivalence
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..substrate import SCoord, embed_text, embed_molecule
from ..kernel import Kernel


@dataclass
class Stmt:
    op: str
    args: dict


# ─── parser ──────────────────────────────────────────────────────────

_S_COORD_RE = re.compile(r"S\(\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*,\s*([-\d.e]+)\s*\)")


def _parse_s_coord(text: str) -> SCoord:
    m = _S_COORD_RE.search(text)
    if not m:
        raise SyntaxError(f"expected S(k,t,e) in: {text}")
    k, t, e = map(float, m.groups())
    return SCoord(k, t, e)


def _parse_str_arg(line: str) -> str:
    m = re.search(r'"([^"]*)"', line)
    if not m:
        raise SyntaxError(f"expected quoted string in: {line}")
    return m.group(1)


def parse_vahera(source: str) -> list[Stmt]:
    """Parse vaHera source to a list of statements."""
    stmts: list[Stmt] = []
    for raw in source.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("resolve "):
            target = line[len("resolve "):].strip()
            stmts.append(Stmt("resolve", {"target": target}))

        elif line.startswith("describe "):
            m = re.match(r'describe\s+(\S+)\s+with\s+"([^"]*)"', line)
            if not m:
                raise SyntaxError(f"malformed: {line}")
            stmts.append(Stmt("describe",
                              {"target": m.group(1), "text": m.group(2)}))

        elif line.startswith("spawn "):
            m = re.match(r'spawn\s+(\S+)\s+from\s+(\S+)', line)
            if not m:
                raise SyntaxError(f"malformed: {line}")
            stmts.append(Stmt("spawn",
                              {"program": m.group(1), "target": m.group(2)}))

        elif line == "navigate to penultimate":
            stmts.append(Stmt("navigate", {"mode": "penultimate"}))

        elif line == "complete trajectory":
            stmts.append(Stmt("complete", {}))

        elif line.startswith("memory create at"):
            coord = _parse_s_coord(line)
            stmts.append(Stmt("memory_create", {"coord": coord}))

        elif line.startswith("memory store"):
            m = re.match(r'memory\s+store\s+"([^"]*)"\s*=\s*"([^"]*)"', line)
            if not m:
                raise SyntaxError(f"malformed: {line}")
            stmts.append(Stmt("memory_store",
                              {"name": m.group(1), "text": m.group(2)}))

        elif line.startswith("memory find nearest"):
            m = re.match(r'memory\s+find\s+nearest\s+"([^"]*)"(?:\s+k=(\d+))?', line)
            if not m:
                raise SyntaxError(f"malformed: {line}")
            k = int(m.group(2)) if m.group(2) else 3
            stmts.append(Stmt("memory_find",
                              {"query": m.group(1), "k": k}))

        elif line == "demon sort":
            stmts.append(Stmt("demon_sort", {}))

        elif line == "controller verify":
            stmts.append(Stmt("controller_verify", {}))

        else:
            raise SyntaxError(f"unknown vaHera: {line}")

    return stmts


# ─── interpreter ─────────────────────────────────────────────────────

def _resolve_coord(name: str, text: str, molecule_data: dict) -> SCoord:
    """
    Figure out which embedding to use for a describe/resolve target.
    If the name or text mentions a known molecule (case-insensitive token
    match), use the molecule embedding. Otherwise fall back to text.
    """
    if not molecule_data:
        return embed_text(text)
    name_lc = name.lower()
    text_lc = text.lower()
    # exact name match
    for mol_name, props in molecule_data.items():
        if mol_name.lower() == name_lc:
            return embed_molecule(mol_name, props)
    # token match in name or text
    name_tokens = set(re.split(r"[\s_\-]+", name_lc)) | set(text_lc.split())
    for mol_name, props in molecule_data.items():
        if mol_name.lower() in name_tokens:
            return embed_molecule(mol_name, props)
    return embed_text(text)


@dataclass
class ExecContext:
    kernel: Kernel
    targets: dict[str, SCoord]          # name -> final-state coord
    processes: dict[str, Any]           # program_name -> Process
    results: list[Any]
    trace: list[str]


def execute_vahera(program: str, kernel: Optional[Kernel] = None,
                   molecule_data: dict | None = None) -> ExecContext:
    """Parse and execute a vaHera program on a kernel."""
    kernel = kernel or Kernel()
    molecule_data = molecule_data or {}

    ctx = ExecContext(kernel=kernel, targets={}, processes={},
                      results=[], trace=[])

    stmts = parse_vahera(program)

    for stmt in stmts:
        op = stmt.op

        if op == "describe":
            name = stmt.args["target"]
            text = stmt.args["text"]
            coord = _resolve_coord(name, text, molecule_data)
            ctx.targets[name] = coord
            ctx.trace.append(f"describe {name} -> S({coord.k:.3f},{coord.t:.3f},{coord.e:.3f})")

        elif op == "resolve":
            name = stmt.args["target"]
            if name not in ctx.targets:
                # resolve by content-derived coord from the name itself
                ctx.targets[name] = _resolve_coord(name, name, molecule_data)
            coord = ctx.targets[name]
            ctx.trace.append(f"resolve {name} -> {coord}")

        elif op == "spawn":
            prog = stmt.args["program"]
            tgt = stmt.args["target"]
            if tgt not in ctx.targets:
                raise RuntimeError(f"spawn: unresolved target {tgt}")
            s_f = ctx.targets[tgt]
            s_i = SCoord(1.0, 0.0, 0.0)  # root = maximum uncertainty
            p = kernel.spawn(prog, s_i, s_f)
            ctx.processes[prog] = p

        elif op == "navigate":
            if not ctx.processes:
                raise RuntimeError("navigate: no active process")
            prog, p = next(iter(ctx.processes.items()))
            kernel.navigate(p, p.s_final)

        elif op == "complete":
            if not ctx.processes:
                raise RuntimeError("complete: no active process")
            prog, p = next(iter(ctx.processes.items()))
            kernel.complete(p, p.s_final)

        elif op == "memory_create":
            kernel.allocate(stmt.args["coord"])

        elif op == "memory_store":
            text = stmt.args["text"]
            coord = embed_text(text)
            obj = kernel.store(coord, text, metadata={"name": stmt.args["name"]})
            ctx.trace.append(
                f"memory_store name={stmt.args['name']} -> addr={obj.address}")

        elif op == "memory_find":
            q_coord = embed_text(stmt.args["query"])
            results = kernel.find_nearest(q_coord, k=stmt.args["k"])
            ctx.results.extend(results)
            ctx.trace.append(
                f"memory_find query='{stmt.args['query']}' -> {len(results)} hits")

        elif op == "demon_sort":
            objs = [(o.coord, o) for o in kernel.cmm.all_objects()]
            sorted_objs = kernel.dic.categorical_sort(objs)
            ctx.results.extend(sorted_objs)

        elif op == "controller_verify":
            stats = kernel.tem.stats()
            ctx.trace.append(
                f"controller_verify samples={stats['samples']} "
                f"alerts={stats['alerts']} max_delta={stats['max_delta']:.5f}")

        else:
            raise RuntimeError(f"unimplemented vaHera op: {op}")

    return ctx
