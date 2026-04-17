"""
The Buhera microkernel.

Orchestrates the five subsystems: CMM, PSS, DIC, PVE, TEM.
Exposes a single entry point — execute_vahera(program) — that accepts
a parsed vaHera program and walks it through the subsystems.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..substrate import SCoord, backward_navigate, completion_morphism, s_distance
from .cmm import CMM
from .pss import PSS, Process
from .dic import DIC
from .pve import PVE, PVEError
from .tem import TEM


@dataclass
class KernelResult:
    success: bool
    payload: Any = None
    trace: list[str] = None
    error: str = ""


class Kernel:
    def __init__(self, depth: int = 12):
        self.depth = depth
        self.cmm = CMM(depth=depth)
        self.pss = PSS()
        self.dic = DIC()
        self.pve = PVE()
        self.tem = TEM()
        self._trace: list[str] = []

    # ─────────── interpreter-facing primitives ───────────

    def allocate(self, coord: SCoord, payload=None, metadata=None):
        self.pve.verify("memory_create", {"coord": coord})
        obj = self.cmm.allocate(coord, payload, metadata)
        self.tem.sample(coord, f"allocate addr={obj.address}")
        self._trace.append(f"ALLOCATE coord={coord} -> addr={obj.address}")
        return obj

    def store(self, coord: SCoord, data: Any, metadata: dict | None = None):
        return self.allocate(coord, data, metadata or {})

    def find_nearest(self, target: SCoord, k: int = 5):
        """Proximity query through CMM k-d index, then DIC retrieval."""
        # CMM does the proximity scan (structure oracle)
        candidates = self.cmm.proximity_query(target, k=k * 3)
        # DIC performs surgical retrieval on the shortlist
        source = [(obj.coord, obj) for obj, _ in candidates]
        retrieved = self.dic.retrieve(source, target, max_results=k)
        self._trace.append(f"FIND_NEAREST target={target} -> {len(retrieved)} results")
        return retrieved

    def spawn(self, program_name: str, s_initial: SCoord, s_final: SCoord):
        self.pve.verify("resolve", {"target": program_name})
        p = self.pss.spawn(program_name, s_initial, s_final)
        self.tem.sample(s_final, f"spawn pid={p.pid}")
        self._trace.append(
            f"SPAWN pid={p.pid} {program_name} d_traj={s_distance(s_initial, s_final):.3f}")
        return p

    def navigate(self, process: Process, s_final: SCoord):
        """Backward-navigate `process` to the penultimate state of s_final."""
        self.pve.verify("navigate", {"mode": "penultimate"})
        traj = backward_navigate(s_final, self.depth)
        penultimate = traj.path[-2] if len(traj.path) >= 2 else traj.path[0]
        self.pss.advance(process, penultimate)
        self.tem.sample(penultimate, f"navigate pid={process.pid}")
        self._trace.append(
            f"NAVIGATE pid={process.pid} {traj.steps} steps "
            f"miracles={traj.miracle_count}")
        return traj

    def complete(self, process: Process, s_final: SCoord):
        """Apply the completion morphism from penultimate to final state."""
        self.pve.verify("complete", {"s_penultimate": process.s_current})
        new_coord = completion_morphism(process.s_current, s_final)
        self.pss.advance(process, new_coord)
        self.tem.sample(new_coord, f"complete pid={process.pid}")
        self._trace.append(f"COMPLETE pid={process.pid}")
        return new_coord

    # ─────────── diagnostics ───────────

    def activity_log(self) -> list[str]:
        """Unified activity trace across all subsystems."""
        log = []
        log.extend(self.cmm.events())
        log.extend(self.pss.events())
        log.extend(self.dic.events())
        log.extend(self.pve.events())
        log.extend(self.tem.events())
        return log

    def trace(self) -> list[str]:
        return list(self._trace)

    def stats(self) -> dict:
        return {
            "cmm_objects": len(self.cmm),
            "dic": self.dic.stats(),
            "pve": self.pve.stats(),
            "tem": self.tem.stats(),
        }
