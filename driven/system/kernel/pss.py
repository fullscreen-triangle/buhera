"""
Penultimate State Scheduler.

Schedules processes by trajectory distance d_traj(p) = d(S_cur, S_f).
The process with minimum trajectory distance gets CPU priority.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

from ..substrate import SCoord, s_distance


@dataclass
class Process:
    pid: int
    program_name: str
    s_initial: SCoord
    s_final: SCoord
    s_current: SCoord
    state: str = "ready"      # ready | running | completed | blocked
    result: Any = None
    events: list[str] = field(default_factory=list)


class PSS:
    """Penultimate State Scheduler."""

    def __init__(self):
        self._heap: list[tuple[float, int, Process]] = []
        self._tie = itertools.count()
        self._processes: dict[int, Process] = {}
        self._next_pid = itertools.count(1)
        self._events: list[str] = []

    def spawn(self, program_name: str, s_initial: SCoord,
              s_final: SCoord) -> Process:
        pid = next(self._next_pid)
        p = Process(pid=pid, program_name=program_name,
                    s_initial=s_initial, s_final=s_final,
                    s_current=s_initial)
        self._processes[pid] = p
        d = s_distance(s_initial, s_final)
        heapq.heappush(self._heap, (d, next(self._tie), p))
        self._events.append(
            f"PSS.spawn pid={pid} program={program_name} d_traj={d:.3f}")
        return p

    def next(self) -> Optional[Process]:
        """Pop the process with minimum trajectory distance."""
        while self._heap:
            d, _, p = heapq.heappop(self._heap)
            if p.state == "ready":
                p.state = "running"
                self._events.append(
                    f"PSS.schedule pid={p.pid} d_traj={d:.3f}")
                return p
        return None

    def advance(self, p: Process, new_coord: SCoord):
        p.s_current = new_coord
        d = s_distance(new_coord, p.s_final)
        p.events.append(f"advance to {new_coord} d_traj={d:.3f}")
        if d < 1e-3:
            p.state = "completed"
            self._events.append(f"PSS.complete pid={p.pid}")
        else:
            p.state = "ready"
            heapq.heappush(self._heap, (d, next(self._tie), p))

    def get(self, pid: int) -> Optional[Process]:
        return self._processes.get(pid)

    def events(self) -> list[str]:
        return list(self._events)
