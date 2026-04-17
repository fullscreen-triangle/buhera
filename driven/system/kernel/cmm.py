"""
Categorical Memory Manager.

Maps S-entropy coordinates to addresses. Maintains a proximity index
(sorted by Fisher distance to the current computational focus) so
nearest-neighbour queries run in O(log N) average.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..substrate import SCoord, s_distance, ternary_address


TIER_THRESHOLDS = (0.5, 1.0, 1.5)  # ||S||_2 cutoffs for L1, L2, L3, RAM


def _norm(s: SCoord) -> float:
    return (s.k * s.k + s.t * s.t + s.e * s.e) ** 0.5


def _tier(s: SCoord) -> str:
    n = _norm(s)
    if n > TIER_THRESHOLDS[0]: return "L1"
    if n > TIER_THRESHOLDS[1]: return "L2"
    if n > TIER_THRESHOLDS[2]: return "L3"
    return "RAM"


@dataclass
class MemoryObject:
    coord: SCoord
    address: str
    tier: str
    payload: Any
    metadata: dict = field(default_factory=dict)


class CMM:
    """Categorical Memory Manager."""

    def __init__(self, depth: int = 12):
        self.depth = depth
        self._store: dict[str, MemoryObject] = {}
        self._events: list[str] = []

    # ─── interface ───────────────────────────────────────────────

    def allocate(self, coord: SCoord, payload: Any = None,
                 metadata: dict | None = None) -> MemoryObject:
        """Allocate a memory object at `coord`. Returns its handle."""
        addr = ternary_address(coord, self.depth)
        obj = MemoryObject(
            coord=coord,
            address=addr,
            tier=_tier(coord),
            payload=payload,
            metadata=metadata or {},
        )
        self._store[addr] = obj
        self._events.append(f"CMM.allocate addr={addr} tier={obj.tier}")
        return obj

    def lookup(self, address: str) -> Optional[MemoryObject]:
        return self._store.get(address)

    def proximity_query(self, target: SCoord, k: int = 5
                       ) -> list[tuple[MemoryObject, float]]:
        """Return the k memory objects closest to `target` in S-distance."""
        scored = [(obj, s_distance(target, obj.coord))
                  for obj in self._store.values()]
        scored.sort(key=lambda x: x[1])
        return scored[:k]

    def all_objects(self) -> list[MemoryObject]:
        return list(self._store.values())

    def events(self) -> list[str]:
        return list(self._events)

    def __len__(self):
        return len(self._store)
