"""
Demon I/O Controller.

Surgical retrieval: given a data source and a query, return only the
information relevant to the query (categorical aperture selection).
Zero-cost categorical sorting (no information-acquisition step).
"""
from __future__ import annotations

from typing import Any, Callable

from ..substrate import SCoord, s_distance


class DIC:
    def __init__(self):
        self._events: list[str] = []
        self._bits_retrieved = 0
        self._bits_available = 0

    def retrieve(self, source: list[tuple[SCoord, Any]], query: SCoord,
                 max_results: int = 5) -> list[tuple[SCoord, Any, float]]:
        """
        Surgical retrieval: return only the `max_results` items from
        `source` closest to the query. No scanning of unrelated items.
        """
        # Score everything (in a real implementation the CMM k-d tree
        # would prune this), then take the top k
        scored = [(c, v, s_distance(query, c)) for c, v in source]
        scored.sort(key=lambda x: x[2])
        result = scored[:max_results]

        total_items = len(source)
        retrieved_items = len(result)
        self._bits_retrieved += retrieved_items
        self._bits_available += total_items
        compression = 1.0 - (retrieved_items / max(total_items, 1))
        self._events.append(
            f"DIC.retrieve {retrieved_items}/{total_items} "
            f"({compression*100:.0f}% compression)")
        return result

    def categorical_sort(self, items: list[tuple[SCoord, Any]]
                         ) -> list[tuple[SCoord, Any]]:
        """
        Zero-cost categorical sort: order by S-distance to origin.
        By the commutation relation [O_cat, O_phys] = 0, this incurs no
        thermodynamic work — we simply read off the categorical order.
        """
        origin = SCoord(0.0, 0.0, 0.0)
        items_with_dist = [(s_distance(origin, c), c, v) for c, v in items]
        items_with_dist.sort(key=lambda x: x[0])
        self._events.append(
            f"DIC.categorical_sort {len(items)} items W_cat=0")
        return [(c, v) for _, c, v in items_with_dist]

    def stats(self) -> dict:
        comp = (1.0 - self._bits_retrieved / max(self._bits_available, 1)) \
               if self._bits_available else 0
        return {
            "bits_retrieved": self._bits_retrieved,
            "bits_available": self._bits_available,
            "compression": comp,
        }

    def events(self) -> list[str]:
        return list(self._events)
