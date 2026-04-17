"""
Triple Equivalence Monitor.

Samples the kernel state every tick and verifies that the oscillator,
categorical, and partition descriptions yield the same entropy.
Records divergence events for later analysis.
"""
from __future__ import annotations

import math
from typing import Optional

from ..substrate import SCoord


class TEM:
    def __init__(self, threshold: float = 1e-3):
        self.threshold = threshold
        self._samples: list[dict] = []
        self._alerts: list[str] = []

    def sample(self, coord: SCoord, description: str = ""):
        """Record a state sample; check triple equivalence."""
        # For each triple (k,t,e) compute "three equivalent entropies"
        # as three different reductions of the same coordinate vector.
        # In the idealized substrate these are equal; we just log.
        k, t, e = coord.k, coord.t, coord.e
        S_osc = -sum(v * math.log(v + 1e-9) for v in (k, t, e))
        S_cat = (k + t + e) / 3 * math.log(3)
        S_par = math.sqrt(k * k + t * t + e * e) * math.log(3)

        # The three should be within the threshold after normalization
        max_s = max(S_osc, S_cat, S_par, 1e-9)
        delta = max(abs(S_osc - S_cat), abs(S_cat - S_par), abs(S_osc - S_par)) / max_s

        self._samples.append({
            "coord": coord,
            "S_osc": S_osc, "S_cat": S_cat, "S_par": S_par,
            "delta": delta, "description": description,
        })

        if delta > self.threshold:
            self._alerts.append(
                f"TEM alert: delta={delta:.4f} at {description}")

    def stats(self) -> dict:
        return {
            "samples": len(self._samples),
            "alerts": len(self._alerts),
            "max_delta": max((s["delta"] for s in self._samples), default=0),
        }

    def events(self) -> list[str]:
        return list(self._alerts)
