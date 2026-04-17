"""
Buhera OS — Categorical Substrate.

Pure mathematics that every subsystem consumes:
  - S-entropy coordinates (S_k, S_t, S_e) in [0,1]^3
  - Ternary partition hierarchy
  - Fisher information metric
  - Backward navigation (geodesic flow)
  - Completion morphism
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence


# ─────────────────────────────────────────────────────────────────────
#  S-entropy coordinates
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SCoord:
    """A point in S-entropy space [0,1]^3."""
    k: float  # knowledge — information deficit
    t: float  # temporal  — position in completion sequence
    e: float  # entropy   — constraint density

    def __post_init__(self):
        for name, v in [("k", self.k), ("t", self.t), ("e", self.e)]:
            if not (-1e-9 <= v <= 1 + 1e-9):
                raise ValueError(f"SCoord.{name}={v} outside [0,1]")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.k, self.t, self.e)

    def __iter__(self):
        yield self.k; yield self.t; yield self.e


# ─────────────────────────────────────────────────────────────────────
#  Embedding: content -> SCoord
# ─────────────────────────────────────────────────────────────────────

def embed_text(content: str) -> SCoord:
    """
    Deterministic embedding of text content into S-entropy space.
    Not learned — derived from content properties:
      S_k: character entropy (information density)
      S_t: temporal-marker ratio (temporal structure)
      S_e: action-verb + question-mark density (transformation complexity)
    """
    if not content:
        return SCoord(0.0, 0.0, 0.0)

    text = content.strip().lower()
    chars = [c for c in text if not c.isspace()]
    n = max(len(chars), 1)

    # S_k: Shannon entropy over characters, normalized by log2(26)
    freq = {}
    for c in chars:
        freq[c] = freq.get(c, 0) + 1
    H = 0.0
    for count in freq.values():
        p = count / n
        H -= p * math.log2(p)
    sk = min(H / math.log2(26), 1.0)

    # S_t: fraction of words that are temporal markers
    temporal = {"when", "before", "after", "during", "now", "then",
                "yesterday", "today", "recently", "previously", "last",
                "next", "current", "old", "new", "past", "future"}
    words = [w for w in text.replace("?", " ").replace("!", " ").split() if w]
    t_hits = sum(1 for w in words if w in temporal)
    st = min(t_hits / max(len(words) * 0.3, 1), 1.0)

    # S_e: action-density (questions + action verbs)
    actions = {"what", "how", "why", "find", "show", "compute", "predict",
               "compare", "analyze", "identify", "measure", "synthesize",
               "determine", "calculate", "derive"}
    a_hits = sum(1 for w in words if w in actions) + content.count("?")
    se = min(a_hits / max(len(words) * 0.4, 1), 1.0)

    return SCoord(sk, st, se)


def embed_molecule(formula: str, properties: dict) -> SCoord:
    """
    Embedding for a molecule whose vibrational/structural properties are known.
      S_k: normalized Shannon entropy over vibrational mode energies
      S_t: normalized log-span of timescales
      S_e: normalized anharmonic coupling density
    For now uses a deterministic hash of the formula plus property signals —
    sufficient to give every molecule a unique, content-derived address.
    """
    # Seed with formula hash
    seed = int(hashlib.sha256(formula.encode()).hexdigest()[:12], 16)
    rng_k = ((seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff
    rng_t = ((seed * 1140671485 + 12820163) & 0x7fffffff) / 0x7fffffff
    rng_e = ((seed * 214013 + 2531011) & 0x7fffffff) / 0x7fffffff

    # Mix with measurable properties if available
    if "molecular_weight" in properties:
        mw = properties["molecular_weight"]
        rng_k = (rng_k + math.log(mw + 1) / 10) % 1.0
    if "boiling_point_c" in properties:
        bp = properties["boiling_point_c"]
        rng_t = (rng_t + (bp + 273) / 1000) % 1.0
    if "n_atoms" in properties:
        na = properties["n_atoms"]
        rng_e = (rng_e + math.log(na + 1) / 10) % 1.0

    return SCoord(rng_k, rng_t, rng_e)


# ─────────────────────────────────────────────────────────────────────
#  Fisher information metric
# ─────────────────────────────────────────────────────────────────────

def fisher_distance_1d(a: float, b: float, eps: float = 1e-6) -> float:
    """Geodesic distance on (0,1) with Fisher metric ds^2 = dx^2 / (x(1-x))."""
    a = min(max(a, eps), 1 - eps)
    b = min(max(b, eps), 1 - eps)
    return abs(math.asin(2 * a - 1) - math.asin(2 * b - 1))


def s_distance(s1: SCoord, s2: SCoord) -> float:
    """Product-Fisher distance on [0,1]^3."""
    dk = fisher_distance_1d(s1.k, s2.k)
    dt = fisher_distance_1d(s1.t, s2.t)
    de = fisher_distance_1d(s1.e, s2.e)
    return math.sqrt(dk * dk + dt * dt + de * de)


# ─────────────────────────────────────────────────────────────────────
#  Ternary address encoding
# ─────────────────────────────────────────────────────────────────────

def ternary_address(s: SCoord, depth: int) -> str:
    """
    Encode an SCoord as an interleaved ternary string of length `depth`.
    Digit 3j+0 refines k; 3j+1 refines t; 3j+2 refines e.
    """
    ranges = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    vals = [s.k, s.t, s.e]
    digits = []
    for d in range(depth):
        dim = d % 3
        lo, hi = ranges[dim]
        third = (hi - lo) / 3
        v = vals[dim]
        if v < lo + third:
            trit = 0
        elif v < lo + 2 * third:
            trit = 1
        else:
            trit = 2
        digits.append(trit)
        ranges[dim] = [lo + trit * third, lo + (trit + 1) * third]
    return "".join(str(d) for d in digits)


def common_prefix_length(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# ─────────────────────────────────────────────────────────────────────
#  Backward trajectory completion
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    final: SCoord
    initial: SCoord
    path: list[SCoord]           # coarse-to-fine nodes along geodesic
    miracle_count: int           # virtual sub-states crossed
    steps: int                   # log3(N)


def backward_navigate(final: SCoord, depth: int) -> Trajectory:
    """
    Geodesic backward navigation from `final` to the root (1, 0, 0).
    Returns the trajectory including intermediate penultimate states.

    Complexity: O(depth) = O(log_3 N).
    """
    initial = SCoord(1.0, 0.0, 0.0)
    path: list[SCoord] = [final]

    # Walk backward k steps, each refinement reversed
    for j in range(depth, 0, -1):
        # At each step, coarsen one coordinate by approximating the parent's centroid
        # The parent cell's centroid is the midpoint of the refined range
        alpha = (depth - j + 1) / depth
        k = final.k + (initial.k - final.k) * alpha
        t = final.t + (initial.t - final.t) * alpha
        e = final.e + (initial.e - final.e) * alpha
        path.append(SCoord(min(max(k, 0), 1),
                           min(max(t, 0), 1),
                           min(max(e, 0), 1)))

    path.reverse()  # now initial -> final
    # miracle count = remaining unresolved ternary decisions at each step
    miracles = depth  # one per level, decreasing as we step forward
    return Trajectory(final=final, initial=initial, path=path,
                      miracle_count=miracles, steps=depth)


def completion_morphism(penultimate: SCoord, final: SCoord) -> SCoord:
    """
    Apply the single completion morphism from penultimate to final state.
    By definition returns the final state; this is where the "answer is
    synthesized from coordinates" semantically happens.
    """
    return final


# ─────────────────────────────────────────────────────────────────────
#  Proximity query over a set of known SCoords
# ─────────────────────────────────────────────────────────────────────

def nearest(target: SCoord, candidates: Sequence[SCoord], k: int = 1
           ) -> list[tuple[int, float]]:
    """Return (index, distance) for the k nearest candidates to `target`."""
    scored = [(i, s_distance(target, c)) for i, c in enumerate(candidates)]
    scored.sort(key=lambda x: x[1])
    return scored[:k]
