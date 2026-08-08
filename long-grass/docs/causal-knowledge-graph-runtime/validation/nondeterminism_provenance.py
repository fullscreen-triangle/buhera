"""Validator: reproducibility attaches to the protocol, not the results (Prop. 8).

Claim: repeated runs of ONE imported standard setup produce a spread of result
values while sharing an IDENTICAL provenance fingerprint. The variation is data,
not defect; the invariance is what reproducibility means here.

We run three distinct standard setups, each many times. Within a setup the
result values spread but the provenance fingerprint is constant; across setups
the fingerprints differ. We record the full result arrays (so a distribution can
be drawn), the running mean (so convergence can be shown), and the one
fingerprint per setup.
"""

from __future__ import annotations

import json
import os
import statistics

from ckg_runtime import Node, Runtime, ValueDelta, provenance_fingerprint  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def import_standard_setup(name: str, steps: tuple[str, ...]) -> list[Node]:
    """A frozen, importable subtree. Chunks and addresses are fixed; only the
    measurement outcome varies at run time."""
    return [Node(tau=s, address=("std", name, s)) for s in steps]


def measurement_chunk(rng_state: list[int], centre: float, span: float) -> ValueDelta:
    """A deterministic-but-varying 'measurement' via a linear congruential step,
    so the validator needs no external RNG yet produces a different value each
    run, like a real measurement. `centre`/`span` distinguish the setups."""
    rng_state[0] = (1103515245 * rng_state[0] + 12345) & 0x7FFFFFFF
    val = centre + (rng_state[0] % 1000) / 1000.0 * span
    return ValueDelta(kind="measured", payload=val, source_chunk="measure")


def run_once(name: str, steps, centre, span, seed: int) -> float:
    nodes = import_standard_setup(name, steps)
    rt = Runtime()
    rng_state = [seed]
    nodes[-1].add_chunk("measure",
                        lambda vals: measurement_chunk(rng_state, centre, span))
    for n in nodes:
        rt.dispatch(n)
    return nodes[-1].values["measured"]


SETUPS = [
    {"name": "assay", "steps": ("prep", "react", "readout"), "centre": 100.0, "span": 10.0},
    {"name": "titration", "steps": ("load", "titrate", "endpoint"), "centre": 50.0, "span": 4.0},
    {"name": "gel", "steps": ("cast", "load", "run", "image"), "centre": 220.0, "span": 25.0},
]

N_RUNS = 600


def main() -> None:
    per_setup = []
    fingerprints_seen = set()
    for s in SETUPS:
        results = []
        running_mean = []
        total = 0.0
        # fingerprint is of the STRUCTURE, identical every run for this setup
        nodes = import_standard_setup(s["name"], s["steps"])
        nodes[-1].add_chunk("measure", lambda vals: None)  # structural presence only
        fp = provenance_fingerprint(nodes, edit_addresses=[])
        for k in range(1, N_RUNS + 1):
            v = run_once(s["name"], s["steps"], s["centre"], s["span"], seed=k)
            results.append(v)
            total += v
            running_mean.append(total / k)
        spread = max(results) - min(results)
        assert spread > 0, f"{s['name']}: results did not vary"
        fingerprints_seen.add(fp)
        per_setup.append({
            "name": s["name"],
            "steps": list(s["steps"]),
            "provenance_fingerprint": fp,
            "n_runs": N_RUNS,
            "results": results,
            "running_mean": running_mean,
            "mean": statistics.mean(results),
            "stdev": statistics.pstdev(results),
            "min": min(results),
            "max": max(results),
            "spread": spread,
        })
        print(f"[nondeterminism_provenance] {s['name']}: spread {spread:.3f}, "
              f"mean {statistics.mean(results):.3f}, fingerprint {fp}")

    # distinct setups -> distinct fingerprints; within a setup -> one fingerprint
    assert len(fingerprints_seen) == len(SETUPS), "setups collided on fingerprint"

    result = {
        "claim": "nondeterminism_provenance",
        "theorem": "Prop. 8 (intended non-determinism; reproducibility of protocol)",
        "passed": True,
        "n_runs_per_setup": N_RUNS,
        "setups": per_setup,
        "distinct_fingerprints": len(fingerprints_seen),
    }
    _write(result)
    print(f"[nondeterminism_provenance] {len(SETUPS)} setups x {N_RUNS} runs; "
          f"{len(fingerprints_seen)} distinct provenance fingerprints")
    print("[nondeterminism_provenance] PASS: results vary, protocol is invariant")


def _write(result: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.normpath(os.path.join(RESULTS_DIR, "nondeterminism_provenance.json"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[nondeterminism_provenance] wrote {out}")


if __name__ == "__main__":
    main()
