"""Validator: reproducibility attaches to the protocol, not the results (Prop. 8).

Claim: repeated runs of ONE imported standard setup produce a spread of result
values while sharing an IDENTICAL provenance fingerprint. The variation is data,
not defect; the invariance is what reproducibility means here.

We import one standard setup (a fixed node subtree with a fixed chunk bag and
fixed edit addresses), run it many times with a measurement-like chunk whose
output varies between runs, and check: (a) the result values spread, and (b) the
provenance fingerprint -- which excludes values by construction -- is constant.
"""

from __future__ import annotations

import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ckg_runtime import Node, Runtime, ValueDelta, provenance_fingerprint  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def import_standard_setup() -> list[Node]:
    """The frozen, importable subtree. Chunks and addresses are fixed; only the
    measurement outcome varies at run time."""
    nodes = []
    for step in ("prep", "react", "readout"):
        n = Node(tau=step, address=("std", "assay", step))
        nodes.append(n)
    return nodes


def measurement_chunk(rng_state: list[int]) -> ValueDelta:
    """A deterministic-but-varying 'measurement': a linear congruential step, so
    the validator needs no external RNG and is itself reproducible as a protocol,
    while still producing a *different value each run* like a real measurement."""
    rng_state[0] = (1103515245 * rng_state[0] + 12345) & 0x7FFFFFFF
    val = 100.0 + (rng_state[0] % 1000) / 100.0  # ~100..110
    return ValueDelta(kind="measured", payload=val, source_chunk="measure")


def run_once(seed: int) -> float:
    nodes = import_standard_setup()
    rt = Runtime()
    rng_state = [seed]
    # attach the measurement chunk to the readout node for this run
    readout = nodes[-1]
    readout.add_chunk("measure", lambda vals: measurement_chunk(rng_state))
    for n in nodes:
        rt.dispatch(n)
    return readout.values["measured"]


def main() -> None:
    edit_addresses: list[tuple[str, ...]] = []  # coarse import: no surgical edits

    # provenance is computed from the STRUCTURE, identical every run
    fingerprints = set()
    results = []
    for seed in range(1, 401):
        # rebuild the setup each run so the fingerprint reflects the protocol,
        # not accumulated state
        nodes = import_standard_setup()
        rng_state = [seed]
        nodes[-1].add_chunk("measure", lambda vals, s=rng_state: measurement_chunk(s))
        fp = provenance_fingerprint(nodes, edit_addresses)
        fingerprints.add(fp)
        results.append(run_once(seed))

    spread = max(results) - min(results)
    assert len(fingerprints) == 1, f"provenance drifted across runs: {fingerprints}"
    assert spread > 0, "results did not vary -- non-determinism not exercised"

    fp = next(iter(fingerprints))
    print(f"[nondeterminism_provenance] runs: {len(results)}")
    print(f"[nondeterminism_provenance] result spread: {spread:.2f} "
          f"(min {min(results):.2f}, max {max(results):.2f}, "
          f"mean {statistics.mean(results):.2f})")
    print(f"[nondeterminism_provenance] single provenance fingerprint: {fp}")
    print("[nondeterminism_provenance] PASS: results vary, protocol is invariant")

    _plot(results, fp)


def _plot(results, fp) -> None:
    fig, (ax_hist, ax_prov) = plt.subplots(1, 2, figsize=(12.5, 3.8),
                                           gridspec_kw={"width_ratios": [2, 1]})
    ax_hist.hist(results, bins=30, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax_hist.axvline(sum(results) / len(results), color="#d62728", lw=2, label="mean")
    ax_hist.set_title("Results vary across runs (this is data, not defect)", fontsize=12)
    ax_hist.set_xlabel("measured value")
    ax_hist.set_ylabel("runs")
    ax_hist.legend()

    ax_prov.axis("off")
    ax_prov.set_title("Provenance: identical every run", fontsize=12)
    ax_prov.text(0.5, 0.72, "imported setup", ha="center", fontsize=10, color="#555")
    ax_prov.text(0.5, 0.60, "std/assay/{prep,react,readout}", ha="center", fontsize=8.5,
                 family="monospace")
    ax_prov.text(0.5, 0.44, "edits: (none -- coarse import)", ha="center", fontsize=9,
                 color="#555")
    ax_prov.text(0.5, 0.26, "fingerprint", ha="center", fontsize=10, color="#555")
    ax_prov.text(0.5, 0.14, fp, ha="center", fontsize=13, family="monospace",
                 color="#2ca02c", weight="bold")
    ax_prov.set_xlim(0, 1)
    ax_prov.set_ylim(0, 1)

    fig.suptitle("Reproducibility is of the protocol, not the numbers", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.normpath(os.path.join(FIG_DIR, "nondeterminism_provenance.png"))
    fig.savefig(out, dpi=150)
    print(f"[nondeterminism_provenance] wrote {out}")


if __name__ == "__main__":
    main()
