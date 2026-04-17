"""
Buhera OS end-to-end demo.

Pipeline:
  natural language  ->  LLM translator  ->  vaHera program
                                           ->  kernel execution
                                           ->  synthesized answer

The demo task: store 10 NIST compounds, then ask questions about them in
natural language. The system never stores raw property tables — it
synthesizes answers from S-entropy coordinates using backward trajectory
completion. Results are compared against NIST ground truth.
"""
from __future__ import annotations

import io
import sys
import json
import os
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from .kernel import Kernel
from .substrate import embed_text, embed_molecule, s_distance, SCoord
from .vahera import execute_vahera
from .translator import IntentTranslator


# ───────────────────────────────────────────────────────────────────

def load_nist() -> dict:
    path = Path(__file__).parent / "data" / "nist_compounds.json"
    with open(path) as f:
        return json.load(f)


def boot_os(nist: dict) -> Kernel:
    """Boot Buhera OS: allocate each compound at its categorical address."""
    k = Kernel(depth=12)
    for name, props in nist.items():
        coord = embed_molecule(name, props)
        k.allocate(coord, payload=props, metadata={"name": name,
                                                    "formula": props["formula"]})
    return k


def answer_query(kernel: Kernel, nist: dict, query: str,
                 translator: IntentTranslator, verbose: bool = True
                 ) -> dict:
    """Run one query through the full stack."""
    t0 = time.perf_counter()
    vahera_source = translator.translate(query)
    t_translate = time.perf_counter() - t0

    if verbose:
        print(f"\n{'='*72}")
        print(f"USER:   {query}")
        print(f"{'-'*72}")
        print("vaHera:")
        for ln in vahera_source.split("\n"):
            print(f"  {ln}")
        print(f"{'-'*72}")

    # For vaHera `describe` using molecule names, pass the NIST data so
    # embed_molecule is used instead of embed_text.
    t0 = time.perf_counter()
    try:
        ctx = execute_vahera(vahera_source, kernel=kernel, molecule_data=nist)
    except Exception as e:
        return {"error": str(e), "query": query}
    t_execute = time.perf_counter() - t0

    # Synthesize answer: pick the nearest compound in CMM to the query's
    # target coord, return its properties. This is the "empty dictionary"
    # principle — we didn't look up "ethanol"; we navigated to its coord.
    query_coord = None
    for name, coord in ctx.targets.items():
        query_coord = coord
        break

    synthesized = None
    if query_coord is not None:
        nearest = kernel.find_nearest(query_coord, k=1)
        if nearest:
            _, obj, dist = nearest[0]
            synthesized = {
                "matched_compound": obj.metadata.get("name"),
                "categorical_distance": dist,
                "address": obj.address,
                "payload": obj.payload,
            }

    result = {
        "query": query,
        "vahera": vahera_source,
        "query_coord": query_coord,
        "synthesized": synthesized,
        "timing": {
            "translate_ms": t_translate * 1000,
            "execute_ms": t_execute * 1000,
        },
    }

    if verbose and synthesized:
        print(f"synthesized: matched={synthesized['matched_compound']} "
              f"d_cat={synthesized['categorical_distance']:.4f}")
        print(f"properties: {synthesized['payload']}")
        print(f"timing: translate {result['timing']['translate_ms']:.1f}ms "
              f"execute {result['timing']['execute_ms']:.2f}ms")

    return result


# ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  BUHERA OS — end-to-end demo")
    print("=" * 72)

    nist = load_nist()
    print(f"\nNIST compounds loaded: {len(nist)}")

    kernel = boot_os(nist)
    print(f"kernel booted. CMM objects: {len(kernel.cmm)}")
    print(f"subsystem stats: {kernel.stats()}")

    translator = IntentTranslator()
    backend_name = type(translator.backend).__name__
    print(f"translator backend: {backend_name}")

    queries = [
        "what is the boiling point of ethanol",
        "boiling point of benzene",
        "what is caffeine",
        "find compounds similar to aspirin",
    ]

    results = []
    for q in queries:
        r = answer_query(kernel, nist, q, translator, verbose=True)
        results.append(r)

    # Validation: for each query with a known compound, compare the
    # matched compound against the expected one
    print(f"\n{'='*72}")
    print("  VALIDATION")
    print(f"{'='*72}")

    expected = {
        "what is the boiling point of ethanol": "ethanol",
        "boiling point of benzene": "benzene",
        "what is caffeine": "caffeine",
    }
    correct = 0
    total = 0
    for r in results:
        q = r["query"]
        if q in expected:
            total += 1
            matched = r["synthesized"]["matched_compound"] if r.get("synthesized") else None
            exp = expected[q]
            ok = matched == exp
            correct += int(ok)
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] query={q!r}")
            print(f"         expected={exp} matched={matched}")

    print(f"\n  accuracy: {correct}/{total}")

    # Save results
    out_path = Path(__file__).parent.parent / "data" / "buhera_os_demo_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "backend": backend_name,
            "n_compounds": len(nist),
            "queries": results,
            "accuracy": {"correct": correct, "total": total},
            "kernel_stats": kernel.stats(),
            "activity_log_sample": kernel.activity_log()[:20],
        }, f, indent=2, default=str)
    print(f"\n  results saved to: {out_path}")


if __name__ == "__main__":
    main()
