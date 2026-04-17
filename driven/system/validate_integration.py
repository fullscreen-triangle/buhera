"""
Integration validation for the Buhera OS stack.

Runs six experiments measuring:
  1. End-to-end latency across the four layers
  2. Accuracy across query types
  3. Dispatch protocol activity (which subsystems, how often)
  4. Scaling with number of stored compounds
  5. Empty dictionary principle (coord-proximity of matches)
  6. Layer robustness / error handling

Saves all results to data/integration_validation_results.json.
"""
from __future__ import annotations

import io
import sys
import json
import time
import math
import statistics
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from .kernel import Kernel
from .substrate import (embed_molecule, embed_text, s_distance, SCoord,
                         ternary_address)
from .vahera import execute_vahera
from .translator import IntentTranslator


# ═══════════════════════════════════════════════════════════════════

DATA_DIR = Path(__file__).parent / "data"
COMPOUNDS_PATH = DATA_DIR / "nist_compounds_extended.json"
RESULTS_PATH = Path(__file__).parent.parent / "data" / "integration_validation_results.json"


def load_compounds() -> dict:
    with open(COMPOUNDS_PATH) as f:
        return json.load(f)


def boot_with_n(compounds: dict, n: int, depth: int = 12) -> tuple[Kernel, dict]:
    """Boot a kernel with the first n compounds."""
    k = Kernel(depth=depth)
    subset = dict(list(compounds.items())[:n])
    for name, props in subset.items():
        coord = embed_molecule(name, props)
        k.allocate(coord, payload=props, metadata={"name": name,
                                                    "formula": props["formula"]})
    return k, subset


def run_query(kernel: Kernel, nist: dict, query: str,
              translator: IntentTranslator) -> dict:
    """Run one query through the stack, recording everything measurable."""
    t0 = time.perf_counter()
    vahera = translator.translate(query)
    t_trans = time.perf_counter() - t0

    # Snapshot kernel state before
    pre_stats = kernel.stats()

    t0 = time.perf_counter()
    try:
        ctx = execute_vahera(vahera, kernel=kernel, molecule_data=nist)
        err = None
    except Exception as e:
        ctx = None
        err = str(e)
    t_exec = time.perf_counter() - t0

    # Snapshot after
    post_stats = kernel.stats()

    # Subsystem call deltas
    pve_calls = post_stats["pve"]["verified"] - pre_stats["pve"]["verified"]
    pve_rejects = post_stats["pve"]["rejected"] - pre_stats["pve"]["rejected"]
    tem_samples = post_stats["tem"]["samples"] - pre_stats["tem"]["samples"]

    # Find match
    match = None
    q_coord = None
    d_match = None
    if ctx and ctx.targets:
        q_coord = next(iter(ctx.targets.values()))
        nearest = kernel.find_nearest(q_coord, k=1)
        if nearest:
            _, obj, dist = nearest[0]
            match = obj.metadata.get("name")
            d_match = dist

    return {
        "query": query,
        "vahera": vahera,
        "vahera_lines": len([l for l in vahera.split("\n") if l.strip()]),
        "t_translate_ms": t_trans * 1000,
        "t_execute_ms": t_exec * 1000,
        "t_total_ms": (t_trans + t_exec) * 1000,
        "match": match,
        "d_match": d_match,
        "query_coord": q_coord.as_tuple() if q_coord else None,
        "pve_calls": pve_calls,
        "pve_rejects": pve_rejects,
        "tem_samples": tem_samples,
        "error": err,
    }


# ═══════════════════════════════════════════════════════════════════

def experiment_1_latency():
    """Measure end-to-end latency across many queries."""
    print("=" * 60)
    print("EXP 1: End-to-End Latency")
    print("=" * 60)

    compounds = load_compounds()
    kernel, nist = boot_with_n(compounds, 40)
    translator = IntentTranslator()

    queries = [
        "what is the boiling point of ethanol",
        "boiling point of benzene",
        "density of water",
        "melting point of aspirin",
        "what is caffeine",
        "what is ibuprofen",
        "what is methanol",
        "molecular weight of glucose",
        "boiling point of hexane",
        "what is paracetamol",
        "density of acetone",
        "what is toluene",
        "melting point of sucrose",
        "what is naphthalene",
        "what is chloroform",
        "boiling point of nicotine",
        "what is pentane",
        "what is urea",
        "what is glycerol",
        "what is phenol",
    ]

    results = []
    for q in queries:
        r = run_query(kernel, nist, q, translator)
        results.append(r)
        print(f"  {q[:35]:35s}  t={r['t_total_ms']:.2f}ms  match={r['match']}")

    t_trans = [r["t_translate_ms"] for r in results]
    t_exec = [r["t_execute_ms"] for r in results]
    t_total = [r["t_total_ms"] for r in results]
    return {
        "queries": results,
        "n_queries": len(results),
        "translate_ms_mean": statistics.mean(t_trans),
        "translate_ms_median": statistics.median(t_trans),
        "execute_ms_mean": statistics.mean(t_exec),
        "execute_ms_median": statistics.median(t_exec),
        "total_ms_mean": statistics.mean(t_total),
        "total_ms_median": statistics.median(t_total),
        "total_ms_p95": sorted(t_total)[int(len(t_total) * 0.95)],
    }


def experiment_2_accuracy():
    """Accuracy on property-retrieval queries."""
    print("\n" + "=" * 60)
    print("EXP 2: Accuracy")
    print("=" * 60)

    compounds = load_compounds()
    kernel, nist = boot_with_n(compounds, 40)
    translator = IntentTranslator()

    # query -> expected compound
    test_cases = {
        "boiling point of ethanol": "ethanol",
        "boiling point of benzene": "benzene",
        "boiling point of hexane": "hexane",
        "boiling point of toluene": "toluene",
        "density of water": "water",
        "density of acetone": "acetone",
        "melting point of aspirin": "aspirin",
        "melting point of caffeine": "caffeine",
        "what is methanol": "methanol",
        "what is caffeine": "caffeine",
        "what is ibuprofen": "ibuprofen",
        "what is glucose": "glucose",
        "what is nicotine": "nicotine",
        "what is paracetamol": "paracetamol",
        "what is naphthalene": "naphthalene",
        "what is morphine": "morphine",
        "what is aniline": "aniline",
        "what is phenol": "phenol",
    }

    results = []
    correct_matches = []
    incorrect_matches = []

    for q, expected in test_cases.items():
        r = run_query(kernel, nist, q, translator)
        ok = r["match"] == expected
        r["expected"] = expected
        r["correct"] = ok
        results.append(r)
        if ok:
            correct_matches.append(r["d_match"])
        else:
            incorrect_matches.append(r["d_match"])
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {q[:35]:35s}  exp={expected}  got={r['match']}  "
              f"d={r['d_match']:.4f}" if r['d_match'] is not None else "d=None")

    n = len(results)
    correct = sum(1 for r in results if r["correct"])

    return {
        "queries": results,
        "n_queries": n,
        "n_correct": correct,
        "accuracy": correct / n,
        "mean_d_correct": statistics.mean(correct_matches) if correct_matches else 0,
        "mean_d_incorrect": statistics.mean(incorrect_matches) if incorrect_matches else 0,
    }


def experiment_3_dispatch():
    """Record dispatch activity per query type."""
    print("\n" + "=" * 60)
    print("EXP 3: Dispatch Protocol Activity")
    print("=" * 60)

    compounds = load_compounds()
    kernel, nist = boot_with_n(compounds, 40)
    translator = IntentTranslator()

    query_types = {
        "what is X": [
            "what is ethanol", "what is benzene", "what is caffeine",
            "what is aspirin", "what is water",
        ],
        "boiling point": [
            "boiling point of ethanol", "boiling point of benzene",
            "boiling point of hexane", "boiling point of water",
            "boiling point of aspirin",
        ],
        "similarity": [
            "find compounds similar to aspirin",
            "find compounds similar to caffeine",
            "find compounds similar to ethanol",
            "find compounds similar to benzene",
            "find compounds similar to glucose",
        ],
    }

    by_type = {}
    for qtype, queries in query_types.items():
        runs = []
        for q in queries:
            r = run_query(kernel, nist, q, translator)
            runs.append(r)
        by_type[qtype] = {
            "n": len(runs),
            "pve_calls_mean": statistics.mean(r["pve_calls"] for r in runs),
            "tem_samples_mean": statistics.mean(r["tem_samples"] for r in runs),
            "vahera_lines_mean": statistics.mean(r["vahera_lines"] for r in runs),
            "t_total_mean": statistics.mean(r["t_total_ms"] for r in runs),
        }
        print(f"  {qtype:20s} pve={by_type[qtype]['pve_calls_mean']:.1f}  "
              f"tem={by_type[qtype]['tem_samples_mean']:.1f}  "
              f"lines={by_type[qtype]['vahera_lines_mean']:.1f}  "
              f"t={by_type[qtype]['t_total_mean']:.2f}ms")

    return {
        "by_query_type": by_type,
        "final_kernel_stats": kernel.stats(),
    }


def experiment_4_scaling():
    """Latency as a function of number of stored compounds."""
    print("\n" + "=" * 60)
    print("EXP 4: Scaling")
    print("=" * 60)

    compounds = load_compounds()
    translator = IntentTranslator()
    test_query = "what is ethanol"

    sizes = [5, 10, 20, 40]
    per_size = []
    for n in sizes:
        kernel, nist = boot_with_n(compounds, n)
        # Warm up
        run_query(kernel, nist, test_query, translator)
        # Measure
        runs = []
        for _ in range(10):
            r = run_query(kernel, nist, test_query, translator)
            runs.append(r)
        mean_t = statistics.mean(r["t_total_ms"] for r in runs)
        mean_exec = statistics.mean(r["t_execute_ms"] for r in runs)
        per_size.append({
            "n_compounds": n,
            "mean_t_total_ms": mean_t,
            "mean_t_execute_ms": mean_exec,
            "queries_per_sec": 1000 / mean_t if mean_t > 0 else 0,
        })
        print(f"  n={n:3d}  t_total={mean_t:.3f}ms  t_exec={mean_exec:.3f}ms  "
              f"qps={per_size[-1]['queries_per_sec']:.0f}")

    return {"scaling": per_size}


def experiment_5_empty_dictionary():
    """The empty-dictionary principle: coord-proximity of matches."""
    print("\n" + "=" * 60)
    print("EXP 5: Empty Dictionary Principle")
    print("=" * 60)

    compounds = load_compounds()
    kernel, nist = boot_with_n(compounds, 40)
    translator = IntentTranslator()

    # For each stored compound, ask about it and measure how close the
    # reconstructed query coord is to the original compound coord.
    # If near 0, we've recovered the coord without the name table.
    results = []
    for name in list(nist.keys())[:25]:
        q = f"what is {name}"
        r = run_query(kernel, nist, q, translator)
        original_coord = embed_molecule(name, nist[name])
        # The query coord and the stored compound coord should match
        # if the empty-dictionary principle holds
        if r["query_coord"]:
            q_sc = SCoord(*r["query_coord"])
            d_to_original = s_distance(q_sc, original_coord)
        else:
            d_to_original = None

        results.append({
            "name": name,
            "query": q,
            "original_coord": original_coord.as_tuple(),
            "query_coord": r["query_coord"],
            "d_query_to_original": d_to_original,
            "match": r["match"],
            "correct": r["match"] == name,
        })
        status = "PASS" if r["match"] == name else "FAIL"
        d_str = f"{d_to_original:.4f}" if d_to_original is not None else "N/A"
        print(f"  [{status}] {name:20s}  d_to_original={d_str}")

    valid = [r for r in results if r["d_query_to_original"] is not None]
    correct = [r for r in valid if r["correct"]]

    return {
        "samples": results,
        "n_samples": len(valid),
        "n_correct": len(correct),
        "recovery_rate": len(correct) / max(len(valid), 1),
        "mean_d_to_original": statistics.mean(
            r["d_query_to_original"] for r in valid) if valid else 0,
        "max_d_to_original": max(
            (r["d_query_to_original"] for r in valid), default=0),
    }


def experiment_6_robustness():
    """Layer robustness under edge cases."""
    print("\n" + "=" * 60)
    print("EXP 6: Layer Robustness")
    print("=" * 60)

    compounds = load_compounds()
    kernel, nist = boot_with_n(compounds, 40)
    translator = IntentTranslator()

    edge_cases = [
        ("empty", ""),
        ("garbage", "asdf xyz qwerty"),
        ("unknown compound", "what is unobtainium"),
        ("short", "hi"),
        ("long", "what is " + "very " * 20 + "long query"),
        ("normal", "what is ethanol"),
    ]

    results = []
    for label, q in edge_cases:
        try:
            r = run_query(kernel, nist, q, translator)
            results.append({
                "label": label,
                "query": q,
                "completed": True,
                "t_total_ms": r["t_total_ms"],
                "match": r["match"],
                "pve_rejects": r["pve_rejects"],
                "error": r["error"],
            })
        except Exception as e:
            results.append({
                "label": label,
                "query": q,
                "completed": False,
                "error": str(e),
            })
        print(f"  {label:20s}  completed={results[-1]['completed']}  "
              f"error={results[-1].get('error', None)}")

    n_completed = sum(1 for r in results if r["completed"])
    return {
        "edge_cases": results,
        "n_tested": len(results),
        "n_completed": n_completed,
        "completion_rate": n_completed / len(results),
    }


# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  BUHERA OS — Integration Validation")
    print("=" * 70)

    results = {
        "experiment_1_latency": experiment_1_latency(),
        "experiment_2_accuracy": experiment_2_accuracy(),
        "experiment_3_dispatch": experiment_3_dispatch(),
        "experiment_4_scaling": experiment_4_scaling(),
        "experiment_5_empty_dictionary": experiment_5_empty_dictionary(),
        "experiment_6_robustness": experiment_6_robustness(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    e1 = results["experiment_1_latency"]
    e2 = results["experiment_2_accuracy"]
    e4 = results["experiment_4_scaling"]
    e5 = results["experiment_5_empty_dictionary"]
    e6 = results["experiment_6_robustness"]

    summary = {
        "n_queries_tested": e1["n_queries"] + e2["n_queries"] + e5["n_samples"],
        "accuracy": e2["accuracy"],
        "mean_total_latency_ms": e1["total_ms_mean"],
        "p95_total_latency_ms": e1["total_ms_p95"],
        "scaling_flat": abs(e4["scaling"][0]["mean_t_total_ms"] -
                            e4["scaling"][-1]["mean_t_total_ms"]) /
                        max(e4["scaling"][0]["mean_t_total_ms"], 1e-9) < 2.0,
        "empty_dict_recovery": e5["recovery_rate"],
        "mean_d_to_original": e5["mean_d_to_original"],
        "robustness_completion_rate": e6["completion_rate"],
    }
    results["summary"] = summary

    for k, v in summary.items():
        print(f"  {k:40s}  {v}")

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
