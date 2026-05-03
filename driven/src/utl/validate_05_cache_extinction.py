"""V5: Cache extinction — when a class becomes indistinguishable
(cache hit), tau drops structurally to 0, producing a TP jump."""
from __future__ import annotations

import numpy as np

from .common import SEED, banner, save_results


def validate():
    banner("UTL V5 — CACHE EXTINCTION")
    rng = np.random.default_rng(SEED)
    records = []
    speedups = []
    for trial in range(20):
        N = 4
        tau = rng.uniform(1.0, 5.0, (N, N))
        g = rng.uniform(0.3, 1.0, (N, N))
        g = (g + g.T) / 2
        tp_inv_uncached = (tau * g).sum() / (N * N)

        # Simulate cache hit: pair (0, 1) becomes indistinguishable, tau[0,1]=tau[1,0]=0
        tau_cached = tau.copy()
        tau_cached[0, 1] = 0.0
        tau_cached[1, 0] = 0.0
        tp_inv_cached = (tau_cached * g).sum() / (N * N)

        speedup = tp_inv_uncached / max(tp_inv_cached, 1e-12)
        speedups.append(speedup)
        records.append({
            "trial": trial,
            "tp_inv_uncached": tp_inv_uncached,
            "tp_inv_cached": tp_inv_cached,
            "speedup": speedup,
            "extinction_ok": tp_inv_cached < tp_inv_uncached,
        })

    summary = {
        "claim": "Cache hit => tau structurally 0 => TP jump",
        "n_trials": len(records),
        "all_speedup_positive": all(r["extinction_ok"] for r in records),
        "min_speedup": float(np.min(speedups)),
        "max_speedup": float(np.max(speedups)),
        "median_speedup": float(np.median(speedups)),
        "overall_pass": all(r["extinction_ok"] for r in records),
    }
    print(f"  N trials: {len(records)}  speedup range: [{summary['min_speedup']:.3f}, {summary['max_speedup']:.3f}]")
    out = save_results("05_cache_extinction", {"summary": summary, "records": records})
    print(f"  Saved: {out}\n  PASS: {summary['overall_pass']}")
    return summary


if __name__ == "__main__":
    validate()
