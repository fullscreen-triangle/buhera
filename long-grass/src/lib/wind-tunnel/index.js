/* ============================================================================
 * wind-tunnel: emergent-behavior testing, ported as a TS analysis pass.
 *
 * Named after fullscreen-triangle/wind-tunnel (a Rust workspace testing
 * whole call-*cycles* for emergent behavior — semantic entropy, holonomy,
 * a Kuramoto-borrowed order parameter — rather than individual units in
 * isolation, the way unit/property tests do).
 *
 * This module ports the *idea*, not the binary: for interceptor's generated
 * code there is no existing indexed repo to run the real `wt` CLI against —
 * the "cycle" here is one program run repeatedly. What we can measure without
 * the full Rust toolchain:
 *
 *   - holonomy   : how much a run's output deviates from the reference run
 *                  (the first run is treated as the declared "spec"; later
 *                  runs are checked against it — a literal port of "deviation
 *                  of a cycle's actual behavior from declared spec")
 *   - order parameter (R) : fraction of runs whose output exactly matches the
 *                  reference, a simplified stand-in for the Kuramoto-style
 *                  ensemble coherence measure (1.0 = perfectly phase-locked /
 *                  fully deterministic; 0.0 = turbulent / no two runs agree)
 *   - regime     : R mapped onto wind-tunnel's named regimes
 *
 * This is intentionally a simplification — no dependency graph, no ablation
 * over real call-sites. It answers the one question interceptor needs: "does
 * this generated program behave the same way every time it runs?"
 * ========================================================================== */

// Regime thresholds, following wind-tunnel's 5-regime naming.
const REGIMES = [
  { min: 0.95, name: "Phase-locked", note: "all runs produced identical output" },
  { min: 0.75, name: "Synchronized", note: "runs mostly agree; minor deviation" },
  { min: 0.45, name: "Partially-locked", note: "runs split between outcomes" },
  { min: 0.15, name: "Desynchronized", note: "most runs disagree" },
  { min: 0.0, name: "Turbulent", note: "no consistent output across runs" },
];

export function classifyRegime(orderParameter) {
  for (const r of REGIMES) {
    if (orderParameter >= r.min) return { name: r.name, note: r.note };
  }
  return REGIMES[REGIMES.length - 1];
}

/**
 * Line-level diff between two output strings. Returns the count of differing
 * lines and a small sample of the first divergence, without pulling in a
 * diff library — the outputs here are captured stdout, not source files, and
 * a line-set comparison is enough to detect nondeterminism.
 */
export function diffOutputs(reference, candidate) {
  const a = reference.split("\n");
  const b = candidate.split("\n");
  const len = Math.max(a.length, b.length);
  let differing = 0;
  let firstDivergence = null;
  for (let i = 0; i < len; i++) {
    if ((a[i] ?? "") !== (b[i] ?? "")) {
      differing++;
      if (firstDivergence == null) {
        firstDivergence = { line: i + 1, expected: a[i] ?? "(missing)", actual: b[i] ?? "(missing)" };
      }
    }
  }
  return { differing_lines: differing, total_lines: len, first_divergence: firstDivergence };
}

/**
 * Compute holonomy + order parameter across a set of run results (as
 * returned by exec-sandbox's runCode / interceptor-run.js's output_delta).
 * The first successful run is the reference ("declared spec"); every other
 * run is compared against it.
 *
 * @param {Array<{ ok: boolean, stdout: string, stderr: string, exit_code: number|null }>} runs
 */
export function computeStability(runs) {
  if (!Array.isArray(runs) || runs.length === 0) {
    return {
      runs: 0,
      order_parameter: 0,
      regime: classifyRegime(0),
      reference_index: null,
      per_run: [],
      crash_count: 0,
    };
  }

  const referenceIndex = runs.findIndex((r) => r.ok);
  const reference = referenceIndex >= 0 ? runs[referenceIndex] : runs[0];
  const crashCount = runs.filter((r) => !r.ok).length;

  const perRun = runs.map((r, i) => {
    if (i === referenceIndex) {
      return { index: i, matches_reference: true, holonomy: 0, ok: r.ok, diff: null };
    }
    const combinedRef = `${reference.stdout || ""}${reference.stderr || ""}`;
    const combinedRun = `${r.stdout || ""}${r.stderr || ""}`;
    const matches = combinedRef === combinedRun;
    const diff = matches ? null : diffOutputs(combinedRef, combinedRun);
    // Holonomy: fraction of lines that deviate from the reference (0 = no
    // deviation, matching wind-tunnel's "deviation from declared spec").
    const holonomy = diff ? diff.differing_lines / Math.max(1, diff.total_lines) : 0;
    return { index: i, matches_reference: matches, holonomy, ok: r.ok, diff };
  });

  const matchCount = perRun.filter((p) => p.matches_reference).length;
  const orderParameter = matchCount / runs.length;

  return {
    runs: runs.length,
    order_parameter: orderParameter,
    regime: classifyRegime(orderParameter),
    reference_index: referenceIndex,
    per_run: perRun,
    crash_count: crashCount,
  };
}
