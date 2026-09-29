/**
 * Image lineage & runtime-health — the second per-repo lineage.
 *
 * A repo's χ (see `chi.ts`) is the lineage of its *sense* over time. This module
 * is the lineage of its *runtime health* over time: the record of exercising each
 * built image version and judging the resulting traces with wind-tunnel.
 *
 * The motivating fact (project_image_knowability): ordinary Docker's success
 * criterion is "it built" — strictly weaker than runtime-correctness. An image is
 * a mute artifact whose health is unknowable on the external machine *unless a
 * human happens to hit a bug*. Bloodhound owns the two missing tools — agents to
 * EXERCISE an image, wind-tunnel to JUDGE the {t,state[]} traces — so health
 * becomes proactively knowable, and each immutable image version carries a regime
 * map alongside it.
 *
 * This file is the pure, browser-side DATA MODEL for the read verbs
 * (`show lineage / regime / health`). It computes nothing effectful: building,
 * exercising and judging happen behind a backend that the browser interpreter does
 * not have. When no lineage has been recorded, the read verbs report that
 * honestly — they never fabricate a regime map.
 */

/**
 * The wind-tunnel regime map — a coherence *description*, not a pass/fail verdict.
 * Mirrors wind-tunnel's WT=(R_dyn, S_flat, H, D, δS). All fields optional because a
 * build-only repo yields "assembled in a clean container" but no dynamics to map.
 */
export interface RegimeMap {
  /** R_dyn — dynamical regime (e.g. "convergent", "oscillatory", "divergent"). */
  rDyn?: string;
  /** S_flat — flatness / spread of the state trajectory. */
  sFlat?: number;
  /** H — trajectory entropy. */
  h?: number;
  /** D — effective dimensionality of the explored state. */
  d?: number;
  /** δS — entropy change across the run (drift). */
  deltaS?: number;
}

/** How an image version was built (which level of the build ladder produced it). */
export type BuildLevel =
  | "devcontainer" // L1: the repo's own devcontainer image
  | "synthesized"  // L2: a Dockerfile synthesized because the repo lacked one
  | "composed";    // L3: multi-stage compose (only for env-compatible subsets)

/** The outcome of `docker build` — ground truth, never the model's opinion. */
export type BuildStatus = "built" | "failed" | "unbuilt";

/**
 * How thoroughly an image version's runtime health is KNOWN.
 * The whole point of the tool is to move versions up this ladder.
 */
export type HealthKnowledge =
  | "unknown"    // never exercised — health is the ordinary-Docker blind spot
  | "assembled"  // built in a clean container, but nothing runnable to exercise
  | "exercised"  // agents drove it, traces captured, not yet judged
  | "judged";    // wind-tunnel produced a regime map for it

/** One immutable image in a repo's lineage: repo:vN, with what is known about it. */
export interface ImageVersion {
  /** Monotonic lineage index (v1, v2, …). */
  version: number;
  /** The git commit the Dockerfile / build context was taken from. */
  commit?: string;
  /** Content-addressed image digest, once built (immutable identity). */
  digest?: string;
  /** Which build ladder rung produced it. */
  level: BuildLevel;
  /** docker build result — the ground-truth gate. */
  build: BuildStatus;
  /** How far up the knowability ladder this version has been carried. */
  knowledge: HealthKnowledge;
  /** The regime map, present only once `knowledge === "judged"`. */
  regime?: RegimeMap;
  /** One-line human note (e.g. why a build failed, what was exercised). */
  note?: string;
}

/** A repo's full image lineage — v1 → v2 → v3, newest last. */
export interface ImageLineage {
  versions: ImageVersion[];
}

// ── read helpers (pure) ─────────────────────────────────────────────────────

/** The newest version in a lineage, or undefined for an empty lineage. */
export function latest(lin: ImageLineage | undefined): ImageVersion | undefined {
  if (!lin || lin.versions.length === 0) return undefined;
  return lin.versions[lin.versions.length - 1];
}

/** Human-readable one-liner for a single image version. */
export function versionLine(v: ImageVersion): string {
  const id = v.digest ? v.digest.slice(0, 12) : v.commit ? `@${v.commit.slice(0, 7)}` : "—";
  const tail =
    v.knowledge === "judged" && v.regime
      ? `regime ${regimeSummary(v.regime)}`
      : v.knowledge;
  const note = v.note ? `  (${v.note})` : "";
  return `v${v.version}  [${v.level}]  build:${v.build}  ${id}  ${tail}${note}`;
}

/** Compact one-line rendering of a regime map (only the fields that are present). */
export function regimeSummary(r: RegimeMap): string {
  const parts: string[] = [];
  if (r.rDyn) parts.push(r.rDyn);
  if (r.sFlat !== undefined) parts.push(`S_flat ${r.sFlat.toFixed(2)}`);
  if (r.h !== undefined) parts.push(`H ${r.h.toFixed(2)}`);
  if (r.d !== undefined) parts.push(`D ${r.d.toFixed(1)}`);
  if (r.deltaS !== undefined) parts.push(`δS ${r.deltaS.toFixed(3)}`);
  return parts.length ? parts.join(", ") : "(no dynamics — build-only)";
}

/** Full multi-line rendering of a regime map for `show regime`. */
export function regimeLines(v: ImageVersion): string[] {
  if (v.knowledge !== "judged" || !v.regime) {
    // Honest: no fabricated numbers. Say exactly how far knowability got.
    switch (v.knowledge) {
      case "unknown":
        return [`v${v.version}: not exercised — runtime health is unknown (ordinary-Docker blind spot)`];
      case "assembled":
        return [`v${v.version}: assembled in a clean container, but nothing runnable to exercise — no regime map`];
      case "exercised":
        return [`v${v.version}: exercised (traces captured) but not yet judged by wind-tunnel`];
      default:
        return [`v${v.version}: no regime map`];
    }
  }
  const r = v.regime;
  const rows: string[] = [`v${v.version} regime map:`];
  if (r.rDyn) rows.push(`  R_dyn  = ${r.rDyn}`);
  if (r.sFlat !== undefined) rows.push(`  S_flat = ${r.sFlat.toFixed(3)}`);
  if (r.h !== undefined) rows.push(`  H      = ${r.h.toFixed(3)}`);
  if (r.d !== undefined) rows.push(`  D      = ${r.d.toFixed(3)}`);
  if (r.deltaS !== undefined) rows.push(`  δS     = ${r.deltaS.toFixed(4)}`);
  return rows;
}
