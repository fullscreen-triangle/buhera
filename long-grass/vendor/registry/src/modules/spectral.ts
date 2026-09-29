/* ============================================================================
 * spectral — spectral sequence homology and matched-filter motif scanning
 * (specification specs/spectral.md). Wraps the pure numerical core of
 * gospel's vivid-symbolism (alphabets, embedding, fft, matched_filter),
 * vendored.
 *
 * These are the operations a synopsis program names — `project … by
 * spectral(coeffs)`, `compare … by shader(cosine)`, `compare … by
 * xcorr(normalised)`, `detect peaks` — but this module is NOT a synopsis
 * evaluator: it computes on sequences a caller supplies, with parameters the
 * caller states. The synopsis module checks programs; neither runs them.
 *
 * The engine's timing figures (performance.now) are dropped: acts must be
 * deterministic.
 * ========================================================================== */

import type { ActResult, Instruction, Json, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";

/** The subset of the vendored vivid-symbolism core this adapter uses. */
export interface SpectralEngine {
  spectralEmbedding(seq: string, K: number, kind: "dna" | "protein"): Float32Array;
  shaderKernelScan(dbFlat: Float32Array, dim: number, query: Float32Array, topK: number): { scores: Float32Array; topK: Array<{ index: number; score: number }> };
  prepareTarget(target: string, maxQueryLen: number): unknown;
  matchedFilterScan(query: string, ctx: unknown): { scores: Float64Array; lagCount: number };
  backgroundStats(scores: Float64Array, exclude?: Array<[number, number]>): { mean: number; std: number };
  findPeaks(scores: Float64Array, minScore: number, minSeparation: number, maxPeaks?: number): Array<{ index: number; score: number }>;
}

export const SPECTRAL_ID = "spectral";
const MAX_SEQ = 2_000_000;
const MAX_DB = 20_000;

/** Two short protein fragments against a small panel — an openly synthetic demo. */
const DEMO = {
  query: "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
  database: [
    { name: "near-identical", sequence: "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ" },
    { name: "one substitution", sequence: "MKTAYIAKQRQISFVKSHFSRQLEERLGLIDVQ" },
    { name: "shuffled", sequence: "QVEILGLREELQRSFHSKVFSIQRQKAIYATKM" },
    { name: "unrelated", sequence: "GSSGSSGWWGWWPPPPGGGAAAACCCCDDDDEEEE" },
  ],
};

function str(x: Json | undefined): string | null {
  return typeof x === "string" && x.length > 0 ? x : null;
}
function int(x: Json | undefined, fallback: number): number | null {
  if (x === undefined || x === null) return fallback;
  return typeof x === "number" && Number.isInteger(x) ? x : null;
}
function real(x: Json | undefined, fallback: number): number | null {
  if (x === undefined || x === null) return fallback;
  return typeof x === "number" && Number.isFinite(x) ? x : null;
}

export function makeSpectralModule(engine: SpectralEngine): Module {
  const expected = '"demo" or { kind: "embed" | "homology" | "motif", … }';

  function homology(query: string, database: Array<{ name: string; sequence: string }>, alphabet: "dna" | "protein", K: number, top: number): ActResult {
    const q = engine.spectralEmbedding(query, K, alphabet);
    const dim = q.length;
    const flat = new Float32Array(dim * database.length);
    database.forEach((d, i) => flat.set(engine.spectralEmbedding(d.sequence, K, alphabet), i * dim));
    const { topK } = engine.shaderKernelScan(flat, dim, q, top);
    return done(
      {
        kind: "spectral_ranked",
        summary: `spectral: ${database.length} sequence(s) ranked by cosine of ${K}-coefficient embeddings`,
        alphabet,
        coeffs: K,
        dim,
        ranked: topK.map((t) => ({ name: database[t.index]!.name, index: t.index, cosine: t.score })),
      },
      0,
    );
  }

  return {
    id: SPECTRAL_ID,
    describe: () => ({
      id: SPECTRAL_ID,
      description:
        "Spectral homology — embed sequences by their low-frequency DFT magnitudes per channel (L2-normalised), rank a " +
        "database by cosine (the shader kernel), and scan a DNA target for a motif by FFT matched filtering with " +
        "background z-scores and peak picking. The numerical operations synopsis programs name; not a synopsis evaluator.",
      instructions: [
        'dispatch("spectral", "demo")',
        'dispatch("spectral", { kind: "embed", sequence, alphabet: "protein", coeffs: 8 })',
        'dispatch("spectral", { kind: "homology", query, database: [{ name, sequence }], alphabet: "protein", coeffs: 8, top: 10 })',
        'dispatch("spectral", { kind: "motif", query, target, z: 4, min_distance: 30, min_score: 0.35 })',
      ],
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      try {
        if (instruction === "demo") return homology(DEMO.query, DEMO.database, "protein", 8, 4);
        if (!instruction || typeof instruction !== "object" || Array.isArray(instruction)) return invalid(SPECTRAL_ID, expected);
        const kind = instruction["kind"];
        const alphabet = instruction["alphabet"] ?? "protein";
        if (alphabet !== "dna" && alphabet !== "protein") return invalid(SPECTRAL_ID, 'alphabet: "dna" | "protein"');
        const K = int(instruction["coeffs"], 8);
        if (K == null || K < 1 || K > 256) return invalid(SPECTRAL_ID, "coeffs: integer 1..256");

        if (kind === "embed") {
          const seq = str(instruction["sequence"]);
          if (!seq || seq.length > MAX_SEQ) return invalid(SPECTRAL_ID, '{ kind: "embed", sequence, alphabet?, coeffs? }');
          const v = engine.spectralEmbedding(seq, K, alphabet);
          return done({ kind: "spectral_embedding", alphabet, coeffs: K, dim: v.length, vector: [...v] }, 0);
        }
        if (kind === "homology") {
          const query = str(instruction["query"]);
          const db = instruction["database"];
          const top = int(instruction["top"], 10);
          if (!query || !Array.isArray(db) || db.length === 0 || db.length > MAX_DB || top == null || top < 1) {
            return invalid(SPECTRAL_ID, `{ kind: "homology", query, database: [1..${MAX_DB} { name, sequence }], top? }`);
          }
          const database: Array<{ name: string; sequence: string }> = [];
          for (const d of db) {
            if (!d || typeof d !== "object" || Array.isArray(d) || !str(d["sequence"])) return invalid(SPECTRAL_ID, "database entries are { name, sequence }");
            database.push({ name: typeof d["name"] === "string" ? d["name"] : `#${database.length}`, sequence: d["sequence"] as string });
          }
          return homology(query, database, alphabet, K, top);
        }
        if (kind === "motif") {
          const query = str(instruction["query"]);
          const target = str(instruction["target"]);
          const z = real(instruction["z"], NaN);
          const minDistance = int(instruction["min_distance"], -1);
          const minScore = real(instruction["min_score"], NaN);
          // Like synopsis's `detect peaks`, every threshold must be stated: no defaults.
          if (!query || !target || target.length > MAX_SEQ || z == null || Number.isNaN(z) || minDistance == null || minDistance < 0 || minScore == null || Number.isNaN(minScore)) {
            return invalid(SPECTRAL_ID, '{ kind: "motif", query, target, z, min_distance, min_score } — every threshold stated');
          }
          if (query.length > target.length) return fail(["spectral: the query is longer than the target"], "query longer than target");
          const ctx = engine.prepareTarget(target, query.length);
          const { scores, lagCount } = engine.matchedFilterScan(query, ctx);
          const { mean, std } = engine.backgroundStats(scores);
          const zScores = scores.map((s) => (s - mean) / std);
          const peaks = engine
            .findPeaks(zScores, z, minDistance)
            .filter((p) => scores[p.index]! >= minScore)
            .map((p) => ({ offset: p.index, z: p.score, score: scores[p.index]! }));
          return done(
            {
              kind: "spectral_motif",
              summary: `spectral: ${peaks.length} hit(s) over ${lagCount} offset(s) at z ≥ ${z}, score ≥ ${minScore}`,
              lags: lagCount,
              background: { mean, std },
              thresholds: { z, min_distance: minDistance, min_score: minScore },
              hits: peaks,
            },
            0,
          );
        }
        return invalid(SPECTRAL_ID, expected);
      } catch (err) {
        return fail([`spectral: ${errorText(err)}`], errorText(err));
      }
    },

    outputCell: () => ({ kind: "spectral_cell" }),
  };
}
