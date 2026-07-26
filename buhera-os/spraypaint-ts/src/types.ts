// ─────────────────────────────────────────────────────────────────────────────
// spraypaint runtime types
//
// These interfaces are typed directly against the REAL `spraypaint … --json`
// output, NOT against any DSL. The binary is the source of truth; every field
// here appears verbatim in the JSON emitted by the installed `spraypaint` crate.
//
// Captured from `spraypaint <cmd> --json` v0.1.0. If the binary schema changes,
// these types change with it — there is no second grammar to keep in sync.
// ─────────────────────────────────────────────────────────────────────────────

// ── The canonical artifact ──────────────────────────────────────────────────
//
// Every input surface (a typed prompt, a hand-edited field, a chart gesture)
// converges on ONE runnable thing: an AskQuery. This is the analogue of the
// mock web app's ".grf script" — except it is exactly what the binary accepts
// as `spraypaint ask <query> -k <budget> [--scenes a,b] [--flat]`.
//
// Crossfilter gestures do not rewrite prose or a DSL; they produce a QueryDiff
// against this object (see crossfilter.ts). That is what makes the loop
// bidirectional: charts are editable views of the AskQuery.
export interface AskQuery {
  /** Free-text search intent — the positional <QUERY> argument. */
  query: string;
  /** `-k / --budget`: total passages allocated across all scenes by water-filling. */
  budget: number;
  /**
   * `--scenes a,b`: restrict search to these scene names. `null`/empty ⇒ all
   * scenes (no restriction). A scene name is a top-level dir or a scenes.toml key.
   */
  scenes: string[] | null;
  /** `--flat`: emit one global ranked order instead of grouping by scene. */
  flat: boolean;
}

/** A sensible default query object for a fresh session. */
export const DEFAULT_QUERY: AskQuery = {
  query: "",
  budget: 12,
  scenes: null,
  flat: false,
};

// ── `spraypaint ask --json` ──────────────────────────────────────────────────

/** One ranked passage. The snippet is re-read from disk at query time (Inv 3). */
export interface AskHit {
  path: string;
  scene: string;
  score: number;
  start_line: number;
  end_line: number;
  snippet: string;
}

/**
 * Per-scene water-filling outcome. `allocated` is how many slots this scene won
 * at the clearing price; `available` is how many passages it could have offered.
 * A scene with `available > 0` but `allocated === 0` was priced out below p*.
 */
export interface SceneAllocation {
  scene: string;
  allocated: number;
  available: number;
}

/** Full result of one `ask`. Mirrors the binary's JSON object exactly. */
export interface AskResult {
  results: AskHit[];
  allocation: SceneAllocation[];
  /** `-k` echoed back. */
  budget: number;
  /** Clearing price p* from the water-filling bisection. */
  price: number;
  /** Tokenised query terms actually searched. */
  query_terms: string[];
  /** Inv 2: monotone committed count AFTER this ask incremented it. */
  committed_count: number;
  /** Inv 1: blake3 self-graph fingerprint, e.g. "b3:e32d98…". */
  identity_fingerprint: string;
}

// ── `spraypaint identity --json` (Inv 1) ─────────────────────────────────────

export interface Identity {
  /** blake3 fingerprint of the canonicalised self-graph, prefixed "b3:". */
  fingerprint: string;
  /** χ — Stoer–Wagner min-cut of the self-graph. Must stay ≥ floor. */
  char_invariant: number;
  /** Structural floor; χ ≥ floor > 0 is the identity-conservation guarantee. */
  floor: number;
  n_vertices: number;
  n_edges: number;
}

// ── `spraypaint count --json` (Inv 2) ────────────────────────────────────────

export interface CountResult {
  /** Never-resetting committed-ask count. */
  committed_count: number;
}

// ── `spraypaint scenes --json` ───────────────────────────────────────────────

export interface SceneInfo {
  name: string;
  documents: number;
  passages: number;
}

// ── `spraypaint verify --json` (all four invariants) ─────────────────────────

export interface InvariantCheck {
  pass: boolean;
  detail: string;
}

export interface VerifyResult {
  inv1_identity: InvariantCheck;
  inv2_count: InvariantCheck;
  inv3_search_not_fetch: InvariantCheck;
  inv4_phases: InvariantCheck;
  /** Overall — true iff all four passed. Nonzero binary exit on false. */
  pass: boolean;
}

// ── Query serialisation ──────────────────────────────────────────────────────

/**
 * Turn an AskQuery into the argv the binary expects (after the `ask`
 * subcommand). Kept here so every caller — client, tests, UI preview — renders
 * the exact same command string.
 */
export function queryToArgs(q: AskQuery): string[] {
  const args: string[] = [q.query, "--json", "-k", String(q.budget)];
  if (q.scenes && q.scenes.length > 0) {
    args.push("--scenes", q.scenes.join(","));
  }
  if (q.flat) args.push("--flat");
  return args;
}

/** Human-readable command line for display in a UI (never executed as a string). */
export function queryToDisplay(q: AskQuery): string {
  return ["spraypaint", "ask", JSON.stringify(q.query), ...queryToArgs(q).slice(1)].join(" ");
}
