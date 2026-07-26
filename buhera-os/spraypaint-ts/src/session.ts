// ─────────────────────────────────────────────────────────────────────────────
// SpraypaintSession — the closed loop as a single object.
//
// Holds the current AskQuery, the last AskResult, the undo history, and a
// `stale` flag. A gesture (or prompt/editor edit) applies a QueryDiff: the query
// advances, history records it, and results are marked stale until the next run.
// `run()` executes the (now-current) query against the real binary and clears
// stale. This is the headless core a UI wires its charts to.
//
//     applyGesture(diff)   → query changes, stale=true, history.push
//     run()                → ask(query) via client, stale=false, result stored
// ─────────────────────────────────────────────────────────────────────────────

import type { AskQuery, AskResult } from "./types.js";
import { DEFAULT_QUERY } from "./types.js";
import type { SpraypaintClient } from "./client.js";
import { applyDiff, type QueryDiff, type QuerySource } from "./crossfilter.js";
import { QueryHistory, type QuerySnapshot } from "./undo.js";

export interface SessionState {
  query: AskQuery;
  result: AskResult | null;
  /** True when the query changed since the displayed result was produced. */
  stale: boolean;
}

/** Monotonic clock injected for testability (Date.now by default). */
export type Clock = () => number;

export class SpraypaintSession {
  private query: AskQuery;
  private result: AskResult | null = null;
  private stale = false;
  private readonly history = new QueryHistory();

  constructor(
    private readonly client: SpraypaintClient,
    initial: AskQuery = DEFAULT_QUERY,
    private readonly clock: Clock = () => Date.now(),
  ) {
    this.query = initial;
    this.history.push({
      query: initial,
      timestamp: this.clock(),
      source: "editor",
      description: "initial query",
    });
  }

  state(): SessionState {
    return { query: this.query, result: this.result, stale: this.stale };
  }

  /**
   * Apply a diff from any surface. Advances the query, marks results stale, and
   * records a peer snapshot on the unified history. Does NOT run — the UI decides
   * when to commit (e.g. explicit Run, or debounce), keeping construction and
   * commitment phases distinct.
   */
  applyGesture(diff: QueryDiff, source: QuerySource): AskQuery {
    this.query = applyDiff(this.query, diff);
    this.stale = this.result !== null;
    this.history.push({
      query: this.query,
      timestamp: this.clock(),
      source,
      description: diff.describe,
    });
    return this.query;
  }

  /** Execute the current query against the real binary. Clears stale. */
  async run(): Promise<AskResult> {
    const res = await this.client.ask(this.query);
    this.result = res;
    this.stale = false;
    return res;
  }

  /** Diagnostic run that does not increment the committed count (Inv 3). */
  async preview(): Promise<AskResult> {
    return this.client.dryRun(this.query);
  }

  // ── undo / redo restore query AND mark stale (charts detach until re-run) ──

  undo(): AskQuery | null {
    const snap = this.history.undo();
    if (!snap) return null;
    this.query = snap.query;
    this.stale = this.result !== null;
    return this.query;
  }

  redo(): AskQuery | null {
    const snap = this.history.redo();
    if (!snap) return null;
    this.query = snap.query;
    this.stale = this.result !== null;
    return this.query;
  }

  canUndo(): boolean {
    return this.history.canUndo();
  }
  canRedo(): boolean {
    return this.history.canRedo();
  }
  timeline(): QuerySnapshot[] {
    return this.history.history();
  }
}
