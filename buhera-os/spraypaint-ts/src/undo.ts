// ─────────────────────────────────────────────────────────────────────────────
// Unified query-snapshot stack.
//
// Every semantic action — a typed prompt, a hand-edited field, a chart gesture —
// produces a snapshot of the canonical AskQuery. They are PEERS: the stack does
// not privilege manual edits over crossfilter gestures. This is what makes
// "manipulate the chart" and "edit the query" the same operation from undo's
// point of view, which is the whole reason the loop is bidirectional.
//
// Snapshot semantics (not command/diff): each entry holds a full AskQuery, so
// undo/redo is O(1) and never has to invert a diff. Branch-on-push discards the
// redo tail, matching editor conventions.
// ─────────────────────────────────────────────────────────────────────────────

import type { AskQuery } from "./types.js";
import type { QuerySource } from "./crossfilter.js";

export interface QuerySnapshot {
  query: AskQuery;
  timestamp: number;
  source: QuerySource;
  description: string;
}

export class QueryHistory {
  private stack: QuerySnapshot[] = [];
  private cursor = -1;
  private readonly maxSize: number;

  constructor(maxSize = 200) {
    this.maxSize = maxSize;
  }

  /** Push a new snapshot, discarding any redo tail (we branched). */
  push(entry: QuerySnapshot): void {
    this.stack = this.stack.slice(0, this.cursor + 1);
    this.stack.push(entry);
    if (this.stack.length > this.maxSize) this.stack.shift();
    this.cursor = this.stack.length - 1;
  }

  undo(): QuerySnapshot | null {
    if (this.cursor <= 0) return null;
    this.cursor--;
    return this.stack[this.cursor] ?? null;
  }

  redo(): QuerySnapshot | null {
    if (this.cursor >= this.stack.length - 1) return null;
    this.cursor++;
    return this.stack[this.cursor] ?? null;
  }

  current(): QuerySnapshot | null {
    return this.stack[this.cursor] ?? null;
  }

  /** History up to and including the cursor (oldest → current). */
  history(): QuerySnapshot[] {
    return this.stack.slice(0, this.cursor + 1);
  }

  canUndo(): boolean {
    return this.cursor > 0;
  }

  canRedo(): boolean {
    return this.cursor < this.stack.length - 1;
  }
}
