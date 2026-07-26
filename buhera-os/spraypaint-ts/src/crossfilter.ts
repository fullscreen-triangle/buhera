// ─────────────────────────────────────────────────────────────────────────────
// Crossfilter → query propagation.  THE mechanism.
//
// Charts are not read-only outputs; they are editable views of the AskQuery.
// A gesture on a chart (drag the clearing-price band, click a scene bar) is
// inverted into a QueryDiff — a structured, typed change to the canonical
// AskQuery object. applyDiff produces the next query; re-running it redraws the
// charts. That closes the loop:
//
//     AskQuery ──ask──▶ AskResult ──draw──▶ charts
//        ▲                                     │
//        └────── applyDiff ◀── invert*() ◀─────┘  (a gesture)
//
// WHY BIDIRECTIONAL (the user's two reasons, made concrete):
//
//  1. The AskQuery is the only runnable artifact. A typed prompt, a hand-edited
//     field, and a chart gesture are three peer surfaces that must all write to
//     it. If charts were read-only, only users fluent enough to hand-edit
//     `-k`/`--scenes` could steer a running search — defeating the accessibility
//     the prompt surface promises. So gestures must produce query edits too.
//
//  2. The charts show quantities with no pre-existing intuition (clearing price
//     p*, χ, committed count). Watching a gesture rewrite `-k 12` → `-k 20`
//     teaches — by operation — that "this p* line I'm dragging IS the budget in
//     the query." The diff is the Rosetta Stone between an unfamiliar chart and
//     the query grammar. You learn the mapping by operating it.
//
// Unlike the mock (which string-replaced `.grf` DSL text and could emit
// malformed source), these inversions edit a typed object, so an invalid query
// is unrepresentable.
// ─────────────────────────────────────────────────────────────────────────────

import type { AskQuery, AskResult, SceneAllocation } from "./types.js";

/** Where a query edit originated. All three are peers on one undo stack. */
export type QuerySource = "prompt" | "editor" | "crossfilter";

/**
 * A structured change to an AskQuery. `patch` is a partial overlay; `describe`
 * is human-readable text for the undo history / status bar. `gesture` records
 * which chart interaction produced it (empty for prompt/editor edits).
 */
export interface QueryDiff {
  patch: Partial<AskQuery>;
  describe: string;
  gesture: string;
}

/** Apply a diff to a query, returning a new query (never mutates the input). */
export function applyDiff(query: AskQuery, diff: QueryDiff): AskQuery {
  return { ...query, ...diff.patch };
}

// ── Scene-bar toggle (SceneAllocationTab) ────────────────────────────────────
//
// Clicking a scene bar toggles whether that scene participates. Maps to
// `--scenes a,b`. Toggling OFF the last unrestricted scene would search nothing,
// so we clamp: removing the final scene clears the restriction (all scenes)
// rather than producing an empty corpus.

export function invertSceneToggle(
  query: AskQuery,
  allocation: SceneAllocation[],
  clickedScene: string,
): QueryDiff | null {
  // Current effective scene set: explicit restriction, or "all scenes present".
  const allScenes = allocation.map((a) => a.scene);
  const current = new Set(query.scenes ?? allScenes);

  let next: Set<string>;
  if (current.has(clickedScene)) {
    next = new Set(current);
    next.delete(clickedScene);
    // Clamp: never allow an empty scene set — fall back to "all scenes".
    if (next.size === 0) {
      return {
        patch: { scenes: null },
        describe: `include all scenes (was only ${clickedScene})`,
        gesture: `toggle-scene:${clickedScene}`,
      };
    }
  } else {
    next = new Set(current);
    next.add(clickedScene);
  }

  // If the next set is every scene, represent it as null (no restriction) so the
  // query normalises to a canonical form.
  const nextArr = allScenes.filter((s) => next.has(s));
  const isAll = nextArr.length === allScenes.length;
  return {
    patch: { scenes: isAll ? null : nextArr },
    describe: current.has(clickedScene)
      ? `exclude scene ${clickedScene}`
      : `include scene ${clickedScene}`,
    gesture: `toggle-scene:${clickedScene}`,
  };
}

// ── Clearing-price / budget drag (water-filling allocation chart) ────────────
//
// The allocation chart draws the clearing price p* as a threshold: passages
// scoring above p* are admitted, below are priced out. Dragging that band down
// admits more passages; dragging up admits fewer. In the real binary the knob
// that moves admission is the budget `-k` (raising k lowers the effective p*,
// letting more passages in). So a price-drag inverts to a budget change.
//
// We translate a desired p* into the k that would produce it: k = number of
// passages across all scenes whose score exceeds the dragged price. This is
// exactly the binary's own demand(p) function, computed here from the last
// result's per-scene scores (passed in as `scoresByScene`).

export function invertPriceDrag(
  query: AskQuery,
  result: AskResult,
  draggedPrice: number,
): QueryDiff | null {
  // Reconstruct available scores per scene from the hit list. We only have the
  // returned hits' scores (the top ones), plus `available` counts. For a faithful
  // demand curve we use the scores we can see; this is monotone and sufficient to
  // move k in the right direction, which is what the gesture communicates.
  const scores = result.results.map((h) => h.score).sort((a, b) => b - a);
  const demand = scores.filter((s) => s > draggedPrice).length;

  // Clamp to a sane range: at least 1, and don't exceed total available passages.
  const totalAvailable = result.allocation.reduce((n, a) => n + a.available, 0);
  const nextBudget = Math.max(1, Math.min(demand || 1, totalAvailable || query.budget));

  if (nextBudget === query.budget) return null;
  const direction = nextBudget > query.budget ? "widen" : "narrow";
  return {
    patch: { budget: nextBudget },
    describe: `${direction} budget to -k ${nextBudget} (p* ≈ ${draggedPrice.toFixed(2)})`,
    gesture: `drag-price:${draggedPrice.toFixed(3)}`,
  };
}

// ── Budget stepper (explicit +/- on the allocation chart) ────────────────────

export function invertBudgetStep(query: AskQuery, delta: number): QueryDiff | null {
  const next = Math.max(1, query.budget + delta);
  if (next === query.budget) return null;
  return {
    patch: { budget: next },
    describe: `set budget -k ${next}`,
    gesture: `step-budget:${delta > 0 ? "+" : ""}${delta}`,
  };
}

// ── Flat / grouped toggle (result view) ──────────────────────────────────────
//
// Toggling between "global order" and "grouped by scene" maps to `--flat`. This
// is a presentation gesture that still lives in the query, so it participates in
// undo like any other edit.

export function invertFlatToggle(query: AskQuery): QueryDiff {
  return {
    patch: { flat: !query.flat },
    describe: query.flat ? "group results by scene" : "flatten to global order",
    gesture: "toggle-flat",
  };
}

// ── Prose / editor edits as peer diffs ───────────────────────────────────────
//
// So the undo stack treats prompt- and editor-originated changes identically to
// crossfilter gestures, we expose constructors for them too.

export function editQueryText(newText: string): QueryDiff {
  return { patch: { query: newText }, describe: "edit query text", gesture: "" };
}

export function setBudget(newBudget: number): QueryDiff {
  return {
    patch: { budget: Math.max(1, newBudget) },
    describe: `set budget -k ${Math.max(1, newBudget)}`,
    gesture: "",
  };
}

export function setScenes(scenes: string[] | null): QueryDiff {
  return {
    patch: { scenes: scenes && scenes.length > 0 ? scenes : null },
    describe: scenes && scenes.length > 0 ? `restrict to ${scenes.join(", ")}` : "all scenes",
    gesture: "",
  };
}
