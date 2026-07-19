// @buhera/purpose — public surface.
//
// Two entry heights over one implementation (reconciliation Decision 1):
//   - the pure core (over a ContextGraphView) for callers that own their
//     own graph and session (Graffiti);
//   - the Session class (over Step[]) for callers that want purpose to
//     hold the history (long-grass / Buhera).
//
// buhera-specifications.md §4.4 asks the top level to export the Session,
// the shared types, and the pure operators `seek`/`necessary`/`knapsack`/
// `floor`/`residue` in a Step[]-flavoured form. Those wrappers are here;
// the graph-view-flavoured operators live under the `./core` subpath.
import { buildGraph, seek as seekView, necessary as necessaryView, reach as reachView, floor as floorSteps, residue as residueSteps, defaultValue, carryGreedy, } from "./core/index.js";
// ---- The Session class (stateful layer) ----
export { Session } from "./session.js";
// ---- Pure operators (Step[]-flavoured, buhera §4.3) ----
/** Paper §5.1: reachable set from goal terms, with BFS distance. */
export function seek(steps, goal, opts) {
    return seekView(buildGraph(steps, opts), goal);
}
/** Paper §5.2: load-bearing subset of the reachable set (v1: = reachable). */
export function necessary(steps, reached, goal, opts) {
    const view = buildGraph(steps, opts);
    return necessaryView(view, new Set(reached.keys()), goal);
}
/** Paper §6: value-density greedy knapsack over the necessary set. */
export function knapsack(_steps, necessarySet, distances, residues, budget, costOf) {
    const items = [];
    for (const id of necessarySet) {
        const r = residues.get(id) ?? 0;
        const d = distances.get(id) ?? 0;
        items.push({ id, value: defaultValue(r, d), cost: costOf(id) });
    }
    const out = carryGreedy(items, budget);
    return { keep: out.keep, totalCost: out.totalCost, relaxationGap: out.relaxationGap };
}
/** Paper §3: ambient floor β of the graph induced by a step set. */
export function floor(steps, opts) {
    return floorSteps(steps, opts);
}
/** Paper §3: residue of one step relative to the induced graph. */
export function residue(steps, stepId, opts) {
    return residueSteps(steps, stepId, opts);
}
// ---- Re-exports for graph-view-flavoured callers (Graffiti) ----
export { reachView as reachInView, buildGraph, defaultValue };
export { carryGreedy, carryExact } from "./core/index.js";
//# sourceMappingURL=index.js.map