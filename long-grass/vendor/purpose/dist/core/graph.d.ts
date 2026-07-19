import type { ContextGraphView, Goal, ItemId, Step, WeightedEdge } from "./types.js";
/** How an edge weight is derived from the number of shared terms. */
export type EdgeWeight = (sharedTermCount: number) => number;
/** Default edge weight: identity on the shared-term count. */
export declare const identityWeight: EdgeWeight;
/**
 * Build a ContextGraphView from a set of steps (the Session's job, and
 * the reference construction for graffiti's adapter to match).
 *
 * - one item per step (terms carried through),
 * - an undirected edge between two steps sharing ≥1 term, weighted by
 *   edgeWeight(sharedCount),
 * - an edge from every step to the medium, weighted by the step's own
 *   term count (its cost of being told apart from "everything else"),
 * - floor = minimum positive edge weight.
 */
export declare function buildGraph(steps: ReadonlyArray<Step>, opts?: {
    medium?: string;
    edgeWeight?: EdgeWeight;
}): ContextGraphView;
/** The floor β: minimum positive edge weight; 0 if no positive edges. */
export declare function floorOfEdges(edges: ReadonlyArray<WeightedEdge>): number;
/**
 * Ambient floor of the graph induced by a step set (paper §3).
 * 0 if fewer than two steps or no shared-term edges exist
 * (buhera-specifications.md §4.3 `floor`).
 */
export declare function floor(steps: ReadonlyArray<Step>, opts?: {
    medium?: string;
    edgeWeight?: EdgeWeight;
}): number;
/**
 * Residue of one step relative to the graph induced by a step set
 * (paper §3). v1 approximation: the minimum positive weight of an edge
 * incident to the step (its cheapest separator). Guaranteed ≥ floor for
 * every non-isolated step (buhera-specifications.md §4.3 `residue`).
 * Returns the medium-edge weight for an otherwise-isolated step, which is
 * that step's cost of being told apart from everything else.
 */
export declare function residue(steps: ReadonlyArray<Step>, stepId: ItemId, opts?: {
    edgeWeight?: EdgeWeight;
}): number;
/** Adjacency over shared-term edges only (excludes the medium). */
export declare function termAdjacency(view: ContextGraphView): Map<ItemId, Set<ItemId>>;
/** The set of items whose terms intersect the goal (distance-0 seeds). */
export declare function goalSeeds(view: ContextGraphView, goal: Goal): Set<ItemId>;
//# sourceMappingURL=graph.d.ts.map