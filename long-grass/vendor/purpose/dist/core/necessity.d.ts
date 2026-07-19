import type { ContextGraphView, Goal, ItemId } from "./types.js";
/**
 * Reach(goal): items whose terms are transitively goal-connected, with
 * BFS distance from the goal (paper Def: Reachability from the Goal).
 * Linear in the graph size.
 */
export declare function seek(view: ContextGraphView, goal: Goal): Map<ItemId, number>;
/**
 * The reachable set as ids (drops the distances).
 */
export declare function reach(view: ContextGraphView, goal: Goal): Set<ItemId>;
/**
 * nec(W, goal): the necessary (load-bearing) subset of a retained set W
 * (paper §5, Necessity). v1 interim: the members of W that are reachable
 * from the goal — `necessary ⊆ reachable`, with equality on tree-like
 * contexts (see reconciliation Decision 3). The exact dominator test
 * will narrow this on redundant contexts without changing the signature.
 */
export declare function necessary(view: ContextGraphView, retained: ReadonlySet<ItemId>, goal: Goal): Set<ItemId>;
/**
 * contrib(u, goal): does dropping u change the goal's reachable set?
 * v1 interim proxy consistent with `necessary := seek`: 1 if u is
 * reachable and removing it disconnects at least one otherwise-reachable
 * step from the goal, else 0. This is the cheap dominator-flavoured test;
 * it already narrows the pure `necessary := seek` on simple diamonds and
 * is the seed of the exact dominator upgrade.
 */
export declare function contribution(view: ContextGraphView, item: ItemId, goal: Goal, _retained: ReadonlySet<ItemId>): number;
//# sourceMappingURL=necessity.d.ts.map