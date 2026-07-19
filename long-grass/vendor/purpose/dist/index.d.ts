import type { Goal, Step, StepId } from "./core/index.js";
import { buildGraph, reach as reachView, defaultValue, type EdgeWeight } from "./core/index.js";
export type { StepId, ItemId, Term, TermSet, MediumId, Step, TaggedItem, Goal, WeightedEdge, ContextGraphView, } from "./core/index.js";
export { Session } from "./session.js";
export type { SessionConfig, CarryResult, SessionSnapshot } from "./session.js";
/** Paper §5.1: reachable set from goal terms, with BFS distance. */
export declare function seek(steps: ReadonlyArray<Step>, goal: Goal, opts?: {
    edgeWeight?: EdgeWeight;
}): ReadonlyMap<StepId, number>;
/** Paper §5.2: load-bearing subset of the reachable set (v1: = reachable). */
export declare function necessary(steps: ReadonlyArray<Step>, reached: ReadonlyMap<StepId, number>, goal: Goal, opts?: {
    edgeWeight?: EdgeWeight;
}): ReadonlySet<StepId>;
/** Paper §6: value-density greedy knapsack over the necessary set. */
export declare function knapsack(_steps: ReadonlyArray<Step>, necessarySet: ReadonlySet<StepId>, distances: ReadonlyMap<StepId, number>, residues: ReadonlyMap<StepId, number>, budget: number, costOf: (id: StepId) => number): {
    keep: StepId[];
    totalCost: number;
    relaxationGap: number;
};
/** Paper §3: ambient floor β of the graph induced by a step set. */
export declare function floor(steps: ReadonlyArray<Step>, opts?: {
    edgeWeight?: EdgeWeight;
}): number;
/** Paper §3: residue of one step relative to the induced graph. */
export declare function residue(steps: ReadonlyArray<Step>, stepId: StepId, opts?: {
    edgeWeight?: EdgeWeight;
}): number;
export { reachView as reachInView, buildGraph, defaultValue };
export { carryGreedy, carryExact } from "./core/index.js";
export type { CarryItem, CarryOutcome, EdgeWeight } from "./core/index.js";
//# sourceMappingURL=index.d.ts.map