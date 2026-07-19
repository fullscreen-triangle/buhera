import type { ItemId } from "./types.js";
/** One candidate for the carry. */
export interface CarryItem {
    id: ItemId;
    value: number;
    cost: number;
}
/** The result of a carry: which items to keep, and the totals. */
export interface CarryOutcome {
    keep: ItemId[];
    totalValue: number;
    totalCost: number;
    /** Upper bound on the relative optimality gap (cost_max / budget). */
    relaxationGap: number;
}
/** Canonical value: residue discounted by distance from the goal (§6). */
export declare function defaultValue(residue: number, distanceFromGoal: number): number;
/**
 * Value-density greedy carry (Thm: Value-Density Greedy Is Optimal Under
 * the Relaxation). Admits items in decreasing value/cost while the budget
 * allows. Optimal under the fractional relaxation; within cost_max/budget
 * of the integral optimum.
 */
export declare function carryGreedy(items: ReadonlyArray<CarryItem>, budget: number): CarryOutcome;
/**
 * Exact 0/1 knapsack DP (Thm: The Optimal Carry Is a 0/1 Knapsack).
 * O(items * budget). Requires integer costs and budget; for non-integer
 * costs, callers should round costs up to integers (a conservative carry)
 * or use carryGreedy. Falls back to greedy if any cost is non-integer.
 */
export declare function carryExact(items: ReadonlyArray<CarryItem>, budget: number): CarryOutcome;
//# sourceMappingURL=knapsack.d.ts.map