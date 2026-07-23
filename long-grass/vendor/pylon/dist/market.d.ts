/**
 * The yield market (network-yield §5-§7).
 *
 * Each execution slot is priced at its separation cost sep(e, A) — the marginal
 * yield lost by removing it. Clearing the market to deterministic closure yields
 * the assignment at which no single-agent reassignment improves yield by more
 * than tau_0, and the clearing prices ARE the separation costs. This is one face
 * of the Three-way Equivalence (Thm 7.1): yield-optimality, deterministic
 * closure, and market clearing coincide at a single fixed point.
 *
 * This module operates on abstract agents (anything with an id and a per-slot
 * payoff) and nodes (execution slots), so it is reusable by the cluster and by
 * lightweight callers. Payoff yield_x(e) is supplied by the caller — in the full
 * runtime it is the target-ward progress the agent makes on slot e per tick.
 */
import type { AgentId } from "./agent.js";
/** An execution slot (node) in the yield market. */
export interface Slot {
    readonly id: string;
    /** Thread capacity c(e) >= 1. */
    readonly capacity: number;
    /** Maximum throughput rate v_bar(e) > 0. */
    readonly maxRate: number;
}
/** A yield-market price p(e) = sep(e, A). */
export type Price = number;
/** Per-agent, per-slot payoff yield_x(e): the yield x contributes on slot e. */
export type PayoffFn = (agent: AgentId, slot: string) => number;
/** An assignment A : AgentId -> slot id. */
export type Assignment = ReadonlyMap<AgentId, string>;
/** Utilisation cost g_u: strictly convex, g_u(0)=0, g_u'>0, g_u''>0 (Ax. Cost). */
export type UtilisationCost = (v: number) => number;
/** Default utilisation cost g_u(v) = v^2 (strictly convex, g_u(0)=0). */
export declare const defaultUtilisationCost: UtilisationCost;
/**
 * Network transport yield of an assignment (network-yield Def. 3.4), unit-slot
 * form: numerator is total payoff; denominator is total resource consumption in
 * compute-ticks weighted by utilisation cost. With one occupied slot per agent
 * and a fixed rate, the denominator is a positive constant across full
 * assignments, so ranking by yield ranks by total payoff — which is what the
 * Three-way Equivalence reasons about.
 */
export declare function yieldOf(assignment: Assignment, agents: ReadonlyArray<AgentId>, slots: ReadonlyArray<Slot>, payoff: PayoffFn, tick: number, utilisationCost?: UtilisationCost): number;
/**
 * Separation cost sep(e, A) (network-yield Def. 5.1): the marginal yield lost by
 * removing slot e — the tasks on e must move to their best remaining slot.
 * sep >= 0. A redundant slot has sep = 0; a bottleneck has large sep.
 *
 * Computed on the yield NUMERATOR (total payoff), so it is the shadow price of
 * the slot at the given assignment — a well-defined market price independent of
 * how many slots the counterfactual reallocation happens to leave used. This is
 * what makes clearing prices support individual rationality (Thm 7.1).
 */
export declare function separationCost(slotId: string, assignment: Assignment, agents: ReadonlyArray<AgentId>, slots: ReadonlyArray<Slot>, payoff: PayoffFn, _tick: number, _utilisationCost?: UtilisationCost): Price;
/**
 * Clear the yield market to deterministic closure (network-yield Thm 7.1).
 *
 * Greedy ascent: repeatedly apply the single-agent reassignment / swap that most
 * improves yield, until no move improves it by more than tau_0 (closure). At
 * closure the returned prices p(e) = sep(e, A) are the separation costs, and the
 * assignment is simultaneously yield-optimal and market-clearing.
 *
 * `agents.length <= slots.length` with unit-capacity is the canonical case; the
 * greedy neighbourhood includes moves to empty slots and swaps with occupants.
 */
export declare function clearMarket(agents: ReadonlyArray<AgentId>, slots: ReadonlyArray<Slot>, payoff: PayoffFn, tick: number, utilisationCost?: UtilisationCost): {
    assignment: Map<AgentId, string>;
    prices: Map<string, Price>;
};
/**
 * Forced optimal utilisation (network-yield Thm 7.x, corrected net-yield form).
 *
 * At closure a slot runs at v* maximising net yield P*b(v) - tick*c*g_u(v), with
 * b strictly concave (diminishing returns) and g_u strictly convex. This returns
 * the unique interior v* by 1-D search — the marginal-balance point
 * P*b'(v*) = tick*c*g_u'(v*).
 */
export declare function forcedUtilisation(args: {
    pressure: number;
    capacity: number;
    tick: number;
    maxRate: number;
    benefit?: (v: number) => number;
    utilisationCost?: UtilisationCost;
    steps?: number;
}): number;
//# sourceMappingURL=market.d.ts.map