/**
 * Process agents (network-yield §9) — the load-bearing unification.
 *
 * An allocated compute packet is not passive data the scheduler moves; it is a
 * standing, goal-directed agent that:
 *   - owns its completion target tau(x) as a standing goal;
 *   - descends its own residual r(x) by at least tau_0 each live tick (occupancy,
 *     Thm 9.1 "occupancy");
 *   - carries a strictly-monotone committed-step counter M (incorruptibility);
 *   - persists past r <= beta by GOAL SUCCESSION (Def 9.x): draws a fresh target
 *     from its occupation gamma instead of terminating;
 *   - returns a distinct response each interaction because its monitoring cell
 *     has advanced (ever-fresh response, Prop 9.x);
 *   - retires only when gamma returns null.
 *
 * The four musande invariants map directly:
 *   I1 conserved identity chi   -> AgentId + fixed PartitionCoords self-frame
 *   I2 monotone committed count -> M (never decremented)
 *   I3 search-not-fetch         -> each step reads the current advancing state
 *   I4 exclusive phases         -> observe (read residual/cell) then commit (descend)
 *
 * Residual/monotone-M semantics ported from `crates/srn-node/src/scheduler.rs`.
 */
import type { PartitionCoords } from "./coords.js";
/** The compute-tick tau_0 — the single quantum (physical/algorithmic/market). */
export declare const TICK = 0.001;
/** Resolution floor beta: a goal is "attained" once residual <= FLOOR. beta >= tau_0. */
export declare const FLOOR: number;
/** Opaque, unique, conserved agent identity (invariant I1, chi). */
export type AgentId = string & {
    readonly __brand: "AgentId";
};
/** Build an AgentId from a string. */
export declare function agentId(s: string): AgentId;
/** A completion target tau(x) in the agent's disposition space (R^d). */
export type Target = ReadonlyArray<number>;
/**
 * Occupation gamma (Def 9.x): given the just-attained target and the finite
 * interaction history, produce the next target — or null to retire.
 */
export type Occupation = (attained: Target, history: ReadonlyArray<Target>) => Target | null;
/** Which phase the agent is in (invariant I4: the two are exclusive). */
export type Phase = "observe" | "commit";
/** The agent's lifecycle state. */
export type AgentState = "pursuing" | "attained" | "stalled" | "retired";
/**
 * A persistent, goal-directed process agent.
 *
 * Construct with a self-frame (I1), an initial internal state `loc`, an initial
 * standing goal, and an occupation gamma for succession. Then drive it with
 * step(); it descends its residual, attains, and succeeds to fresh goals until
 * gamma retires it.
 */
export declare class ProcessAgent {
    readonly id: AgentId;
    /** Conserved self-frame (invariant I1, chi). */
    readonly frame: PartitionCoords;
    private loc;
    private goal;
    private readonly occupation;
    private readonly cellWidth;
    /** Monotone committed-step counter M (invariant I2); never decremented. */
    private M;
    private state;
    private phase;
    private readonly history;
    private cell;
    private cellCrossings;
    private successions;
    /** Smallest live-step residual decrement observed (for occupancy checks). */
    private minLiveStep;
    private flatTicks;
    constructor(args: {
        id: AgentId;
        frame: PartitionCoords;
        loc: Target;
        goal: Target;
        occupation?: Occupation;
        cellWidth?: number;
    });
    /** Current residual r(x) = distance from internal state to standing goal. */
    residual(): number;
    /** Monotone committed-step count M (invariant I2). */
    committedStep(): number;
    currentState(): AgentState;
    currentPhase(): Phase;
    goalsAttained(): number;
    cellCrossingCount(): number;
    /** The agent's felt drive is its residual; standing goal is its target. */
    standingGoal(): Target;
    /** A read-only snapshot of internal state (for monitoring — cell index only). */
    monitoringCell(): string;
    /**
     * One tick of autonomous goal pursuit (I4: observe then commit).
     *
     * If the agent is already inside the target cell (r <= FLOOR), it SUCCEEDS to a
     * fresh goal (persistence). Otherwise it descends its residual toward the goal
     * by exactly one compute-tick — unless a full tau step would reach the target
     * cell, in which case it lands (the attainment step, not counted as a >=tau live
     * step). Every action (descent or succession) advances M.
     *
     * Returns the new lifecycle state.
     */
    step(): AgentState;
    /** Run up to `maxTicks` ticks (or until retired). Returns ticks actually run. */
    run(maxTicks: number): number;
    /** The smallest per-tick residual decrement over all live (non-attaining) steps. */
    minLiveStepSeen(): number;
    /** A cell-indexed response — differs across interactions once the cell advances. */
    respond<T>(render: (cell: string, M: number) => T): T;
    private succeed;
    private unit;
}
/**
 * The musande-compatible Agent interface. A ProcessAgent satisfies it. Kept
 * minimal and structural (musande is not vendored in long-grass), so long-grass
 * can treat pylon compute-agents and musande vocational agents uniformly.
 */
export interface Agent {
    readonly id: AgentId;
    readonly frame: PartitionCoords;
    residual(): number;
    committedStep(): number;
    currentState(): AgentState;
    standingGoal(): Target;
    monitoringCell(): string;
}
//# sourceMappingURL=agent.d.ts.map