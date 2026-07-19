import type { Goal, Step, StepId } from "./core/index.js";
import { type EdgeWeight } from "./core/index.js";
/** Optional configuration for a session (buhera §4.1). */
export interface SessionConfig {
    /** Cascade branching factor. v1 is flat; default 1. */
    cascadeArity?: number;
    /** Per-frame residue budget; ignored when cascadeArity === 1. */
    frameBudget?: number;
    /** Weight function for shared-term edges. Default: identity. */
    edgeWeight?: EdgeWeight;
}
/** The result of a carry request (buhera §4.1). Never thrown; returned. */
export type CarryResult = {
    ok: true;
    keep: StepId[];
    regenerable: StepId[];
    dropped: StepId[];
    ambientFloor: number;
    residueMap: ReadonlyMap<StepId, number>;
    diagnostics: {
        totalKeptCost: number;
        budgetRemaining: number;
        knapsackRelaxationGap: number;
    };
} | {
    ok: false;
    error: {
        kind: "unknown-step";
        stepId: StepId;
    } | {
        kind: "empty-goal";
    } | {
        kind: "infeasible";
        message: string;
    };
};
/** A serialized session snapshot (buhera §4.1). */
export interface SessionSnapshot {
    version: 1;
    config: SessionConfig;
    steps: SerializedStep[];
    internal: unknown;
}
/** Steps serialize with terms as arrays (Sets are not JSON-native). */
interface SerializedStep {
    id: StepId;
    terms: string[];
    cost: number;
    timestamp: number;
    payload?: unknown;
}
export declare class Session {
    private readonly steps;
    private readonly config;
    constructor(config?: SessionConfig);
    /** Register a step. Idempotent for identical (id, terms, cost); throws on
     * a conflicting redefinition (a programmer error, per buhera §5.1). */
    addStep(step: Step): void;
    /** Remove a step by id. Returns true if it existed (caller eviction). */
    removeStep(id: StepId): boolean;
    /** Number of steps currently held. */
    stepCount(): number;
    /** Ambient floor β of the current context graph. */
    floor(): number;
    /**
     * Compute the tandem carry for a goal under a budget. Never throws for
     * graph reasons; returns { ok: false, ... } when unsatisfiable.
     */
    carry(args: {
        goal: Goal;
        budget: number;
    }): CarryResult;
    /** Serialize for persistence. */
    snapshot(): SessionSnapshot;
    /** Reconstruct a session from a snapshot. */
    static fromSnapshot(snapshot: SessionSnapshot): Session;
    private view;
    private buildOpts;
    private residueOpts;
    private costOf;
}
export {};
//# sourceMappingURL=session.d.ts.map