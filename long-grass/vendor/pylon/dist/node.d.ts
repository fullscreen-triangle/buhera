/**
 * SRN node — the top-level single-node runtime combining all subsystems.
 *
 * A node N = (Gamma, M, F) where:
 *   Gamma — live cell registry (content-addressed by partition coords)
 *   M     — trajectory count (strictly monotone; the committed-step counter)
 *   F     — receiver frame (this node's identity)
 *
 * Forest Theorem: membership in the SRN network = capability to evaluate SRN
 * expressions. No registration, no administrator (SRN Thm 8.1).
 *
 * Ported from `crates/srn-node/src/node.rs`.
 */
import type { PartitionCoords } from "./coords.js";
import { type Expr, type EvalResult, type ReceiverFrame } from "./srn/expr.js";
import { Task, type TickResult } from "./scheduler.js";
import { type EvalRecord } from "./label.js";
export interface NodeConfig {
    readonly coords: PartitionCoords;
    /** Branching factor for the trajectory-address tree (default 3). */
    readonly addressBranching: number;
    /** Scheduler dispatch budget per schedule call. */
    readonly tickBudget: number;
}
/** A single SRN node. */
export declare class SrnNode {
    readonly config: NodeConfig;
    private frame;
    private readonly registry;
    private readonly scheduler;
    private readonly address;
    /** Append-only evaluation log. */
    readonly evalLog: EvalRecord[];
    private readonly env;
    private clock;
    constructor(config: NodeConfig);
    /** A default reference node at (1,0,0,+1). */
    static reference(coords: PartitionCoords): SrnNode;
    /** The node's current receiver frame (identity + committed count). */
    receiverFrame(): ReceiverFrame;
    /** Monotone committed-step count M. */
    trajectoryCount(): number;
    addressKey(): string;
    /**
     * Evaluate an SRN expression in this node's receiver frame (receiver-relative).
     * Each call advances M, appends to Gamma, and logs the record.
     */
    evaluate(expr: Expr): EvalResult;
    /** Submit a task and run up to tickBudget dispatches. */
    scheduleAndRun(task: Task, workFn: (t: Task) => number): TickResult[];
    /** Fetch the latest successful evaluation for a coord (Fetch / Fetch-Miss). */
    fetch(coords: PartitionCoords): EvalRecord | undefined;
    /** Install a binding into the evaluation environment. */
    install(key: string, value: unknown): void;
    registrySize(): number;
    totalCommitted(): number;
}
//# sourceMappingURL=node.d.ts.map