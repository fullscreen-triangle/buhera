/**
 * Residue-driven scheduler (network-yield §liveness; scheduling-mechanism paper §6).
 *
 * Priority rule (residue priority):
 *   P(t) = Delta_t / max(rho_t - theta_t, beta)
 * where beta > 0 is the entropic floor, Delta_t is the descent rate over a
 * window, rho_t is the current residue, and theta_t is the sufficiency threshold.
 *
 * Invariants (proved in the paper, mirrored in tests):
 *   - a stalled task (Delta = 0) gets P = 0 -> never dispatched while others run;
 *   - a task at theta gets P = +inf -> dispatched first (finishes immediately);
 *   - trajectory count M is strictly monotone across committed units;
 *   - liveness: every live task's residue reaches theta in finite time
 *     (settling bound n_term = 1 + ceil(log_{d+1}(K/d))).
 *
 * Ported from `crates/srn-node/src/scheduler.rs`.
 */
/** Entropic floor beta (strictly positive). */
export declare const FLOOR = 0.01;
/** Stall declared after this many flat committed units. */
export declare const STALL_WINDOW = 5;
export type TaskState = "running" | "stalled" | "sufficient" | "done" | "declined";
/** A live scheduler task. */
export declare class Task {
    readonly id: string;
    /** Categorical complexity K — distinguishable work-trajectories needed. */
    readonly complexity: number;
    /** Operator type count d (for the T(n,d) inflation / termination bound). */
    readonly opTypes: number;
    /** Committed unit count M — strictly monotone. */
    trajectoryCount: number;
    /** Current residue rho in [beta, 100]. */
    residue: number;
    /** Sufficiency threshold theta in [beta, 100]. */
    threshold: number;
    state: TaskState;
    private readonly residueHistory;
    constructor(id: string, complexity: number, opTypes: number, initialResidue: number);
    withThreshold(theta: number): this;
    /** Descent rate Delta over the stall window — non-negative by definition. */
    descentRate(): number;
    /**
     * Residue priority P(t) — the scheduling signal.
     *
     * A running task with no committed units yet is UNTRIED (not stalled): it has
     * no descent history to measure, so it receives a positive bootstrap priority
     * proportional to how far it is above threshold, guaranteeing it is dispatched
     * at least once. Only a task that has been tried and then goes flat is stalled
     * (Delta = 0 -> P = 0). This is what makes liveness hold from a cold start.
     */
    priority(): number;
    /** Record a new residue reading after one committed unit; returns priority. */
    commitUnit(newResidue: number): number;
    /** Expected termination unit count: n_term(K,d) = 1 + ceil(log_{d+1}(K/d)). */
    terminationBound(): number;
    isBehindCurve(): boolean;
}
/** One dispatched unit's result. */
export interface TickResult {
    readonly taskId: string;
    readonly trajectoryCount: number;
    readonly residueBefore: number;
    readonly residueAfter: number;
    readonly priorityAfter: number;
    readonly state: TaskState;
}
/** The residue-driven scheduler. */
export declare class Scheduler {
    readonly tasks: Map<string, Task>;
    addTask(task: Task): void;
    /**
     * One scheduler pass over `budget` dispatches. Repeatedly dispatches the
     * highest-priority running task; a stalled-only frontier (all P=0) declines.
     * `dispatch(task)` performs the work unit and returns the new residue.
     */
    tick(budget: number, dispatch: (task: Task) => number): TickResult[];
    runningCount(): number;
    stalledTasks(): Task[];
}
//# sourceMappingURL=scheduler.d.ts.map