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
export const FLOOR = 0.01;
/** Stall declared after this many flat committed units. */
export const STALL_WINDOW = 5;
/** A live scheduler task. */
export class Task {
    id;
    /** Categorical complexity K — distinguishable work-trajectories needed. */
    complexity;
    /** Operator type count d (for the T(n,d) inflation / termination bound). */
    opTypes;
    /** Committed unit count M — strictly monotone. */
    trajectoryCount = 0;
    /** Current residue rho in [beta, 100]. */
    residue;
    /** Sufficiency threshold theta in [beta, 100]. */
    threshold;
    state = "running";
    residueHistory;
    constructor(id, complexity, opTypes, initialResidue) {
        this.id = id;
        this.complexity = complexity;
        this.opTypes = opTypes;
        this.residue = initialResidue;
        this.threshold = FLOOR;
        this.residueHistory = [initialResidue];
    }
    withThreshold(theta) {
        this.threshold = Math.max(theta, FLOOR);
        return this;
    }
    /** Descent rate Delta over the stall window — non-negative by definition. */
    descentRate() {
        const h = this.residueHistory;
        if (h.length < 2)
            return 0;
        const window = Math.min(STALL_WINDOW, h.length - 1);
        const oldest = h[h.length - 1 - window];
        const newest = h[h.length - 1];
        return Math.max(oldest - newest, 0) / window;
    }
    /**
     * Residue priority P(t) — the scheduling signal.
     *
     * A running task with no committed units yet is UNTRIED (not stalled): it has
     * no descent history to measure, so it receives a positive bootstrap priority
     * proportional to how far it is above threshold, guaranteeing it is dispatched
     * at least once. Only a task that has been tried and then goes flat is stalled
     * (Delta = 0 -> P = 0). This is what makes liveness hold from a cold start.
     */
    priority() {
        if (this.state !== "running")
            return 0;
        if (this.residue <= this.threshold)
            return Infinity; // finish immediately
        const gap = Math.max(this.residue - this.threshold, FLOOR);
        if (this.trajectoryCount === 0)
            return gap; // untried -> bootstrap dispatch
        const delta = this.descentRate();
        if (delta === 0)
            return 0; // tried and flat -> stalled, do not feed
        return delta / gap;
    }
    /** Record a new residue reading after one committed unit; returns priority. */
    commitUnit(newResidue) {
        this.trajectoryCount++; // strictly monotone — never decremented
        this.residue = Math.max(newResidue, FLOOR);
        this.residueHistory.push(this.residue);
        if (this.residue <= this.threshold) {
            this.state = "sufficient";
        }
        else if (this.descentRate() === 0 && this.residueHistory.length > STALL_WINDOW) {
            const flat = this.residueHistory
                .slice(-STALL_WINDOW)
                .every((r) => Math.abs(r - this.residue) < 1e-9);
            if (flat)
                this.state = "stalled";
        }
        return this.priority();
    }
    /** Expected termination unit count: n_term(K,d) = 1 + ceil(log_{d+1}(K/d)). */
    terminationBound() {
        const k = this.complexity;
        const d = this.opTypes;
        if (d <= 0 || k <= 0)
            return 1;
        const inner = Math.log(k / d) / Math.log(d + 1);
        return 1 + Math.ceil(inner);
    }
    isBehindCurve() {
        return this.trajectoryCount > this.terminationBound() && this.state === "running";
    }
}
/** The residue-driven scheduler. */
export class Scheduler {
    tasks = new Map();
    addTask(task) {
        this.tasks.set(task.id, task);
    }
    /**
     * One scheduler pass over `budget` dispatches. Repeatedly dispatches the
     * highest-priority running task; a stalled-only frontier (all P=0) declines.
     * `dispatch(task)` performs the work unit and returns the new residue.
     */
    tick(budget, dispatch) {
        const results = [];
        let remaining = budget;
        while (remaining > 0) {
            // pick the highest-priority running task
            let top;
            let topP = -Infinity;
            for (const t of this.tasks.values()) {
                if (t.state !== "running")
                    continue;
                const p = t.priority();
                if (p > topP) {
                    topP = p;
                    top = t;
                }
            }
            if (top === undefined)
                break; // no runnable tasks
            if (topP === 0)
                break; // all runnable tasks stalled -> decline
            const residueBefore = top.residue;
            const newResidue = dispatch(top);
            const priorityAfter = top.commitUnit(newResidue);
            results.push({
                taskId: top.id,
                trajectoryCount: top.trajectoryCount,
                residueBefore,
                residueAfter: top.residue,
                priorityAfter,
                state: top.state,
            });
            remaining--;
        }
        return results;
    }
    runningCount() {
        let n = 0;
        for (const t of this.tasks.values())
            if (t.state === "running")
                n++;
        return n;
    }
    stalledTasks() {
        return [...this.tasks.values()].filter((t) => t.state === "stalled");
    }
}
//# sourceMappingURL=scheduler.js.map