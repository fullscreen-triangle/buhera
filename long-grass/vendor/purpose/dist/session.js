// The Session class: the stateful layer (buhera-specifications.md §4.2).
//
// A Session accumulates a history of steps and answers `carry(goal, budget)`
// by running the tandem: build the context graph, seek the reachable slice,
// keep the necessary subset, fit it to the budget with the knapsack, and
// partition the history into keep / regenerable / dropped. Pure underneath
// (it delegates every computation to core), stateful only in that it holds
// the step set between calls.
import { buildGraph, carryGreedy, defaultValue, necessary, reach, residue, seek, } from "./core/index.js";
export class Session {
    steps = new Map();
    config;
    constructor(config) {
        this.config = {
            cascadeArity: config?.cascadeArity ?? 1,
            frameBudget: config?.frameBudget ?? Infinity,
            ...(config?.edgeWeight ? { edgeWeight: config.edgeWeight } : {}),
        };
    }
    /** Register a step. Idempotent for identical (id, terms, cost); throws on
     * a conflicting redefinition (a programmer error, per buhera §5.1). */
    addStep(step) {
        const existing = this.steps.get(step.id);
        if (existing) {
            const sameCost = existing.cost === step.cost;
            const sameTerms = existing.terms.size === step.terms.size &&
                [...existing.terms].every((t) => step.terms.has(t));
            if (sameCost && sameTerms)
                return; // idempotent no-op
            throw new Error(`Session.addStep: step '${step.id}' already exists with different terms/cost`);
        }
        this.steps.set(step.id, step);
    }
    /** Remove a step by id. Returns true if it existed (caller eviction). */
    removeStep(id) {
        return this.steps.delete(id);
    }
    /** Number of steps currently held. */
    stepCount() {
        return this.steps.size;
    }
    /** Ambient floor β of the current context graph. */
    floor() {
        const view = this.view();
        return view.floor;
    }
    /**
     * Compute the tandem carry for a goal under a budget. Never throws for
     * graph reasons; returns { ok: false, ... } when unsatisfiable.
     */
    carry(args) {
        const { goal, budget } = args;
        if (goal.terms.size === 0)
            return { ok: false, error: { kind: "empty-goal" } };
        if (!(budget > 0)) {
            return {
                ok: false,
                error: { kind: "infeasible", message: `budget must be > 0, got ${budget}` },
            };
        }
        const steps = [...this.steps.values()];
        const view = this.view();
        // Seek: reachable slice, with distances.
        const distances = seek(view, goal);
        const reachable = new Set(distances.keys());
        // Necessary subset of the reachable set (v1: = reachable; see
        // reconciliation Decision 3).
        const nec = necessary(view, reachable, goal);
        // Fit necessary steps to the budget by residue density.
        const residueMap = new Map();
        const carryItems = [];
        for (const id of nec) {
            const r = residue(steps, id, this.residueOpts());
            residueMap.set(id, r);
            const d = distances.get(id) ?? 0;
            carryItems.push({ id, value: defaultValue(r, d), cost: this.costOf(id) });
        }
        const outcome = carryGreedy(carryItems, budget);
        // Partition the whole history:
        //   keep        = budget-admitted necessary steps
        //   regenerable = necessary/reachable but not admitted (re-fetchable)
        //   dropped     = unreachable from the goal (free to drop)
        const keepSet = new Set(outcome.keep);
        const keep = [];
        const regenerable = [];
        const dropped = [];
        for (const s of steps) {
            if (keepSet.has(s.id))
                keep.push(s.id);
            else if (reachable.has(s.id))
                regenerable.push(s.id);
            else
                dropped.push(s.id);
        }
        return {
            ok: true,
            keep,
            regenerable,
            dropped,
            ambientFloor: view.floor,
            residueMap,
            diagnostics: {
                totalKeptCost: outcome.totalCost,
                budgetRemaining: budget - outcome.totalCost,
                knapsackRelaxationGap: outcome.relaxationGap,
            },
        };
    }
    /** Serialize for persistence. */
    snapshot() {
        return {
            version: 1,
            config: this.config,
            steps: [...this.steps.values()].map((s) => ({
                id: s.id,
                terms: [...s.terms],
                cost: s.cost,
                timestamp: s.timestamp,
                ...(s.payload !== undefined ? { payload: s.payload } : {}),
            })),
            internal: null,
        };
    }
    /** Reconstruct a session from a snapshot. */
    static fromSnapshot(snapshot) {
        const s = new Session(snapshot.config);
        for (const st of snapshot.steps) {
            s.addStep({
                id: st.id,
                terms: new Set(st.terms),
                cost: st.cost,
                timestamp: st.timestamp,
                ...(st.payload !== undefined ? { payload: st.payload } : {}),
            });
        }
        return s;
    }
    // ---- internals ----
    view() {
        return buildGraph([...this.steps.values()], this.buildOpts());
    }
    buildOpts() {
        return this.config.edgeWeight ? { edgeWeight: this.config.edgeWeight } : {};
    }
    residueOpts() {
        return this.buildOpts();
    }
    costOf(id) {
        return this.steps.get(id)?.cost ?? 0;
    }
}
//# sourceMappingURL=session.js.map