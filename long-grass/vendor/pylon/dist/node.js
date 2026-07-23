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
import { evalExpr, serializeExpr } from "./srn/expr.js";
import { Registry } from "./registry.js";
import { Scheduler, Task } from "./scheduler.js";
import { TrajectoryAddress, digestOf } from "./label.js";
/** A single SRN node. */
export class SrnNode {
    config;
    frame;
    registry = new Registry();
    scheduler = new Scheduler();
    address;
    /** Append-only evaluation log. */
    evalLog = [];
    env = new Map();
    clock = 0; // deterministic monotonic timestamp source
    constructor(config) {
        this.config = config;
        this.frame = { coords: config.coords, trajectoryCount: 0 };
        this.address = TrajectoryAddress.root(config.addressBranching);
    }
    /** A default reference node at (1,0,0,+1). */
    static reference(coords) {
        return new SrnNode({ coords, addressBranching: 3, tickBudget: 8 });
    }
    /** The node's current receiver frame (identity + committed count). */
    receiverFrame() {
        return this.frame;
    }
    /** Monotone committed-step count M. */
    trajectoryCount() {
        return this.frame.trajectoryCount;
    }
    addressKey() {
        return this.address.key();
    }
    /**
     * Evaluate an SRN expression in this node's receiver frame (receiver-relative).
     * Each call advances M, appends to Gamma, and logs the record.
     */
    evaluate(expr) {
        const result = evalExpr(expr, this.frame, this.env);
        // advance the trajectory address (one digit per committed unit) -> M grows
        const digit = this.frame.trajectoryCount % this.config.addressBranching;
        this.address.advance(digit);
        this.frame = { coords: this.frame.coords, trajectoryCount: this.address.count };
        const record = {
            exprDigest: digestOf(serializeExpr(expr)).hex,
            address: this.address.key(),
            frame: this.frame,
            result,
            timestampNs: this.clock++,
        };
        this.registry.append(this.config.coords, record);
        this.evalLog.push(record);
        return result;
    }
    /** Submit a task and run up to tickBudget dispatches. */
    scheduleAndRun(task, workFn) {
        this.scheduler.addTask(task);
        return this.scheduler.tick(this.config.tickBudget, workFn);
    }
    /** Fetch the latest successful evaluation for a coord (Fetch / Fetch-Miss). */
    fetch(coords) {
        return this.registry.fetchLatest(coords)?.record;
    }
    /** Install a binding into the evaluation environment. */
    install(key, value) {
        this.env.set(key, value);
    }
    registrySize() {
        return this.registry.cellCount();
    }
    totalCommitted() {
        return this.registry.totalCount;
    }
}
//# sourceMappingURL=node.js.map