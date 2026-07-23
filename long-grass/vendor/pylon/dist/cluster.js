/**
 * The Cluster — the top-level distributed runtime (contract §6.3).
 *
 * A Cluster is a set of SRN nodes (execution slots), a yield market that clears
 * task-to-slot assignments at separation-cost prices, and a bank of Kuramoto
 * oscillators keeping the per-node schedulers phase-locked. submit() takes an SRN
 * expression, selects a receiver by the expression's `to { region }` clause and
 * the current market clearing, instantiates a persistent ProcessAgent for it, and
 * returns a Yield.
 *
 * Forest openness (SRN Thm 8.1): membership is capability, not enrolment. A node
 * added with a valid frame is a peer immediately; there is no whitelist.
 */
import { coordTuple, coordsEqual } from "./coords.js";
import { parseSrn, isParseError } from "./srn/parse.js";
import { evalExpr } from "./srn/expr.js";
import { ProcessAgent, agentId, TICK, } from "./agent.js";
import { clearMarket } from "./market.js";
import { KuramotoBank, criticalCoupling } from "./kuramoto.js";
export class Cluster {
    tick = TICK;
    nodeSpecs;
    utilisationCost;
    coupling;
    kuramoto;
    live = new Map();
    counter = 0;
    constructor(config) {
        this.nodeSpecs = [...config.nodes];
        this.utilisationCost = config.utilisationCost ?? ((v) => v * v);
        this.coupling = config.coupling ?? 2.0;
        // one oscillator per node; small deterministic frequency spread
        const freqs = this.nodeSpecs.map((_, i) => 1.0 + 0.01 * (i - this.nodeSpecs.length / 2));
        this.kuramoto = new KuramotoBank(freqs, this.coupling);
    }
    // ---- read-only inspectors (contract §6.3) ----
    nodes() {
        return this.nodeSpecs;
    }
    /** Add a node. Forest openness: no enrolment; the node is a peer at once. */
    addNode(spec) {
        if (!this.nodeSpecs.some((n) => n.id === spec.id))
            this.nodeSpecs.push(spec);
    }
    price(nodeId) {
        const { prices } = this.clear();
        return prices.get(nodeId) ?? 0;
    }
    /** Kuramoto order parameter (R, psi). */
    orderParameter() {
        // advance the phase model a little so coupling has effect, deterministically
        this.kuramoto.run(200, 0.05);
        return this.kuramoto.order();
    }
    isPhaseLocked() {
        return this.orderParameter().R >= 0.95;
    }
    criticalCoupling(sigmaOmega) {
        return criticalCoupling(sigmaOmega);
    }
    liveAgents() {
        return [...this.live.values()].map((e) => e.agent);
    }
    agent(id) {
        return this.live.get(id)?.agent ?? null;
    }
    // ---- submission (contract §6.3) ----
    /**
     * Submit an SRN expression for evaluation somewhere in the cluster. Selects a
     * receiver by the expression's target region and the current market clearing,
     * instantiates a persistent ProcessAgent, evaluates receiver-relatively, and
     * returns a Yield. A succession map makes the agent persist past completion.
     */
    submit(expr, options) {
        // 1. parse (enforces the mandatory `not` boundary, SRN Cor 2.1)
        let glyph;
        if (typeof expr === "string") {
            const parsed = parseSrn(expr);
            if (isParseError(parsed)) {
                if (parsed.kind === "no-negation-boundary")
                    return { ok: false, error: { kind: "no-negation-boundary" } };
                return { ok: false, error: { kind: "malformed-srn", at: parsed.at } };
            }
            glyph = parsed;
        }
        else {
            glyph = expr;
        }
        // 2. candidate receivers: nodes whose frame lies in the target region. Here
        //    the region is the glyph's target coordinate; a node matches if its frame
        //    equals the target, else all nodes are candidates (region = wildcard).
        const candidates = this.candidateNodes(glyph.target);
        if (candidates.length === 0) {
            return { ok: false, error: { kind: "boundary-rejects", frame: coordTuple(glyph.target) } };
        }
        // 3. clear the market over live agents + this new one to pick the slot
        const newId = agentId(`a${this.counter++}`);
        const chosen = this.chooseSlot(newId, candidates);
        if (chosen === null) {
            return { ok: false, error: { kind: "no-capacity", price: 0 } };
        }
        // 4. instantiate the persistent process agent (network-yield §9)
        const goal = options?.goal ?? defaultGoalFor(glyph.target);
        const agent = new ProcessAgent({
            id: newId,
            frame: glyph.target,
            loc: goal.map(() => 0),
            goal,
            ...(options?.succession ? { occupation: options.succession } : {}),
        });
        this.live.set(newId, { agent, nodeId: chosen.node.id });
        // 5. receiver-relative evaluation at the chosen node's frame (SRN Thm 4.2)
        const frame = { coords: this.frameOf(chosen.node), trajectoryCount: agent.committedStep() };
        const result = evalExpr(glyph, frame, new Map());
        if (result.kind === "rejected") {
            this.live.delete(newId);
            return { ok: false, error: { kind: "boundary-rejects", frame: coordTuple(glyph.target) } };
        }
        // 6. drive the agent one step so it is "occupied" (Thm occupancy), then report
        agent.step();
        const value = result.kind === "value" ? result.value : result;
        return {
            ok: true,
            agent: newId,
            committedStep: agent.committedStep(),
            residual: agent.residual(),
            allocation: { node: chosen.node.id, slot: chosen.slotIndex, price: chosen.price },
            value,
        };
    }
    /** Broadcast a meta-expression to every node's registry (SRN §6). */
    broadcastMeta(_expr) {
        // In the single-process model every node is reachable; a real transport would
        // forward the trajectory. We report the reachable count (Forest: all peers).
        return { reached: this.nodeSpecs.length };
    }
    // ---- persistence ----
    snapshot() {
        return {
            nodes: this.nodeSpecs,
            coupling: this.coupling,
            agents: [...this.live.values()].map((e) => ({
                id: e.agent.id,
                frame: coordTuple(e.agent.frame),
                committedStep: e.agent.committedStep(),
                residual: e.agent.residual(),
                goal: [...e.agent.standingGoal()],
            })),
        };
    }
    static fromSnapshot(snap) {
        const c = new Cluster({ nodes: snap.nodes, coupling: snap.coupling });
        // restore live agents (committed step is preserved -> monotone across restart)
        for (const a of snap.agents) {
            const id = agentId(a.id);
            const frame = { n: a.frame[0], l: a.frame[1], m: a.frame[2], s: a.frame[3] };
            const agent = new ProcessAgent({ id, frame, loc: a.goal.map(() => 0), goal: a.goal });
            // fast-forward committed step so M does not regress (incorruptibility)
            while (agent.committedStep() < a.committedStep)
                agent.step();
            c.live.set(id, { agent, nodeId: snap.nodes[0]?.id ?? "" });
        }
        return c;
    }
    // ---- internals ----
    candidateNodes(target) {
        const exact = this.nodeSpecs.filter((n) => coordsEqual(this.frameOf(n), target));
        return exact.length > 0 ? exact : this.nodeSpecs.slice();
    }
    frameOf(n) {
        const [nn, l, m, s] = n.frame;
        return { n: nn, l, m, s: s };
    }
    slots() {
        return this.nodeSpecs.map((n) => ({ id: n.id, capacity: n.capacity, maxRate: n.maxRate }));
    }
    payoff() {
        // Each agent's payoff on a slot is inversely related to its residual and the
        // slot's load: closer-to-goal work on a faster slot yields more per tick.
        return (a, slotId) => {
            const entry = this.live.get(a);
            const slot = this.nodeSpecs.find((n) => n.id === slotId);
            if (slot === undefined)
                return 0;
            const r = entry ? Math.max(entry.agent.residual(), 1e-6) : 1;
            return (slot.maxRate * slot.capacity) / r;
        };
    }
    clear() {
        const ags = [...this.live.keys()];
        return clearMarket(ags, this.slots(), this.payoff(), this.tick, this.utilisationCost);
    }
    chooseSlot(newId, candidates) {
        if (candidates.length === 0)
            return null;
        // temporarily include the new agent, clear, and read its assigned slot
        const ags = [...this.live.keys(), newId];
        const candidateSlots = candidates.map((n) => ({ id: n.id, capacity: n.capacity, maxRate: n.maxRate }));
        const payoff = (a, slotId) => {
            const slot = candidates.find((n) => n.id === slotId);
            if (!slot)
                return 0;
            if (a === newId)
                return slot.maxRate * slot.capacity; // new agent: prefer fastest free slot
            const entry = this.live.get(a);
            const r = entry ? Math.max(entry.agent.residual(), 1e-6) : 1;
            return (slot.maxRate * slot.capacity) / r;
        };
        const { assignment, prices } = clearMarket(ags, candidateSlots, payoff, this.tick, this.utilisationCost);
        const nodeId = assignment.get(newId);
        if (nodeId === undefined)
            return null;
        const idx = this.nodeSpecs.findIndex((n) => n.id === nodeId);
        const node = this.nodeSpecs[idx];
        if (node === undefined)
            return null;
        return { node, slotIndex: idx, price: prices.get(nodeId) ?? 0 };
    }
}
/** A default goal target derived from a coordinate (small, tick-scale). */
function defaultGoalFor(c) {
    // a 2-D target a few dozen ticks from the origin, deterministic in the coords
    return [0.02 + 0.001 * c.n, 0.02 + 0.001 * (c.l + 1)];
}
//# sourceMappingURL=cluster.js.map