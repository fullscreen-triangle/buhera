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
/** Default utilisation cost g_u(v) = v^2 (strictly convex, g_u(0)=0). */
export const defaultUtilisationCost = (v) => v * v;
/**
 * Network transport yield of an assignment (network-yield Def. 3.4), unit-slot
 * form: numerator is total payoff; denominator is total resource consumption in
 * compute-ticks weighted by utilisation cost. With one occupied slot per agent
 * and a fixed rate, the denominator is a positive constant across full
 * assignments, so ranking by yield ranks by total payoff — which is what the
 * Three-way Equivalence reasons about.
 */
export function yieldOf(assignment, agents, slots, payoff, tick, utilisationCost = defaultUtilisationCost) {
    let numerator = 0;
    for (const a of agents) {
        const s = assignment.get(a);
        if (s !== undefined)
            numerator += payoff(a, s);
    }
    // denominator: sum over used slots of tick * capacity * g_u(rate). Rate is the
    // slot's occupancy fraction; here we use unit rate for the ranking model.
    const used = new Set();
    for (const a of agents) {
        const s = assignment.get(a);
        if (s !== undefined)
            used.add(s);
    }
    let denom = 0;
    for (const slot of slots) {
        if (used.has(slot.id))
            denom += tick * slot.capacity * utilisationCost(1);
    }
    if (denom <= 0)
        return 0;
    return numerator / denom;
}
/** Total payoff of an assignment (the yield numerator; denominator-invariant). */
function payoffSum(assignment, agents, payoff) {
    let s = 0;
    for (const a of agents) {
        const slot = assignment.get(a);
        if (slot !== undefined)
            s += payoff(a, slot);
    }
    return s;
}
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
export function separationCost(slotId, assignment, agents, slots, payoff, _tick, _utilisationCost = defaultUtilisationCost) {
    const base = payoffSum(assignment, agents, payoff);
    const remaining = slots.filter((s) => s.id !== slotId);
    if (remaining.length === 0)
        return base; // removing the only slot costs all payoff
    // reassign every agent currently on e to its best remaining slot
    const realloc = new Map(assignment);
    for (const a of agents) {
        if (assignment.get(a) === slotId) {
            let best = remaining[0].id;
            let bestPay = payoff(a, best);
            for (const s of remaining) {
                const p = payoff(a, s.id);
                if (p > bestPay) {
                    bestPay = p;
                    best = s.id;
                }
            }
            realloc.set(a, best);
        }
    }
    const after = payoffSum(realloc, agents, payoff);
    return Math.max(0, base - after);
}
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
export function clearMarket(agents, slots, payoff, tick, utilisationCost = defaultUtilisationCost) {
    // Initial assignment: a VALID injective placement (agent i -> slot i). Unit
    // capacity means one agent per slot; we keep this invariant throughout by only
    // ever SWAPPING two agents' slots (a permutation move), never stacking.
    const assign = new Map();
    for (let i = 0; i < agents.length; i++) {
        assign.set(agents[i], slots[i % slots.length].id);
    }
    const y = () => yieldOf(assign, agents, slots, payoff, tick, utilisationCost);
    // Greedy ascent to closure over the swap neighbourhood. Each move swaps the
    // slots of two agents (or moves one agent to a genuinely empty slot), so the
    // assignment stays a valid unit-capacity placement at every step.
    const occupantOf = (slotId) => {
        for (const b of agents)
            if (assign.get(b) === slotId)
                return b;
        return undefined;
    };
    let improved = true;
    let guard = 0;
    const maxIters = agents.length * slots.length * 4 + 16;
    while (improved && guard++ < maxIters) {
        improved = false;
        const base = y();
        let bestGain = tick; // only moves beating the floor count (deterministic closure)
        let bestMove = null;
        for (const a of agents) {
            const cur = assign.get(a);
            for (const s of slots) {
                if (s.id === cur)
                    continue;
                const occupant = occupantOf(s.id); // undefined => s is empty (move), else swap
                const prevA = cur;
                assign.set(a, s.id);
                let prevB;
                if (occupant !== undefined && occupant !== a) {
                    prevB = prevA;
                    assign.set(occupant, prevA); // occupant takes a's old slot
                }
                const gain = y() - base;
                // revert
                assign.set(a, prevA);
                if (occupant !== undefined && occupant !== a && prevB !== undefined) {
                    assign.set(occupant, s.id);
                }
                if (gain > bestGain) {
                    bestGain = gain;
                    bestMove = occupant !== undefined && occupant !== a
                        ? { agent: a, slot: s.id, swapWith: occupant }
                        : { agent: a, slot: s.id };
                }
            }
        }
        if (bestMove) {
            const prevA = assign.get(bestMove.agent);
            assign.set(bestMove.agent, bestMove.slot);
            if (bestMove.swapWith !== undefined)
                assign.set(bestMove.swapWith, prevA);
            improved = true;
        }
    }
    const prices = new Map();
    for (const s of slots) {
        prices.set(s.id, separationCost(s.id, assign, agents, slots, payoff, tick, utilisationCost));
    }
    return { assignment: assign, prices };
}
/**
 * Forced optimal utilisation (network-yield Thm 7.x, corrected net-yield form).
 *
 * At closure a slot runs at v* maximising net yield P*b(v) - tick*c*g_u(v), with
 * b strictly concave (diminishing returns) and g_u strictly convex. This returns
 * the unique interior v* by 1-D search — the marginal-balance point
 * P*b'(v*) = tick*c*g_u'(v*).
 */
export function forcedUtilisation(args) {
    const P = args.pressure;
    const c = args.capacity;
    const vbar = args.maxRate;
    const b = args.benefit ?? ((v) => Math.log(1 + v));
    const gu = args.utilisationCost ?? ((v) => (v * v) / (1 - v / vbar));
    const steps = args.steps ?? 20000;
    let bestV = 0;
    let bestY = -Infinity;
    const lo = 1e-4;
    const hi = vbar - 1e-4;
    for (let i = 0; i <= steps; i++) {
        const v = lo + ((hi - lo) * i) / steps;
        const Y = P * b(v) - args.tick * c * gu(v);
        if (Y > bestY) {
            bestY = Y;
            bestV = v;
        }
    }
    return bestV;
}
//# sourceMappingURL=market.js.map