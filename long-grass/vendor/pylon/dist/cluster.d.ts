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
import { type Coord } from "./coords.js";
import { type Glyph } from "./srn/expr.js";
import { type Agent, type AgentId, type Occupation, type Target } from "./agent.js";
import { type Price } from "./market.js";
import type { PylonError } from "./errors.js";
/** A cluster node (execution slot) described by its receiver frame + capacities. */
export interface NodeSpec {
    readonly id: string;
    readonly frame: Coord;
    readonly capacity: number;
    readonly taskDuration: number;
    readonly maxRate: number;
}
/** Configuration for a Cluster. */
export interface ClusterConfig {
    readonly nodes: ReadonlyArray<NodeSpec>;
    /** Utilisation cost g_u; must satisfy g_u(0)=0, g_u'>0, g_u''>0. */
    readonly utilisationCost?: (v: number) => number;
    /** Kuramoto coupling K for scheduler phase-lock. Should exceed K_c* = 2 sigma/pi. */
    readonly coupling?: number;
}
/** The result of a submitted expression (contract §6.1). */
export type Yield = {
    ok: true;
    agent: AgentId;
    committedStep: number;
    residual: number;
    allocation: {
        node: string;
        slot: number;
        price: Price;
    };
    /** Receiver-relative result from the chosen node's evaluation. */
    value: unknown;
} | {
    ok: false;
    error: PylonError;
};
/** An opaque snapshot of cluster assignment state. */
export interface ClusterSnapshot {
    readonly nodes: ReadonlyArray<NodeSpec>;
    readonly coupling: number;
    readonly agents: ReadonlyArray<{
        id: string;
        frame: Coord;
        committedStep: number;
        residual: number;
        goal: Target;
    }>;
}
export declare class Cluster {
    readonly tick = 0.001;
    private readonly nodeSpecs;
    private readonly utilisationCost;
    private readonly coupling;
    private readonly kuramoto;
    private readonly live;
    private counter;
    constructor(config: ClusterConfig);
    nodes(): ReadonlyArray<NodeSpec>;
    /** Add a node. Forest openness: no enrolment; the node is a peer at once. */
    addNode(spec: NodeSpec): void;
    price(nodeId: string): Price;
    /** Kuramoto order parameter (R, psi). */
    orderParameter(): {
        R: number;
        psi: number;
    };
    isPhaseLocked(): boolean;
    criticalCoupling(sigmaOmega: number): number;
    liveAgents(): ReadonlyArray<Agent>;
    agent(id: AgentId): Agent | null;
    /**
     * Submit an SRN expression for evaluation somewhere in the cluster. Selects a
     * receiver by the expression's target region and the current market clearing,
     * instantiates a persistent ProcessAgent, evaluates receiver-relatively, and
     * returns a Yield. A succession map makes the agent persist past completion.
     */
    submit(expr: string | Glyph, options?: {
        succession?: Occupation;
        timeoutTicks?: number;
        goal?: Target;
    }): Yield;
    /** Broadcast a meta-expression to every node's registry (SRN §6). */
    broadcastMeta(_expr: string | Glyph): {
        reached: number;
    };
    snapshot(): ClusterSnapshot;
    static fromSnapshot(snap: ClusterSnapshot): Cluster;
    private candidateNodes;
    private frameOf;
    private slots;
    private payoff;
    private clear;
    private chooseSlot;
}
//# sourceMappingURL=cluster.d.ts.map