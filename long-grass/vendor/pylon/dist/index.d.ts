/**
 * @buhera/pylon — distributed resource-allocation runtime for Buhera OS.
 *
 * Speaks Sango Rine Shumba (SRN) as its transmission unit, clears a
 * compute-tick-bounded yield market, and instantiates each allocated task as a
 * persistent process agent with goal succession (network-yield §9).
 *
 * Public surface (contract §6.5). TS-native reimplementation; the Rust
 * `crates/srn-node` is the reference the semantics are ported from.
 */
export { TICK } from "./agent.js";
export { makeCoords, coordsFromTuple, coordTuple, referenceCoords, shell, shellCapacity, coordKey, coordsEqual, isCoordError, type PartitionCoords, type Coord, type CoordError, } from "./coords.js";
export { glyph, composed, literal, evalExpr, serializeExpr, type Expr, type Glyph, type ComposedExpr, type LiteralExpr, type Operator, type EvalResult, type ReceiverFrame, type Env, } from "./srn/expr.js";
export { parseSrn, isParseError, type ParseError } from "./srn/parse.js";
export { encodeTrajectory, decodeTrajectory, encodeCoords, decodeCoords, type Trajectory, } from "./srn/trajectory.js";
export { ProcessAgent, agentId, FLOOR as AGENT_FLOOR, type Agent, type AgentId, type Target, type Occupation, type Phase, type AgentState, } from "./agent.js";
export { yieldOf, separationCost, clearMarket, forcedUtilisation, defaultUtilisationCost, type Slot, type Price, type Assignment, type PayoffFn, type UtilisationCost, } from "./market.js";
export { SrnNode, type NodeConfig } from "./node.js";
export { Scheduler, Task, type TaskState, type TickResult } from "./scheduler.js";
export { Registry, type RegistryEntry } from "./registry.js";
export { TrajectoryAddress, digestOf, type ContentDigest, type ProcessLabel, type EvalRecord, } from "./label.js";
export { Cluster, type ClusterConfig, type ClusterSnapshot, type NodeSpec, type Yield, } from "./cluster.js";
export { orderParameter, criticalCoupling, KuramotoBank, LOCK_THRESHOLD, } from "./kuramoto.js";
export type { PylonError } from "./errors.js";
//# sourceMappingURL=index.d.ts.map