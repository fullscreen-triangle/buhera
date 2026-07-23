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
// ---- constants ----
export { TICK } from "./agent.js";
// ---- coordinates ----
export { makeCoords, coordsFromTuple, coordTuple, referenceCoords, shell, shellCapacity, coordKey, coordsEqual, isCoordError, } from "./coords.js";
// ---- SRN language ----
export { glyph, composed, literal, evalExpr, serializeExpr, } from "./srn/expr.js";
export { parseSrn, isParseError } from "./srn/parse.js";
export { encodeTrajectory, decodeTrajectory, encodeCoords, decodeCoords, } from "./srn/trajectory.js";
// ---- agents (contract re-exports Agent/AgentId; pylon defines them, since
//      @buhera/musande is not vendored — the interface is musande-compatible) ----
export { ProcessAgent, agentId, FLOOR as AGENT_FLOOR, } from "./agent.js";
// ---- yield market (pure operators, contract §6.4) ----
export { yieldOf, separationCost, clearMarket, forcedUtilisation, defaultUtilisationCost, } from "./market.js";
// ---- single-node runtime ----
export { SrnNode } from "./node.js";
export { Scheduler, Task } from "./scheduler.js";
export { Registry } from "./registry.js";
export { TrajectoryAddress, digestOf, } from "./label.js";
// ---- cluster ----
export { Cluster, } from "./cluster.js";
// ---- Kuramoto phase-lock ----
export { orderParameter, criticalCoupling, KuramotoBank, LOCK_THRESHOLD, } from "./kuramoto.js";
//# sourceMappingURL=index.js.map