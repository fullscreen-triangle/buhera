// @buhera/spraypaint — TypeScript bindings + crossfilter-to-query propagation
// for the spraypaint split-attention search binary.
//
// The core (types, crossfilter, undo, session, client) imports no Node built-ins
// and is browser-safe. runner-node.ts is the only Node-coupled module; import it
// explicitly server-side.

export {
  type AskQuery,
  type AskHit,
  type AskResult,
  type SceneAllocation,
  type Identity,
  type CountResult,
  type SceneInfo,
  type InvariantCheck,
  type VerifyResult,
  DEFAULT_QUERY,
  queryToArgs,
  queryToDisplay,
} from "./types.js";

export {
  SpraypaintClient,
  SpraypaintError,
  type SpraypaintRunner,
  type RunOutput,
  type SpraypaintClientOptions,
} from "./client.js";

export {
  applyDiff,
  invertSceneToggle,
  invertPriceDrag,
  invertBudgetStep,
  invertFlatToggle,
  editQueryText,
  setBudget,
  setScenes,
  type QueryDiff,
  type QuerySource,
} from "./crossfilter.js";

export { QueryHistory, type QuerySnapshot } from "./undo.js";

export {
  SpraypaintSession,
  type SessionState,
  type Clock,
} from "./session.js";

// Node runner is exported from a subpath to keep the barrel browser-safe.
// Import as: `import { NodeRunner } from "@buhera/spraypaint/runner-node"`.
