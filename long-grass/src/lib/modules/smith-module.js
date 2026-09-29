/* ============================================================================
 * Smith module — Agent Smith (split-attention synchronised agents).
 *
 * Binds the library adapter (@buhera/registry, specification
 * specifications/specs/smith.md) to musande's CANONICAL engine, vendored at
 * vendor/agent-smith: the recursive-descent parser, the typechecker that
 * enforces the paper's rules (connected self-graph, costs ≥ floor, strongly
 * convex potentials), and the town runtime.
 *
 * This replaces src/lib/smith, which was a port of smith-ide's regex stub —
 * a different, permissive dialect with no typechecker. Runs are always
 * deterministic with models off; the output keeps the agent_generated shape
 * the ArtifactSmith renderer reads.
 * ========================================================================== */

import { build } from "@musande/agent-smith/compile";
import { makeTown, defaultCtx, runTown } from "@musande/agent-smith/town";
import { makeSmithModule } from "@buhera/registry/modules";

export const smithEngine = { build, makeTown, defaultCtx, runTown };
export const smithModule = makeSmithModule(smithEngine);
