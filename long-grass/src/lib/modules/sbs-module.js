/* ============================================================================
 * SBS module — binds the library adapter (@buhera/registry, specification
 * specifications/specs/sbs.md) to the vendored engine @sachikonye/sbs.
 *
 * The adapter logic (instruction shapes, the sbs_result delta the
 * MetricsDashboard renders, residue = 1 − V, warnings for the engine's known
 * silent behaviours) lives in buhera-os/registry-ts/src/modules/sbs.ts.
 * ========================================================================== */

import * as sbsEngine from "@sachikonye/sbs";
import { makeSbsModule } from "@buhera/registry/modules";

export const sbsModule = makeSbsModule(sbsEngine);
