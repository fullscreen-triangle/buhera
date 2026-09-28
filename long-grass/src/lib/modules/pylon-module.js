/* ============================================================================
 * pylon module — binds the library adapter (specification
 * specifications/specs/pylon.md) to the vendored @buhera/pylon: SRN glyphs,
 * the separation-cost yield market and persistent process agents. The module
 * owns one cluster for the browser session.
 * ========================================================================== */

import * as pylonEngine from "@buhera/pylon";
import { makePylonModule } from "@buhera/registry/modules";

export const pylonModule = makePylonModule(pylonEngine);
