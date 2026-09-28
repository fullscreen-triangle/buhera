/* ============================================================================
 * tempus module — binds the library adapter (specification
 * specifications/specs/tempus.md) to the vendored stella-lorraine Tempus
 * timing language (compile + seeded simulator + construct/compose surfaces).
 * ========================================================================== */

import { compile } from "@stella-lorraine/tempus";
import { createSimulator } from "@stella-lorraine/tempus/runtime";
import { compileConstruct } from "@stella-lorraine/tempus/construct";
import { compileComposition } from "@stella-lorraine/tempus/composition";
import { makeTempusModule } from "@buhera/registry/modules";

export const tempusEngine = { compile, createSimulator, compileConstruct, compileComposition };
export const tempusModule = makeTempusModule(tempusEngine);
