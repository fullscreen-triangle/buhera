// shapeshifter — lavoisier's Shapeshifter (.ss) interpreter, bound to the
// vendored web/src/lib (vendor/shapeshifter, byte-exact). Specification:
// specs/shapeshifter.md. The adapter (@buhera/registry) reports a global
// runtime failure as ok:false, and counts warnings and unresolved db.*
// lookups as residue.
import { compileStage, executeStage } from "@lavoisier/shapeshifter";
import { makeShapeshifterModule } from "@buhera/registry/modules";

export const shapeshifterEngine = { compileStage, executeStage };
export const shapeshifterModule = makeShapeshifterModule(shapeshifterEngine);
