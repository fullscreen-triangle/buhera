// synopsis — gospel's genomic scripting language, bound to the vendored
// front end (vendor/synopsis, byte-exact). Specification: specs/synopsis.md.
// Checks, parses and tokenises; upstream has no evaluator, so it never runs.
import * as synopsisEngine from "@gospel/synopsis";
import { makeSynopsisModule } from "@buhera/registry/modules";

export { synopsisEngine };
export const synopsisModule = makeSynopsisModule(synopsisEngine);
