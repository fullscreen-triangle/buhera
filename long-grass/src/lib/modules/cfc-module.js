// cfc — cause-for-concern, bound to the vendored syndrome engine
// (vendor/cfc, byte-exact). Specification: specs/cfc.md.
import { parse } from "@syndrome/cfc/parser";
import { runSource } from "@syndrome/cfc";
import { EXAMPLES, DEFAULT_FILE } from "@syndrome/cfc/examples";
import { makeCfcModule } from "@buhera/registry/modules";

export const cfcEngine = { parse, runSource, EXAMPLES, DEFAULT_FILE };
export const cfcModule = makeCfcModule(cfcEngine);
