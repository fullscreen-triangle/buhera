// sthurbert — st-Hurbert, bloodhound's repo-query language, bound to the
// vendored thrust repo-lens (vendor/sthurbert, byte-exact). Specification:
// specs/sthurbert.md.
import { compile, run } from "@bloodhound/sthurbert";
import { computeCharacter } from "@bloodhound/sthurbert/chi";
import { makeSthurbertModule } from "@buhera/registry/modules";

export const sthurbertEngine = { compile, run, computeCharacter };
export const sthurbertModule = makeSthurbertModule(sthurbertEngine);
