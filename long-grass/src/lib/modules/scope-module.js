// scope — helicopter's SCOPE microscopy language, bound to the vendored
// scope-lang (vendor/scope-lang, byte-exact). Specification: specs/scope.md.
//
// SCOPE runs as a REPL: cells accumulate into one program in the module's
// session. The terminal's `:scope load` decodes an image and links it here.
import { compile, createSession } from "scope-lang";
import { makeScopeModule } from "@buhera/registry/modules";

export const scopeEngine = { compile, createSession };
export const scopeModule = makeScopeModule(scopeEngine);

/** Link a decoded image ({ data, width, height }) into the live session. */
export function linkScopeImage(imagePayload) {
  scopeModule.linkImage(imagePayload);
}

/** Replace the live session. */
export function resetScopeSession() {
  scopeModule.resetSession();
}
