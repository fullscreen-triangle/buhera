// honjo — Honjo Masamune, bound to the vendored borgia bundle (vendor/honjo,
// byte-exact). Specification: specs/honjo.md.
import { compile, evaluate, deriveAtom, renderValue } from "@borgia/honjo";
import { makeHonjoModule } from "@buhera/registry/modules";

export const honjoEngine = { compile, evaluate, deriveAtom, renderValue };
export const honjoModule = makeHonjoModule(honjoEngine);
