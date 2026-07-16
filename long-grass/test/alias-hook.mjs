// Registers the resolve hook (resolver.mjs) so `node --test` can resolve this
// repo's bundler-only import conventions (`@/` alias + extensionless imports).
//
// Usage: node --import ./test/alias-hook.mjs --test test/*.test.mjs
import { register } from "node:module";
register("./resolver.mjs", import.meta.url);
