/* ============================================================================
 * zangalewa-dsl module — the upstream Zangalewa (NL → DSL chunks that
 * compile), reached through the interceptor broker and the user's local
 * `zangalewa connect` agent (specification specifications/specs/zangalewa-dsl.md).
 *
 * Not to be confused with the host-local `zangalewa` module
 * (zangalewa-module.js), which is zoom-climb's research-card coordinate
 * extractor and predates this integration.
 * ========================================================================== */

import { Interceptor } from "@zangalewa/interceptor-client";
import { makeZangalewaModule, DEFAULT_BROKER } from "@buhera/registry/modules";

const BROKER = process.env.NEXT_PUBLIC_ZANGALEWA_BROKER || DEFAULT_BROKER;

export const zangalewaDslModule = makeZangalewaModule(Interceptor, BROKER);
