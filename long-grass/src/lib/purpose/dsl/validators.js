/**
 * DSL validator adapters.
 *
 * Each Buhera module carries its own DSL with its own compiler. Those
 * compilers disagree on their return shape: some return {valid, errors},
 * some {ok, errors}, some throw, some return a single {error}. This module
 * normalizes each to ONE contract so the generate->validate->repair loop
 * (dsl-generator.js) stays DSL-agnostic:
 *
 *     validate(source) -> { ok: boolean, errors: [{ message, line? }] }
 *
 * The validator is the empty-dictionary ground truth: purpose ships no DSL
 * facts, it only proposes code and lets the DSL's REAL compiler judge it.
 * A generated script is "valid" iff the module's own compiler accepts it.
 *
 * The DSL registry also records, per DSL, which module executes the code
 * (moduleId, for dispatch) and which knowledge pack grounds generation
 * (packId). Adding a DSL is one registry entry plus its normalizer.
 */

import { parseVahera } from "@/lib/vahera";

/**
 * Pull a 1-based line number out of a thrown parser message of the form
 * "line N: ...". Returns null when the message carries no line marker.
 */
function lineFromMessage(message) {
  const m = /(?:^|\b)line\s+(\d+)\b/i.exec(message || "");
  return m ? parseInt(m[1], 10) : null;
}

/**
 * vaHera — `parseVahera(src)` THROWS on the first invalid line, embedding
 * "line N:" in the message. Parsing is pure (no kernel needed), so it is a
 * safe pre-flight check before we hand the source to dispatch("vahera").
 */
export function validateVahera(source) {
  try {
    parseVahera(source);
    return { ok: true, errors: [] };
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    return { ok: false, errors: [{ message, line: lineFromMessage(message) }] };
  }
}

/**
 * The DSL registry. Each entry:
 *   validate : (source) => { ok, errors }
 *   moduleId : the registry module id that executes the generated code
 *   packId   : the knowledge pack that grounds generation
 *   label    : human-readable name
 *
 * vaHera is the first (and hardest) target: it validates by throwing and
 * executes through a stateful kernel owned by the vahera module. SBS,
 * Turbulance, and SCOPE (validate-only) are added here as they come online.
 */
export const DSL_REGISTRY = {
  vahera: {
    label: "vaHera",
    validate: validateVahera,
    moduleId: "vahera",
    packId: "vahera",
  },
};

/** List the DSL ids this generator can currently target. */
export function listDsls() {
  return Object.keys(DSL_REGISTRY);
}

/** Look up a DSL entry, or null if unknown. */
export function getDsl(dslId) {
  return Object.prototype.hasOwnProperty.call(DSL_REGISTRY, dslId)
    ? DSL_REGISTRY[dslId]
    : null;
}

/**
 * Validate `source` against the named DSL's real compiler. Throws if the
 * DSL id is unknown (a programming error, distinct from invalid source
 * which returns {ok:false}).
 */
export function validate(dslId, source) {
  const dsl = getDsl(dslId);
  if (!dsl) throw new Error(`unknown DSL: "${dslId}"`);
  return dsl.validate(source);
}
