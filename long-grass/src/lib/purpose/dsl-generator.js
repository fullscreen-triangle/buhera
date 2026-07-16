/**
 * NL-instructions -> validated DSL code.
 *
 * This is the equalizer between the two Buhera user types. An informed user
 * hand-writes a DSL script; an uninformed user writes natural-language
 * instructions. `generateDsl` compiles those instructions into a script that
 * the module's OWN compiler accepts — so both users reach the identical
 * dispatch(moduleId, source) executor.
 *
 * The loop:
 *   1. GROUND    — inject the DSL's knowledge pack (examples + grammar) into
 *                  the system prompt. (Examples, not facts: empty-dictionary.)
 *   2. GENERATE  — run the FKAC federation (N parallel drafts + integration),
 *                  forcing a { code } schema so we get a clean script.
 *   3. VALIDATE  — run the candidate through the DSL's REAL compiler.
 *   4. REPAIR    — on failure, feed the compiler's errors back and regenerate,
 *                  up to maxRepairs. The compiler is the judge, every round.
 *
 * Returns the validated code plus the federation confidence and a repair
 * trail. It does NOT execute the code — dispatch is the caller's choice
 * (see /api/dsl-generate).
 */

import { buildPackContext, selectPacks } from "@/lib/purpose/knowledge-packs";
import { getDsl } from "@/lib/purpose/dsl/validators";
import { runFederation } from "@/lib/purpose/federation-orchestrator";

/** JSON schema forcing the model to emit only the DSL source. */
const CODE_SCHEMA = {
  type: "object",
  properties: {
    code: {
      type: "string",
      description: "The DSL source code, and nothing else.",
    },
  },
  required: ["code"],
};

/**
 * @param {object}   args
 * @param {string}   args.dslId          which DSL to target (e.g. "vahera")
 * @param {string}   args.instructions   the uninformed user's NL instructions
 * @param {number}   [args.maxRepairs]   repair attempts after the first (default 3)
 * @param {object}   [args.federationOpts] passthrough to runFederation (providers,
 *                                          draftFn, integrateFn — used by tests)
 * @returns {Promise<{ok, code?, dslId, federation?, repairs, attempts, errors?}>}
 */
export async function generateDsl({ dslId, instructions, maxRepairs = 3, federationOpts = {} }) {
  const dsl = getDsl(dslId);
  if (!dsl) {
    return { ok: false, dslId, repairs: 0, attempts: 0, errors: [{ message: `unknown DSL: "${dslId}"` }] };
  }
  if (!instructions || !instructions.trim()) {
    return { ok: false, dslId, repairs: 0, attempts: 0, errors: [{ message: "instructions are empty" }] };
  }

  // ── 1. Ground ────────────────────────────────────────────────────────────
  // Prefer the DSL's own pack; selectPacks also picks up any other packs the
  // instructions happen to trigger.
  const packIds = uniquePacks([dsl.packId, ...selectPacks(`${dslId} ${instructions}`)]);
  const grounding = buildPackContext(packIds);
  const system = buildSystemPrompt(dsl.label, grounding);

  const attempts = [];
  let lastErrors = [];

  // ── 2-4. Generate -> validate -> repair ──────────────────────────────────
  for (let round = 0; round <= maxRepairs; round++) {
    const user =
      round === 0
        ? buildInitialUser(instructions)
        : buildRepairUser(instructions, attempts[attempts.length - 1]);

    const fed = await runFederation({
      system,
      user,
      jsonSchema: CODE_SCHEMA,
      ...federationOpts,
    });

    if (!fed.ok) {
      // Generation itself failed (no provider, all drafts dead) — not a
      // validation failure, so no point repairing. Return immediately.
      return {
        ok: false,
        dslId,
        repairs: round,
        attempts: attempts.length,
        federation: fed.federation,
        errors: [{ message: fed.error || "generation failed", stage: fed.stage }],
      };
    }

    const code = fed.content;
    const check = dsl.validate(code);
    attempts.push({ round, code, ok: check.ok, errors: check.errors });

    if (check.ok) {
      return {
        ok: true,
        dslId,
        code,
        federation: fed.federation,
        repairs: round,
        attempts: attempts.length,
      };
    }
    lastErrors = check.errors;
  }

  // Exhausted repairs — return the last (invalid) attempt for diagnostics.
  const last = attempts[attempts.length - 1];
  return {
    ok: false,
    dslId,
    code: last ? last.code : null,
    repairs: maxRepairs,
    attempts: attempts.length,
    errors: lastErrors,
  };
}

/** Deduplicate pack ids, dropping falsy entries, preserving order. */
function uniquePacks(ids) {
  const seen = new Set();
  const out = [];
  for (const id of ids) {
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }
  return out;
}

function buildSystemPrompt(label, grounding) {
  const parts = [
    `You write ${label} DSL code. Given a user's natural-language instructions,`,
    `produce a single valid ${label} script that carries them out.`,
    "",
    "Rules:",
    `- Output ONLY ${label} source code, no prose, no explanation, no fences.`,
    "- Follow the grammar in the reference material exactly; invalid syntax is rejected.",
    "- If the instructions are ambiguous, choose the simplest faithful interpretation.",
  ];
  if (grounding) {
    parts.push("", grounding);
  }
  return parts.join("\n");
}

function buildInitialUser(instructions) {
  return `Instructions:\n${instructions}`;
}

/**
 * Repair prompt: show the model the exact code it produced and the compiler's
 * verbatim errors, and ask for a corrected script. Line-anchored errors point
 * the model at the offending statement.
 */
function buildRepairUser(instructions, lastAttempt) {
  const errorLines = (lastAttempt.errors || [])
    .map((e) => (e.line != null ? `- line ${e.line}: ${e.message}` : `- ${e.message}`))
    .join("\n");
  return [
    `Instructions:\n${instructions}`,
    "",
    "Your previous script was REJECTED by the compiler:",
    "",
    lastAttempt.code,
    "",
    "Compiler errors:",
    errorLines,
    "",
    "Return a corrected script that fixes these errors and still fulfils the instructions.",
  ].join("\n");
}
