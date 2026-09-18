/* ============================================================================
 * HFQ Module Adapter
 *
 * Wraps @hegel/hfq — the real hegel federated query interpreter — as a Buhera
 * module. This is not a reimplementation: the vendored engine (parser, model,
 * adapters, check, allocate, execute) is the same code that runs in the
 * hegel/consequences notebook and the honjo-masamune federated-query page,
 * copied byte-for-byte into vendor/hfq. This adapter's only job is to run a
 * plan and translate the result into an ActResult.
 *
 * A plan names sources abstractly and declares a budget; the engine runs the
 * real pipeline (parse -> resolve -> check -> allocate -> execute -> emit)
 * against one of four local fixture "worlds" and returns a six-verdict
 * document: answer/empty/surface/timeout/refused/starved, each with a named
 * blocker (model/engine/budget/corpus) where one applies. `empty` and
 * `answer` carry NO blocker — that absence is the point (def:blocker).
 *
 * The `biocat` world is the one built around Mark Doerr's five real
 * questions (mark_q1-mark_q5, see plans.js) plus eight generic Chem-DCAT-AP
 * queries (dcat_g1-dcat_g8) that exercise the same engine against generic
 * dataset-catalog-shaped questions — the natural basis for the LinkML
 * comparison: a shape-only (LinkML/SPARQL-style) view of these questions
 * reports only non-empty/empty; the six-verdict engine additionally reports
 * WHERE and WHY a step failed to answer.
 *
 * Instruction shapes:
 *   • a plain string             — treated as plan source text
 *   • { kind: "run", source }    — same, explicit shape
 *   • { kind: "preset", id }     — run one of the 24 preset plans by id
 *   • { kind: "list_presets" }   — list preset plan ids/sections/blurbs
 *   • "demo" / ""                — the paper's own worked example
 * ========================================================================== */

import runPlan from "@hegel/hfq";
import { PLANS, SECTIONS, byId } from "@hegel/hfq/plans";

const DEMO_SOURCE = `plan healthy_chain {
  budget 200 requests
  let acids = from chebi ask descendants_of("CHEBI:1") within 10
  let kegg  = map acids via chebi2kegg expect partial 0.6
  let rxns  = from rhea ask reactions_consuming(?c) with ?c in kegg within 60
  emit rxns
}`;

function resolveSource(instruction) {
  if (instruction == null || instruction === "" || instruction === "demo") {
    return DEMO_SOURCE;
  }
  if (typeof instruction === "string") return instruction;
  if (typeof instruction === "object") {
    if (instruction.kind === "run" && typeof instruction.source === "string") {
      return instruction.source;
    }
    if (instruction.kind === "preset" && instruction.id) {
      const plan = byId(instruction.id);
      return plan ? plan.source : null;
    }
  }
  return null;
}

/** One-line summary of a run: verdict tally, for the audit log and text fallback. */
function summarize(result) {
  if (!result.ok) return `HFQ: refused/failed at stage "${result.stage}" — ${result.error}`;
  const tally = {};
  for (const s of result.steps) tally[s.verdict] = (tally[s.verdict] || 0) + 1;
  const parts = Object.entries(tally).map(([v, n]) => `${n} ${v}`);
  return `HFQ: ${result.plan} [${result.world}] — ${parts.join(", ")}, ${result.requests_issued} request(s)`;
}

export const hfqModule = {
  id: "hfq",

  describe() {
    return {
      id: "hfq",
      description:
        "HFQ — the real hegel federated query interpreter, vendored unmodified. " +
        "Runs a plan (parse -> resolve -> check -> allocate -> execute -> emit) " +
        "against a local fixture world and returns a six-verdict document " +
        "(answer/empty/surface/timeout/refused/starved), each with a named " +
        "blocker where one applies. Includes Mark Doerr's five real " +
        "biocatalysis questions (mark_q1-mark_q5) and eight generic " +
        "Chem-DCAT-AP queries (dcat_g1-dcat_g8).",
      instructions: [
        'dispatch("hfq", "demo")',
        'dispatch("hfq", { kind: "run", source: "plan p { budget 10 requests ... }" })',
        'dispatch("hfq", { kind: "preset", id: "mark_q1" })',
        'dispatch("hfq", { kind: "list_presets" })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const instr = typeof instruction === "object" && instruction !== null ? instruction : {};

    if (instr.kind === "list_presets") {
      return {
        ok: true,
        output_delta: {
          kind: "hfq_presets",
          sections: Object.keys(SECTIONS),
          plans: PLANS.map((p) => ({ id: p.id, section: p.section, blurb: p.blurb })),
        },
        residue: PLANS.length,
        completed: true,
      };
    }

    const source = resolveSource(instruction);
    if (source == null) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [
            'hfq: instruction must be plan source text, "demo", ' +
              '{ kind: "run", source }, { kind: "preset", id }, or { kind: "list_presets" }',
          ],
        },
        residue: 0,
        completed: true,
        error: "invalid instruction",
      };
    }

    let result;
    try {
      result = runPlan(source);
    } catch (err) {
      // runPlan is documented to never throw; this is defence in depth so no
      // unexpected exception escapes into the registry's generic error path.
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`hfq: unexpected error — ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }

    if (!result.ok) {
      return {
        ok: false,
        output_delta: {
          kind: "hfq_result",
          summary: summarize(result),
          result,
        },
        residue: 0,
        completed: true,
        error: result.error,
      };
    }

    return {
      ok: true,
      output_delta: {
        kind: "hfq_result",
        summary: summarize(result),
        result,
      },
      residue: result.steps.length,
      completed: true,
    };
  },

  outputCell(_instruction) {
    return { kind: "hfq_cell" };
  },
};
