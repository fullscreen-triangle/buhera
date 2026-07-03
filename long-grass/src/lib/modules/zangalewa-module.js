/* ============================================================================
 * Zangalewa Module Adapter
 *
 * Zangalewa is the Minimum Sufficient Interceptor: it turns a natural-language
 * utterance into an S-coord + a fully-rendered research card in one call.
 *
 * The client-side adapter posts to /api/extract; the server route (extract.js)
 * calls OpenAI's structured-output API and returns a RenderResult. Requires
 * OPENAI_API_KEY in .env.local.
 *
 * Instruction shapes:
 *   • plain string           — treated as the utterance
 *   • { utterance: string }  — explicit shape
 * ========================================================================== */

const API_ROUTE = "/api/extract";

function normaliseUtterance(instruction) {
  if (typeof instruction === "string") return instruction;
  if (instruction && typeof instruction === "object") {
    if (typeof instruction.utterance === "string") return instruction.utterance;
    if (typeof instruction.query === "string") return instruction.query;
  }
  return "";
}

export const zangalewaModule = {
  id: "zangalewa",

  describe() {
    return {
      id: "zangalewa",
      description:
        "Zangalewa: Minimum Sufficient Interceptor. Turns a natural-language " +
        "utterance into an S-coord and a research card. Requires " +
        "OPENAI_API_KEY in .env.local.",
      instructions: [
        'dispatch("zangalewa", "your natural-language query")',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const utterance = normaliseUtterance(instruction).trim();
    if (!utterance) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: ["zangalewa: instruction has no utterance."],
        },
        residue: 0,
        completed: true,
      };
    }

    let res;
    try {
      res = await fetch(API_ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ utterance }),
      });
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [
            `zangalewa: network error contacting ${API_ROUTE}.`,
            err.message || String(err),
          ],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }

    let body;
    try {
      body = await res.json();
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`zangalewa: non-JSON response (HTTP ${res.status})`],
        },
        residue: 0,
        completed: true,
        error: `non-JSON response (HTTP ${res.status})`,
      };
    }

    if (!res.ok) {
      const lines = ["zangalewa: " + (body.error || `HTTP ${res.status}`)];
      if (body.stage === "provider") {
        lines.push("set OPENAI_API_KEY in .env.local");
      }
      return {
        ok: false,
        output_delta: { kind: "text", lines },
        residue: 0,
        completed: true,
        error: body.error || `HTTP ${res.status}`,
      };
    }

    // Success: body is a RenderResult { caption, leaves: [{ leaf, coord, params }] }
    const leaves = Array.isArray(body.leaves) ? body.leaves : [];
    const primary = leaves[0] || null;

    return {
      ok: true,
      output_delta: {
        kind: "zangalewa_render",
        caption: body.caption || "",
        leaves,
        // Convenience: surface the primary leaf's coord at top level so
        // turbulance scripts can grab it without index gymnastics.
        coord: primary?.coord ?? null,
        title: primary?.params?.title ?? null,
      },
      // Residue = 1 - S_e proxy: a well-settled lookup (low S_e) is
      // "close to done"; a frontier query (high S_e) needs more work.
      residue: primary?.coord?.S_e ?? 1,
      completed: true,
    };
  },

  outputCell(_instruction) {
    return { kind: "coord_cell" };
  },
};
