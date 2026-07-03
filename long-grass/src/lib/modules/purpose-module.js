/* ============================================================================
 * Purpose Module Adapter
 *
 * Client-side adapter that dispatches to the server-side federation via the
 * /api/purpose-federation route. The API route hosts the JS federation (fs,
 * knowledge packs, HF/Anthropic SDKs); the browser only sees JSON.
 *
 * v1 instruction shapes:
 *   • { kind: "synthesise", description, followups?, field? }
 *   • a plain string — treated as the description with no followups
 *
 * If the server returns ok:false (no provider configured, upstream error,
 * etc.), we surface the message directly in the ActResult so the terminal
 * shows it verbatim.
 * ========================================================================== */

const API_ROUTE = "/api/purpose-federation";

function normaliseInstruction(instruction) {
  if (instruction == null || instruction === "") {
    return { description: "", followups: [], field: null };
  }
  if (typeof instruction === "string") {
    return { description: instruction, followups: [], field: null };
  }
  if (typeof instruction === "object") {
    return {
      description: String(instruction.description || ""),
      followups: Array.isArray(instruction.followups)
        ? instruction.followups.map(String)
        : [],
      field: instruction.field || null,
    };
  }
  return { description: String(instruction), followups: [], field: null };
}

export const purposeModule = {
  id: "purpose",

  describe() {
    return {
      id: "purpose",
      description:
        "Purpose: federated knapsack-allocated cascade over LLMs. " +
        "Takes a description, returns a synthesis document. Requires " +
        "HUGGINGFACE_API_KEY or ANTHROPIC_API_KEY in .env.local.",
      instructions: [
        'dispatch("purpose", "your research description")',
        'dispatch("purpose", { kind: "synthesise", description: "...", followups: ["..."] })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const { description, followups, field } = normaliseInstruction(instruction);
    if (!description.trim()) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: ["purpose: instruction has no description text."],
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
        body: JSON.stringify({ description, followups, field }),
      });
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [
            `purpose: network error contacting ${API_ROUTE}.`,
            err.message || String(err),
          ],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }

    // The API returns JSON on both success and failure — parse either way.
    let body;
    try {
      body = await res.json();
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`purpose: non-JSON response (HTTP ${res.status})`],
        },
        residue: 0,
        completed: true,
        error: `non-JSON response (HTTP ${res.status})`,
      };
    }

    if (!body.ok) {
      const lines = ["purpose: " + (body.error || `HTTP ${res.status}`)];
      if (body.stage === "provider") {
        lines.push("set HUGGINGFACE_API_KEY or ANTHROPIC_API_KEY in .env.local");
      }
      return {
        ok: false,
        output_delta: { kind: "text", lines },
        residue: 0,
        completed: true,
        error: body.error || `HTTP ${res.status}`,
      };
    }

    return {
      ok: true,
      output_delta: {
        kind: "purpose_synthesis",
        synthesis: body.synthesis,
        model: body.model,
        federation: body.federation,
        floor: body.floor,
        packs_used: body.packs_used,
      },
      residue: typeof body.floor === "number" ? body.floor : 0,
      completed: true,
    };
  },

  outputCell(_instruction) {
    return { kind: "synthesis_cell" };
  },
};
