/* ============================================================================
 * Zangalewa Module Adapter
 *
 * Zangalewa is the Minimum Sufficient Interceptor — it takes a natural-language
 * utterance and produces a structured coordinate that the orchestrator uses to
 * route the request.
 *
 * The library at src/lib/zangalewa/ ships a client-side wrapper
 * (coord-extract.ts) that expects a server-side /api/extract endpoint and a
 * shared types module at src/components/render-leaves/types. Neither is
 * present yet. This adapter registers zangalewa in the federation and
 * returns a clear "pending" ActResult until those pieces land.
 *
 * When the server bits arrive, the fetch call below can be uncommented and
 * this becomes a real interceptor.
 * ========================================================================== */

export const zangalewaModule = {
  id: "zangalewa",

  describe() {
    return {
      id: "zangalewa",
      description:
        "Zangalewa: Minimum Sufficient Interceptor. Extracts a coordinate " +
        "from a natural-language utterance. Currently pending server-side wiring.",
      instructions: [
        'dispatch("zangalewa", "your natural-language query")',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const utterance =
      typeof instruction === "string"
        ? instruction
        : typeof instruction === "object" && instruction.utterance
          ? String(instruction.utterance)
          : "";

    // When the server-side extractor is wired up, replace this block with:
    //
    //   const res = await fetch("/api/extract", {
    //     method: "POST",
    //     headers: { "Content-Type": "application/json" },
    //     body: JSON.stringify({ utterance }),
    //   });
    //   if (!res.ok) throw new Error(`extract failed: ${res.status}`);
    //   const coord = await res.json();
    //   return {
    //     ok: true,
    //     output_delta: { kind: "zangalewa_coord", utterance, coord },
    //     residue: 1,
    //     completed: true,
    //   };
    //
    // Until then, we return an honest pending signal so the wiring stays visible.

    return {
      ok: false,
      output_delta: {
        kind: "text",
        lines: [
          `zangalewa: received utterance "${utterance}"`,
          "extractor pending. missing:",
          "  1. /api/extract server route",
          "  2. src/components/render-leaves/types (SCoord, RenderResult)",
          "once those exist, this module extracts a coordinate and returns it.",
        ],
      },
      residue: 0,
      completed: true,
      error: "extractor not yet wired",
    };
  },

  outputCell(_instruction) {
    return { kind: "coord_cell" };
  },
};
