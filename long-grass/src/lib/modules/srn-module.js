/* ============================================================================
 * SRN Module (Sango Rine Shumba)
 *
 * Buhera module that accepts natural-language statements and routes them
 * through the SRN expression language. The user never writes SRN syntax
 * directly — their statement is translated into SRN glyphs server-side via
 * the LLM cascade, compiled, and dispatched to the appropriate SRN node
 * (selected by partition coordinates embedded in the glyph).
 *
 * Output kinds produced:
 *   srn_result   — evaluation result (scalar, vector, or chart data)
 *   srn_peers    — peer list from the forest
 *   srn_probe    — health probe of one node
 *
 * Instruction shapes accepted by execute():
 *   string                  — NL statement → translate → eval
 *   { kind: "eval", glyph } — pre-compiled SRN glyph, skip translation
 *   { kind: "peers" }       — list known peers from the local node
 *   { kind: "probe", node } — probe a specific node URL
 *   { kind: "gossip" }      — trigger gossip round
 * ========================================================================== */

export const srnModule = {
  id: "srn",

  describe() {
    return {
      id: "srn",
      description:
        "SRN (Sango Rine Shumba) expression language: translate natural-language " +
        "statements into SRN glyphs and evaluate them on the nearest forest node.",
      instructions: [
        'dispatch("srn", "show network bandwidth across all nodes")',
        'dispatch("srn", "what is the entropy of the current mesh?")',
        'dispatch("srn", { kind: "peers" })',
        'dispatch("srn", { kind: "probe", node: "http://100.77.3.78:7700" })',
        'dispatch("srn", { kind: "eval", glyph: "◈(n=2,l=1,m=0,s=-1)" })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    try {
      const body = _buildBody(instruction);
      const res = await fetch("/api/srn", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => res.statusText);
        return _err(`HTTP ${res.status}: ${text}`);
      }

      const data = await res.json();
      if (data.error) return _err(data.error);

      return {
        ok: true,
        output_delta: data.output_delta,
        residue: data.residue ?? 1,
        completed: true,
      };
    } catch (err) {
      return _err(err.message || String(err));
    }
  },

  outputCell(instruction) {
    const kind = typeof instruction === "object" ? instruction?.kind : "eval";
    return { kind: `srn_${kind || "result"}` };
  },
};

// ---------------------------------------------------------------------------

function _buildBody(instruction) {
  if (typeof instruction === "string") {
    return { kind: "nl", text: instruction };
  }
  if (instruction && typeof instruction === "object") {
    return instruction;
  }
  return { kind: "nl", text: String(instruction) };
}

function _err(message) {
  return {
    ok: false,
    output_delta: { kind: "srn_error", message },
    residue: 0,
    completed: true,
    error: message,
  };
}
