/* ============================================================================
 * Purpose CLI Module
 *
 * Wraps the existing /api/purpose route (pages/api/purpose.js), which spawns
 * the Rust `purpose` CLI (semantics/purpose/mechanistic-synthesis) — symbol
 * definition lookup and codebase-goal navigation. Not an LLM, and distinct
 * from the "purpose" module (purpose-module.js), which is the federated
 * LLM-synthesis cascade over /api/purpose-federation.
 *
 * Instructions:
 *   • { kind: "query", utterance, mode? }   — "purpose ask <utterance>"
 *       mode: "value" (default, parsed JSON value) or "fragment" (compiled
 *       vaHera fragment — see pages/api/purpose.js and integration.md §5/§9.4)
 *   • a plain string                        — shorthand for { kind: "query", utterance: <string> }
 * ========================================================================== */

async function callPurpose(body) {
  let res;
  try {
    res = await fetch("/api/purpose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
  let json = null;
  try { json = await res.json(); } catch { /* not JSON */ }
  if (!res.ok || !json?.ok) {
    return { ok: false, error: json?.error || `HTTP ${res.status}`, stderr: json?.stderr };
  }
  return json;
}

export const purposeCliModule = {
  id: "purpose-cli",

  describe() {
    return {
      id: "purpose-cli",
      description:
        "purpose-cli — symbol definitions and codebase navigation via the Rust `purpose` CLI, " +
        "spawned server-side. Not an LLM; a local index lookup.",
      instructions: [
        'dispatch("purpose-cli", "Resolver")',
        'dispatch("purpose-cli", { kind: "query", utterance: "build_ontology" })',
        'dispatch("purpose-cli", { kind: "query", utterance: "main", mode: "fragment" })',
      ],
    };
  },

  async execute(instruction) {
    const inst = typeof instruction === "string" ? { kind: "query", utterance: instruction } : instruction || {};
    const kind = inst.kind || "query";

    if (kind !== "query") {
      return {
        ok: false,
        output_delta: { kind: "text", lines: [`purpose-cli: unknown kind "${kind}"`] },
        residue: 0,
        completed: true,
        error: `unknown kind "${kind}"`,
      };
    }

    const utterance = String(inst.utterance || "").trim();
    if (!utterance) {
      return {
        ok: false,
        output_delta: { kind: "text", lines: ["purpose-cli: utterance is required"] },
        residue: 0,
        completed: true,
        error: "no-utterance",
      };
    }
    const mode = inst.mode === "fragment" ? "fragment" : "value";

    const res = await callPurpose({ utterance, mode });
    if (!res.ok) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`purpose-cli: ${res.error}`, ...(res.stderr ? [res.stderr.trim()] : [])],
        },
        residue: 0,
        completed: true,
        error: res.error,
      };
    }

    const payload = mode === "fragment" ? res.fragment : res.value;
    return {
      ok: true,
      output_delta: {
        kind: "purpose_cli_result",
        utterance,
        mode,
        payload,
        elapsed_ms: res.elapsed_ms,
        lines: [`purpose-cli (${mode}, ${res.elapsed_ms}ms):`, JSON.stringify(payload, null, 2)],
      },
      residue: 1,
      completed: true,
    };
  },

  outputCell() {
    return { kind: "purpose_cli_cell" };
  },
};
