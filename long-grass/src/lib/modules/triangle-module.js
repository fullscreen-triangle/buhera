/* ============================================================================
 * Triangle Module
 *
 * Wraps /api/triangle, which proxies to the four-corners backend
 * (four-sided-triangle) using a single server-held service credential —
 * separate from the 5 buhera-gateway profiles, which don't need their own
 * four-corners accounts.
 *
 * Instructions:
 *   • "sources" | { kind: "sources" }                          — list sources
 *   • { kind: "add_source", sourceKind, config }                — add a source
 *   • { kind: "remove_source", id }                             — remove a source
 * ========================================================================== */

async function callTriangle(body) {
  let res;
  try {
    res = await fetch("/api/triangle", {
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
    return { ok: false, error: json?.error || `HTTP ${res.status}` };
  }
  return json;
}

export const triangleModule = {
  id: "triangle",

  describe() {
    return {
      id: "triangle",
      description:
        "triangle — four-sided-triangle knowledge sources, proxied server-side through a " +
        "shared four-corners service account.",
      instructions: [
        'dispatch("triangle", "sources")',
        'dispatch("triangle", { kind: "add_source", sourceKind: "url", config: { url: "..." } })',
        'dispatch("triangle", { kind: "remove_source", id: "..." })',
      ],
    };
  },

  async execute(instruction) {
    const inst = typeof instruction === "string" ? { kind: instruction } : instruction || {};
    const kind = inst.kind || "sources";

    if (kind === "sources") {
      const res = await callTriangle({ kind: "sources" });
      if (!res.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`triangle: ${res.error}`] },
          residue: 0,
          completed: true,
          error: res.error,
        };
      }
      const items = res.sources || [];
      const lines = items.length
        ? items.map((s) => `  ${s.id} — ${s.kind}`)
        : ["  (no sources configured)"];
      return {
        ok: true,
        output_delta: { kind: "triangle_sources", entries: items, count: items.length, lines: ["triangle sources:", ...lines] },
        residue: items.length,
        completed: true,
      };
    }

    if (kind === "add_source") {
      const sourceKind = String(inst.sourceKind || "").trim();
      const config = inst.config;
      if (!sourceKind || !config) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: ["triangle add_source: sourceKind and config are required"] },
          residue: 0,
          completed: true,
          error: "missing-fields",
        };
      }
      const res = await callTriangle({ kind: "add_source", sourceKind, config });
      if (!res.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`triangle add_source: ${res.error}`] },
          residue: 0,
          completed: true,
          error: res.error,
        };
      }
      return {
        ok: true,
        output_delta: { kind: "text", lines: [`triangle: added source ${res.source?.id ?? ""}`] },
        residue: 1,
        completed: true,
      };
    }

    if (kind === "remove_source") {
      const id = String(inst.id || "").trim();
      if (!id) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: ["triangle remove_source: id is required"] },
          residue: 0,
          completed: true,
          error: "no-id",
        };
      }
      const res = await callTriangle({ kind: "remove_source", id });
      if (!res.ok) {
        return {
          ok: false,
          output_delta: { kind: "text", lines: [`triangle remove_source: ${res.error}`] },
          residue: 0,
          completed: true,
          error: res.error,
        };
      }
      return {
        ok: true,
        output_delta: { kind: "text", lines: [`triangle: removed source ${id}`] },
        residue: 0,
        completed: true,
      };
    }

    return {
      ok: false,
      output_delta: { kind: "text", lines: [`triangle: unknown kind "${kind}"`] },
      residue: 0,
      completed: true,
      error: `unknown kind "${kind}"`,
    };
  },

  outputCell() {
    return { kind: "triangle_cell" };
  },
};
