/* ============================================================================
 * Spraypaint Module Adapter
 *
 * Browser-side client for the spraypaint search backend: /api/spraypaint
 * (local, full-text passage retrieval over this repo) and /api/web-search
 * (internet, via an LLM's own browsing tool). Module code runs in the
 * browser, so both are called over fetch — the actual CLI spawn and the
 * actual Gemini call happen server-side in those two routes.
 *
 * Instruction shapes:
 *   { kind: "ask", query, root?, budget?, scenes? }  → local search
 *   { kind: "index", root? }                          → (re)build the local index
 *   { kind: "web", query }                             → internet search
 *   { kind: "both", query, root?, budget?, scenes? }   → both, in parallel
 *   a plain string                                     → sugar for { kind: "ask", query: <string> }
 * ========================================================================== */

async function postJSON(path, body) {
  let res;
  try {
    res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
  const json = await res.json().catch(() => null);
  if (!res.ok || !json) {
    return { ok: false, error: json?.error || `HTTP ${res.status}`, ...json };
  }
  return { ok: true, ...json };
}

async function runAsk(inst) {
  const query = String(inst.query || "");
  if (!query.trim()) {
    return {
      ok: false,
      output_delta: { kind: "text", lines: ["spraypaint ask: query is required"] },
      residue: 0,
      completed: true,
      error: "no-query",
    };
  }
  const res = await postJSON("/api/spraypaint", {
    action: "ask",
    query,
    root: inst.root,
    budget: inst.budget,
    scenes: inst.scenes,
  });
  if (!res.ok) {
    return {
      ok: false,
      output_delta: { kind: "text", lines: [`spraypaint ask: ${res.error}`] },
      residue: 0,
      completed: true,
      error: res.error,
    };
  }
  return {
    ok: true,
    output_delta: res.output_delta,
    residue: res.residue ?? 0,
    completed: true,
  };
}

async function runIndex(inst) {
  const res = await postJSON("/api/spraypaint", { action: "index", root: inst.root });
  if (!res.ok) {
    return {
      ok: false,
      output_delta: { kind: "text", lines: [`spraypaint index: ${res.error}`] },
      residue: 0,
      completed: true,
      error: res.error,
    };
  }
  return {
    ok: true,
    output_delta: res.output_delta,
    residue: res.residue ?? 0,
    completed: true,
  };
}

async function runWeb(inst) {
  const query = String(inst.query || "");
  if (!query.trim()) {
    return {
      ok: false,
      output_delta: { kind: "text", lines: ["spraypaint web: query is required"] },
      residue: 0,
      completed: true,
      error: "no-query",
    };
  }
  const res = await postJSON("/api/web-search", { query });
  if (!res.ok) {
    return {
      ok: false,
      output_delta: { kind: "text", lines: [`spraypaint web: ${res.error}`] },
      residue: 0,
      completed: true,
      error: res.error,
    };
  }
  return {
    ok: true,
    output_delta: res.output_delta,
    residue: res.residue ?? 0,
    completed: true,
  };
}

async function runBoth(inst) {
  const [local, web] = await Promise.all([runAsk(inst), runWeb(inst)]);
  return {
    ok: local.ok || web.ok,
    output_delta: {
      kind: "search_combined",
      query: inst.query,
      local: local.output_delta,
      local_ok: local.ok,
      web: web.output_delta,
      web_ok: web.ok,
    },
    residue: (local.residue || 0) + (web.residue || 0),
    completed: true,
  };
}

export const spraypaintModule = {
  id: "spraypaint",

  describe() {
    return {
      id: "spraypaint",
      description:
        "Search: spraypaint (local full-text passage retrieval over this repo, " +
        "BM25 within scenes + water-filling across them) and internet search " +
        "(an LLM's own browsing tool). No shared state with vaHera's own " +
        "memory — this searches files on disk / the web, not what you've " +
        "`memory store`d.",
      instructions: [
        'dispatch("spraypaint", { kind: "ask", query: "admissibility floor" })',
        'dispatch("spraypaint", { kind: "ask", query: "...", scenes: ["long-grass"], budget: 5 })',
        'dispatch("spraypaint", { kind: "index" })',
        'dispatch("spraypaint", { kind: "web", query: "..." })',
        'dispatch("spraypaint", { kind: "both", query: "..." })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const inst = typeof instruction === "string" ? { kind: "ask", query: instruction } : instruction || {};
    const kind = inst.kind || "ask";

    if (kind === "ask") return runAsk(inst);
    if (kind === "index") return runIndex(inst);
    if (kind === "web") return runWeb(inst);
    if (kind === "both") return runBoth(inst);

    return {
      ok: false,
      output_delta: { kind: "text", lines: [`spraypaint: unknown kind "${kind}"`] },
      residue: 0,
      completed: true,
      error: `unknown kind "${kind}"`,
    };
  },

  outputCell() {
    return { kind: "spraypaint_cell" };
  },
};
