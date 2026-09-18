/* ============================================================================
 * Real catalysts for the graffiti module.
 *
 * Two catalysts land in v1:
 *
 *   • kernel_search — local (namespace: "local")
 *       Backed by vahera's Kernel + substrate embedding. Given a query
 *       string, embeds it into an S-coord and returns the nearest stored
 *       object's payload as a claim. Zero external dependencies.
 *
 *   • hf_inference — remote (namespace: "inference")
 *       Goes through /api/hf-inference. The route hits HuggingFace with the
 *       claim as input and returns a refined claim. Requires
 *       HUGGINGFACE_API_KEY in .env.local; without it, the catalyst returns
 *       the input unchanged with power 0 so scripts still run.
 *
 * Real Buhera integrations can register additional catalysts implementing
 * the same CatalystDefinition contract — the graffiti calculus is namespace-
 * neutral by theorem, so registering a new one is a one-liner.
 * ========================================================================== */

import { embedText, sDistance } from "@/lib/substrate";

/**
 * kernel_search — reads from a Kernel instance passed in at build time.
 * Instruction args:
 *   - "query" (string): the search text; defaults to currentClaim
 *
 * Result:
 *   claim = top-hit payload (or the string form if payload is missing)
 *   power = 0.5 + 0.4 * (1 - normalisedDistance)   in [0.5, 0.9]
 */
export function createKernelSearchCatalyst(name, kernel, power = 0.7) {
  return {
    name,
    namespace: "local",
    provider: async (ctx) => {
      const query = String(ctx.args?.query ?? ctx.currentClaim ?? "");
      if (!query.trim() || !kernel || kernel.store.size === 0) {
        return {
          claim: `${name}:no-corpus:${query}`,
          power: 0,
        };
      }
      const targetCoord = embedText(query);
      const hits = kernel.proximity(targetCoord, 1);
      if (hits.length === 0) {
        return { claim: `${name}:miss:${query}`, power: 0 };
      }
      const [topObj, dist] = hits[0];
      // Distance in [0, some_max]; convert to a power in [0.5, 0.9].
      // sDistance is Fisher-metric on [0,1]^3, capped around ~1.7 in practice.
      const normalisedCloseness = Math.max(0, 1 - dist / 1.5);
      const p = Math.min(0.9, 0.5 + 0.4 * normalisedCloseness);
      const claim =
        typeof topObj.payload === "string"
          ? topObj.payload
          : topObj.payload
            ? JSON.stringify(topObj.payload)
            : `${name}:hit:${topObj.address}`;
      return { claim, power: Math.min(p, power) };
    },
  };
}

/**
 * spraypaint_local — full-text passage retrieval over this repo, via
 * /api/spraypaint. Unlike kernel_search, this has no idea what you've
 * `memory store`d — it searches files on disk, ranked by BM25 within
 * "scenes" (top-level directories) and allocated by a water-filling
 * budget across them. Zero shared state with vaHera's kernel.
 *
 * Instruction args:
 *   - "query" (string): the search text; defaults to currentClaim
 *
 * Result:
 *   claim = the top-scoring passage's snippet, or a no-hits marker
 *   power = 0.5 + 0.4 * normalised top score, capped at `power`
 */
export function createSpraypaintCatalyst(name, power = 0.7) {
  return {
    name,
    namespace: "local",
    provider: async (ctx) => {
      const query = String(ctx.args?.query ?? ctx.currentClaim ?? "");
      if (!query.trim()) {
        return { claim: `${name}:no-query`, power: 0 };
      }
      try {
        const res = await fetch("/api/spraypaint", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: "ask", query, budget: 3 }),
        });
        const body = await res.json().catch(() => null);
        if (!res.ok || !body?.output_delta) {
          return { claim: `${name}:unreachable:${query}`, power: 0 };
        }
        const results = body.output_delta.results || [];
        if (results.length === 0) {
          return { claim: `${name}:no-hits:${query}`, power: 0 };
        }
        const top = results[0];
        // score is unbounded BM25; 20 is a generous ceiling seen in practice
        // for this repo's index, used only to keep power in [0, power].
        const normalised = Math.max(0, Math.min(1, top.score / 20));
        const p = Math.min(0.9, 0.5 + 0.4 * normalised);
        const claim = `${top.path}:${top.start_line}-${top.end_line}: ${top.snippet}`;
        return { claim, power: Math.min(p, power) };
      } catch {
        return { claim: `${name}:error:${query}`, power: 0 };
      }
    },
  };
}

/**
 * web_search — internet search via an LLM's own browsing tool, through
 * /api/web-search. Namespace "remote" since it's the one catalyst here that
 * actually leaves the machine.
 *
 * Instruction args:
 *   - "query" (string): the search text; defaults to currentClaim
 *
 * Result:
 *   claim = the grounded answer text
 *   power = 0.6 on success, 0 if unconfigured/unreachable (e.g. no
 *           GEMINI_API_KEY, or the key is invalid)
 */
export function createWebSearchCatalyst(name, power = 0.6) {
  return {
    name,
    namespace: "remote",
    provider: async (ctx) => {
      const query = String(ctx.args?.query ?? ctx.currentClaim ?? "");
      if (!query.trim()) {
        return { claim: `${name}:no-query`, power: 0 };
      }
      try {
        const res = await fetch("/api/web-search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });
        const body = await res.json().catch(() => null);
        if (!res.ok || !body?.output_delta) {
          return { claim: `${name}:unreachable:${query}`, power: 0 };
        }
        return { claim: body.output_delta.content, power };
      } catch {
        return { claim: `${name}:error:${query}`, power: 0 };
      }
    },
  };
}

/**
 * hf_inference — refines a claim by asking a small HF chat model to
 * paraphrase or clarify it. Goes through /api/hf-inference.
 *
 * Instruction args:
 *   - "prompt" (string, optional): overrides the default instruction
 *
 * Result:
 *   claim = the model's refined text
 *   power = 0.6 on success, 0 if the route is unreachable / unconfigured
 */
export function createHfInferenceCatalyst(name, power = 0.6) {
  return {
    name,
    namespace: "inference",
    provider: async (ctx) => {
      const input = String(ctx.currentClaim ?? "");
      const promptOverride =
        typeof ctx.args?.prompt === "string" ? ctx.args.prompt : null;
      if (!input.trim()) {
        return { claim: input, power: 0 };
      }
      try {
        const res = await fetch("/api/hf-inference", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ claim: input, prompt: promptOverride }),
        });
        const body = await res.json().catch(() => null);
        if (!res.ok || !body || !body.ok) {
          return { claim: input, power: 0 };
        }
        return { claim: String(body.refined || input), power };
      } catch {
        return { claim: input, power: 0 };
      }
    },
  };
}
