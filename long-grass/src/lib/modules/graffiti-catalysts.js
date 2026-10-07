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
import { postJSON } from "@/lib/auth/headers";

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
 *   claim = the top passage's evidence lines, cited path:start-end
 *   power = from spraypaint's coverage verdict, never from its score (a BM25
 *           score means something only within one query): covered 0.8,
 *           partial 0.4–0.7 by the share of the query's weight found,
 *           declined 0 — the corpus does not hold what the query names.
 *           An older build with no verdict gets 0.3: a possible look-alike.
 *   All capped at `power`.
 */
function verdictPower(coverage) {
  const v = coverage?.verdict;
  if (v === "covered") return 0.8;
  if (v === "partial") return 0.4 + 0.3 * (coverage.weight_share ?? 0.5);
  if (v === "declined") return 0;
  return 0.3;
}

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
        const { results = [], coverage } = body.output_delta;
        if (coverage?.verdict === "declined") {
          return { claim: `${name}:declined:${coverage.reason}`, power: 0 };
        }
        if (results.length === 0) {
          return { claim: `${name}:no-hits:${query}`, power: 0 };
        }
        // Results come grouped by scene; the best passage is the top score.
        const top = results.reduce((a, b) => (b.score > a.score ? b : a));
        const from = top.evidence_start_line ?? top.start_line;
        const to = top.evidence_end_line ?? top.end_line;
        const claim = `${top.path}:${from}-${to}: ${top.snippet}`;
        return { claim, power: Math.min(verdictPower(coverage), power) };
      } catch {
        return { claim: `${name}:error:${query}`, power: 0 };
      }
    },
  };
}

/**
 * web_search — internet search with a search engine, through /api/web-search.
 * Namespace "remote" since it's the one catalyst here that actually leaves
 * the machine.
 *
 * Instruction args:
 *   - "query" (string): the search text; defaults to currentClaim
 *
 * Result:
 *   claim = the top result: its title, address and the engine's snippet
 *   power = 0.4 when there are results — a snippet is a pointer to a source,
 *           not a reading of it — and 0 when there are none or search fails
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
        const body = await postJSON("/api/web-search", { query });
        if (!body.ok || !body?.output_delta) {
          return { claim: `${name}:unreachable:${query}`, power: 0 };
        }
        const top = body.output_delta.results?.[0];
        if (!top) return { claim: `${name}:no-results:${query}`, power: 0 };
        return { claim: `${top.title} — ${top.url}: ${top.snippet}`, power: Math.min(0.4, power) };
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
