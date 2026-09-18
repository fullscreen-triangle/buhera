// API route: /api/web-search
//
// Internet search, as a thin wrapper around Gemini's Google Search grounding
// tool (chatGeminiGrounded in llm-cascade.js). This exists specifically as
// the complement to /api/spraypaint: spraypaint searches this repo on disk
// and has no network client at all; this route is the only half of "local
// and internet search" that can reach the internet, and it does so through
// an LLM's own browsing tool rather than a dedicated search API, since no
// such API key exists in this deployment.
//
// Contract:
//   POST /api/web-search   body: { query: string }
//   -> { output_delta: { kind: "web_search_result", query, content,
//                         webSearchQueries, sources, supports, elapsed_ms } }
//   -> { ok: false, error: string } on failure (including "no key configured",
//        which is the honest, expected state until GEMINI_API_KEY is fixed)

import { chatGeminiGrounded } from "@/lib/server/llm-cascade";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ ok: false, error: "method not allowed" });
  }

  const { query } = req.body ?? {};
  if (typeof query !== "string" || !query.trim()) {
    return res.status(400).json({ ok: false, error: "query is required" });
  }

  const t0 = Date.now();
  const result = await chatGeminiGrounded({ query });
  const elapsed_ms = Date.now() - t0;

  if (!result.ok) {
    return res.status(502).json({
      ok: false,
      error: result.error || "web search failed",
      provider: result.provider,
      elapsed_ms,
    });
  }

  return res.status(200).json({
    output_delta: {
      kind: "web_search_result",
      query,
      content: result.content,
      webSearchQueries: result.webSearchQueries,
      sources: result.sources,
      supports: result.supports,
      grounded: result.grounded,
      provider: result.provider,
      model: result.model,
      elapsed_ms,
    },
    residue: Array.isArray(result.sources) ? result.sources.length : 0,
  });
}
