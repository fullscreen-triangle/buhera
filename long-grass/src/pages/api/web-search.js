// API route: /api/web-search
//
// Internet search with a search engine (lib/server/web.js: DuckDuckGo by
// default, SearXNG when SEARXNG_URL is set, Brave when BRAVE_SEARCH_KEY is).
// It used to ask Gemini's grounding tool, which made search depend on a
// model key; a search engine needs none and returns sources, not prose.
// The richer route is /api/web, which also reads and keeps pages; this one
// stays for the callers that only search (the spraypaint module's `web`
// kind, the vaHera `web_search` catalyst).
//
// Contract:
//   POST /api/web-search   body: { query: string }
//   -> { output_delta: { kind: "web_search", query, engine, results: [{ title, url, snippet }], elapsed_ms } }
//   -> { ok: false, error: string } on failure

import { allowed } from "@/lib/server/session";
import { search } from "@/lib/server/web";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ ok: false, error: "method not allowed" });
  const who = await allowed(req);
  if (!who.ok) return res.status(who.status).json({ ok: false, error: who.error });
  const query = String(req.body?.query || "").trim();
  if (!query) return res.status(400).json({ ok: false, error: "query is required" });
  const t0 = Date.now();
  try {
    const r = await search(query, { limit: 10 });
    return res.status(200).json({
      output_delta: { kind: "web_search", query, engine: r.engine, results: r.results, elapsed_ms: Date.now() - t0 },
      residue: r.results.length,
    });
  } catch (e) {
    return res.status(502).json({ ok: false, error: e.message || String(e) });
  }
}
