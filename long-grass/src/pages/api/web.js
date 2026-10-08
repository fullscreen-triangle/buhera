// API route for reading the web: search, read a page or a site, the library.
//
//   POST /api/web { action: "search", query }          → a search engine's results
//   POST /api/web { action: "read", url }              → the page, kept in the library
//   POST /api/web { action: "site", url, limit? }      → the page and the pages under its path (≤ 80)
//   POST /api/web { action: "library" }                → every page read
//   POST /api/web { action: "page", url }              → one kept page's Markdown
//   POST /api/web { action: "ask", query, dry_run? }   → spraypaint over what has been read, with its verdict
//
// From this machine, always; from elsewhere, for a signed-in member only
// (lib/server/session.js), and never a private address (lib/server/web.js).

import fs from "fs";
import path from "path";
import { allowed } from "@/lib/server/session";
import { crawl, libraryDir, libraryPages, readKept, readUrl, reindexLibrary as reindex, search } from "@/lib/server/web";
import { findBinary, run } from "@/lib/server/spawn";
import { askArgs, parseJsonLoose } from "@/lib/server/spraypaint";

const strip = ({ outline, ...e }) => ({ ...e, headings: outline?.length || 0 });

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ ok: false, error: "method not allowed" });
  const who = await allowed(req);
  if (!who.ok) return res.status(who.status).json({ ok: false, error: who.error });

  const { action, query, url, limit, dry_run, budget } = req.body ?? {};
  const dir = libraryDir();
  const fail = (status, error) => res.status(status).json({ ok: false, error });
  const validUrl = (u) => { try { return /^https?:$/.test(new URL(u).protocol); } catch { return false; } };

  try {
    switch (action) {
      case "search": {
        if (!String(query || "").trim()) return fail(400, "what should be searched for?");
        const r = await search(query, { limit: 10 });
        return res.status(200).json({ ok: true, query, ...r });
      }
      case "read": {
        if (!validUrl(url)) return fail(400, "give the full address, starting with https://");
        const { entry } = await readUrl(url, { local: who.local, dir });
        const index = await reindex(dir);
        return res.status(200).json({ ok: true, entry, index });
      }
      case "site": {
        if (!validUrl(url)) return fail(400, "give the full address, starting with https://");
        const n = Math.min(80, Math.max(1, Number(limit) || 30));
        const r = await crawl(url, { limit: n, local: who.local, dir });
        const index = await reindex(dir);
        return res.status(200).json({ ok: true, start: url, limit: n, pages: r.pages.map(strip), skipped: r.skipped, errors: r.errors, index });
      }
      case "library":
        return res.status(200).json({ ok: true, dir, pages: libraryPages(dir).map(strip) });
      case "page": {
        const kept = readKept(url, dir);
        if (!kept) return fail(404, `${url} has not been read on this server yet`);
        const { raw, ...rest } = kept;
        return res.status(200).json({ ok: true, ...rest });
      }
      case "ask": {
        if (!String(query || "").trim()) return fail(400, "query is required");
        if (!fs.existsSync(path.join(dir, ".spraypaint", "index.json"))) return fail(409, "nothing has been read yet");
        const bin = findBinary("spraypaint", "SPRAYPAINT_CLI");
        if (!bin) return fail(503, "spraypaint is not installed");
        const r = await run(bin, askArgs({ query, root: dir, budget, dryRun: !!dry_run }), { timeoutMs: 300_000 });
        if (r.code !== 0) return fail(502, `spraypaint exited with ${r.code}: ${r.stderr.trim().slice(0, 200)}`);
        const parsed = parseJsonLoose(r.stdout);
        if (!parsed) return fail(502, "could not read spraypaint's output");
        // A passage's path leads back to the page it came from.
        const byFile = new Map(libraryPages(dir).map((p) => [p.file, p]));
        for (const x of parsed.results || []) {
          const p = byFile.get(x.path);
          if (p) { x.source_url = p.url; x.source_title = p.title; }
        }
        return res.status(200).json({ ok: true, output_delta: { kind: "spraypaint_result", query, corpus: "library", elapsed_ms: r.elapsed_ms, ...parsed } });
      }
      default:
        return fail(400, `unknown action "${action}"`);
    }
  } catch (e) {
    return fail(502, e.message || String(e));
  }
}
