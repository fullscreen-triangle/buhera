/* ============================================================================
 * web — search the web with a search engine, read pages, keep what you read.
 * Server side: pages/api/web.js and lib/server/web.js.
 *
 * Instruction shapes:
 *   "https://…"                                → read that page
 *   "<words>"                                  → search
 *   { kind: "search", query }                  → a search engine's results
 *   { kind: "read", url }                      → the page, as a frame you can read and note
 *   { kind: "site", url, limit? }              → the page and the pages under its path
 *   { kind: "library" }                        → every page read
 *   { kind: "ask", query, dry_run? }           → search what you have read, with a verdict
 * ========================================================================== */

import { postJSON } from "@/lib/auth/headers";

const done = (output_delta, ok = true) => ({ ok, output_delta, residue: ok ? 1 : 0, completed: true });
const failed = (what, error) => done({ kind: "text", lines: [`web ${what}: ${error}`] }, false);
const isUrl = (s) => /^https?:\/\/\S+$/i.test(String(s).trim());

export const webModule = {
  id: "web",

  describe() {
    return {
      id: "web",
      description:
        "The web, read: search with a search engine, read a page or a whole documentation site, and keep what you read " +
        "in your library — so `find` can answer from it, with a verdict, and its pages can be noted, diagrammed and compared.",
      instructions: [
        "web DCAT-AP-PLUS LinkML",
        "read https://semiceu.github.io/DCAT-AP/releases/3.0.1/",
        "read site https://nfdi-de.github.io/dcat-ap-plus/latest/",
        "library",
      ],
    };
  },

  async execute(instruction) {
    let inst = instruction;
    if (typeof inst === "string") {
      const s = inst.trim();
      inst = s === "library" || s === "show" ? { kind: "library" } : isUrl(s) ? { kind: "read", url: s } : { kind: "search", query: s };
    }
    inst = inst || {};
    switch (inst.kind || "library") {
      case "search": {
        const r = await postJSON("/api/web", { action: "search", query: inst.query });
        return r.ok ? done({ kind: "web_search", query: inst.query, engine: r.engine, results: r.results }) : failed("search", r.error);
      }
      case "read": {
        const r = await postJSON("/api/web", { action: "read", url: inst.url });
        if (!r.ok) return failed("read", r.error);
        const { url, title, read, words, outline } = r.entry;
        return done({ kind: "web_page", url, title, read, words, outline, indexed: !!r.index?.ok });
      }
      case "site": {
        const r = await postJSON("/api/web", { action: "site", url: inst.url, limit: inst.limit });
        return r.ok ? done({ kind: "web_site", start: r.start, limit: r.limit, pages: r.pages, skipped: r.skipped, errors: r.errors, indexed: !!r.index?.ok }) : failed("site", r.error);
      }
      case "library": {
        const r = await postJSON("/api/web", { action: "library" });
        return r.ok ? done({ kind: "web_library", pages: r.pages }) : failed("library", r.error);
      }
      case "page": {
        const r = await postJSON("/api/web", { action: "page", url: inst.url });
        return r.ok ? done({ kind: "web_page", url: r.url, title: r.title, read: r.read, words: r.words, outline: r.outline }) : failed("page", r.error);
      }
      case "ask": {
        const r = await postJSON("/api/web", { action: "ask", query: inst.query, dry_run: inst.dry_run !== false, budget: inst.budget });
        return r.ok ? done(r.output_delta) : failed("ask", r.error);
      }
      default:
        return failed("", `unknown kind "${inst.kind}"`);
    }
  },

  outputCell() {
    return { kind: "web_cell" };
  },
};
