/* ============================================================================
 * The web, read: search with a search engine, fetch a page, keep it.
 *
 *   search(query)     a search engine's results: SearXNG when SEARXNG_URL is
 *                     set, Brave when BRAVE_SEARCH_KEY is, else DuckDuckGo's
 *                     HTML endpoint (no key needed)
 *   readUrl(url)      fetch one page and turn it into Markdown with its
 *                     outline (headings and their anchors) and links
 *   crawl(url)        a documentation site: the page, then the pages under
 *                     the same path it links to, up to a limit
 *   the library       every page read is kept as Markdown, LIBRARY_DIR or
 *                     ~/.buhera/library/<host>/<path>.md, with its raw source
 *                     beside it in .raw/ (for extracting tables and schemas),
 *                     and indexed by spraypaint so `find` can answer from
 *                     what you have read, with a coverage verdict
 *
 * Fetching guards against being turned on the server's own network: only
 * http(s), and — for a request from anywhere but this machine — no address
 * that resolves to a loopback, private or link-local range, checked again on
 * every redirect. Pages are capped at 6 MB.
 * ========================================================================== */

import fs from "fs";
import os from "os";
import path from "path";
import dns from "dns";
import net from "net";
import * as cheerio from "cheerio";
import TurndownService from "turndown";
import { gfm } from "@joplin/turndown-plugin-gfm";
import { findBinary, run } from "@/lib/server/spawn";
import { parseJsonLoose } from "@/lib/server/spraypaint";

const UA = "Mozilla/5.0 (compatible; BuheraReader/0.1; +https://github.com/fullscreen-triangle/buhera)";
const MAX_BYTES = 6 * 1024 * 1024;

export function libraryDir(env = process.env) {
  const d = env.LIBRARY_DIR || "~/.buhera/library";
  return d.startsWith("~") ? path.join(os.homedir(), d.slice(1)) : d;
}

// ── the address guard ────────────────────────────────────────────────────

export function isPrivateAddress(ip) {
  if (net.isIPv4(ip)) {
    const [a, b] = ip.split(".").map(Number);
    return a === 10 || a === 127 || a === 0 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) ||
      (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127) || a >= 224;
  }
  const v = ip.toLowerCase();
  if (v.startsWith("::ffff:")) return isPrivateAddress(v.slice(7));
  return v === "::1" || v === "::" || v.startsWith("fc") || v.startsWith("fd") || v.startsWith("fe80");
}

async function guard(u, local) {
  if (!/^https?:$/.test(u.protocol)) throw new Error(`only http and https pages can be read, not ${u.protocol}`);
  if (local) return;
  const host = u.hostname.replace(/^\[|\]$/g, "");
  const addrs = net.isIP(host) ? [{ address: host }] : await dns.promises.lookup(host, { all: true });
  if (addrs.some((a) => isPrivateAddress(a.address))) throw new Error(`${u.hostname} is a private address; this server reads public pages only`);
}

/** Fetch with the guard re-applied on every redirect. → { url, status, contentType, body } */
export async function fetchGuarded(url, { local = false, maxBytes = MAX_BYTES, timeoutMs = 25_000 } = {}) {
  let u = new URL(url);
  for (let hop = 0; hop < 6; hop++) {
    await guard(u, local);
    const res = await fetch(u, { redirect: "manual", headers: { "User-Agent": UA, Accept: "text/html,application/xhtml+xml,text/plain,application/yaml,*/*;q=0.5" }, signal: AbortSignal.timeout(timeoutMs) });
    if (res.status >= 300 && res.status < 400 && res.headers.get("location")) {
      u = new URL(res.headers.get("location"), u);
      continue;
    }
    const len = Number(res.headers.get("content-length") || 0);
    if (len > maxBytes) throw new Error(`the page is ${Math.round(len / 1048576)} MB; the limit is ${Math.round(maxBytes / 1048576)} MB`);
    const buf = Buffer.from(await res.arrayBuffer());
    if (buf.length > maxBytes) throw new Error(`the page is larger than ${Math.round(maxBytes / 1048576)} MB`);
    return { url: u.toString(), status: res.status, contentType: res.headers.get("content-type") || "", body: buf.toString("utf8") };
  }
  throw new Error("too many redirects");
}

// ── HTML → Markdown ──────────────────────────────────────────────────────

const DROP = [
  "script", "style", "noscript", "template", "iframe", "form", "button", "nav", "header", "footer", "aside",
  "[role=navigation]", "[role=banner]", "[role=contentinfo]", "#toc", ".toc", "#respec-ui", ".md-sidebar",
  ".md-header", ".md-footer", ".md-source", ".headerlink", ".skip-link", "[aria-hidden=true]",
];
const MAIN = ["main", "article", "[role=main]", ".md-content", "#content", "body"];

function turndown(base) {
  const td = new TurndownService({ headingStyle: "atx", codeBlockStyle: "fenced", bulletListMarker: "-", emDelimiter: "*" });
  td.use(gfm);
  td.addRule("absoluteLinks", {
    filter: (n) => n.nodeName === "A" && n.getAttribute("href"),
    replacement: (content, n) => {
      const text = content.replace(/\s+/g, " ").trim();
      if (!text) return "";
      try {
        const href = new URL(n.getAttribute("href"), base).toString();
        return `[${text}](${href})`;
      } catch {
        return text;
      }
    },
  });
  td.addRule("images", {
    filter: "img",
    replacement: (_c, n) => {
      const src = n.getAttribute("src");
      if (!src) return "";
      try { return `[image: ${n.getAttribute("alt") || "figure"}](${new URL(src, base)})`; } catch { return ""; }
    },
  });
  return td;
}

const slug = (s) => String(s).toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-+|-+$/g, "");

/**
 * One HTML page as Markdown. Headings keep an anchor (their id, their
 * section's id, or a slug), so a note can cite `url#anchor`.
 * → { title, markdown, outline: [{ level, text, anchor }], links: [absolute urls] }
 */
export function htmlToPage(html, url) {
  const $ = cheerio.load(html);
  const title = ($("title").first().text() || $("h1").first().text() || url).replace(/\s+/g, " ").trim();
  $("*").contents().filter((_, n) => n.type === "comment").remove();
  const links = new Set();
  $("a[href]").each((_, a) => {
    try {
      const u = new URL($(a).attr("href"), url);
      u.hash = "";
      links.add(u.toString());
    } catch { /* not a URL */ }
  });
  DROP.forEach((sel) => $(sel).remove());
  const main = MAIN.map((s) => $(s).first()).find((m) => m.length && m.text().trim().length > 200) || $("body");

  const outline = [];
  const used = new Set();
  main.find("h1, h2, h3, h4").each((_, h) => {
    const el = $(h);
    const text = el.text().replace(/\s+/g, " ").replace(/[¶§#]\s*$/, "").trim();
    if (!text) return;
    // The heading's own id, else its section's (if no earlier heading took
    // it), else a slug: the anchor a citation `url#anchor` can actually reach.
    const sectionId = el.closest("section[id]").attr("id");
    let anchor = el.attr("id") || (sectionId && !used.has(sectionId) ? sectionId : slug(text));
    for (let i = 2; used.has(anchor); i++) anchor = `${anchor.replace(/-\d+$/, "")}-${i}`;
    used.add(anchor);
    outline.push({ level: Number(h.tagName[1]), text, anchor });
    el.text(`${text} {#${anchor}}`);
  });

  let markdown = turndown(url).turndown(main.html() || "");
  markdown = markdown.replace(/\n{3,}/g, "\n\n").trim();
  return { title, markdown, outline, links: [...links] };
}

/** A non-HTML page (YAML, Markdown, plain text) as Markdown. */
export function textToPage(text, url, contentType) {
  const name = decodeURIComponent(new URL(url).pathname.split("/").pop() || url);
  const isMd = /markdown/.test(contentType) || /\.md$/i.test(name);
  const lang = /\.ya?ml$/i.test(name) || /yaml/.test(contentType) ? "yaml" : /json/.test(contentType) ? "json" : "";
  const markdown = isMd ? text : `\`\`\`${lang}\n${text.replace(/```/g, "ˋˋˋ")}\n\`\`\``;
  const outline = isMd ? [...text.matchAll(/^(#{1,4})\s+(.+)$/gm)].map((m) => ({ level: m[1].length, text: m[2].trim(), anchor: slug(m[2]) })) : [];
  return { title: name, markdown, outline, links: [] };
}

// ── the library ──────────────────────────────────────────────────────────

/** Where a URL's page is kept, relative to the library. */
export function libraryFile(url) {
  const u = new URL(url);
  let p = u.pathname.replace(/\/+$/, "/");
  if (p.endsWith("/")) p += "index";
  p = p.replace(/\.(html?|php|aspx?)$/i, "");
  const parts = [u.hostname, ...p.split("/").filter(Boolean)].map((s) => decodeURIComponent(s).replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 80));
  return `${parts.join("/")}.md`;
}

function catalogue(dir) {
  try { return JSON.parse(fs.readFileSync(path.join(dir, "pages.json"), "utf8")); } catch { return {}; }
}

export function libraryPages(dir = libraryDir()) {
  return Object.values(catalogue(dir)).sort((a, b) => String(b.read).localeCompare(String(a.read)));
}

export function libraryEntry(url, dir = libraryDir()) {
  const c = catalogue(dir);
  return c[url] || c[url.replace(/\/$/, "")] || c[`${url}/`] || null;
}

export function readKept(url, dir = libraryDir()) {
  const e = libraryEntry(url, dir);
  if (!e) return null;
  try {
    return { ...e, markdown: fs.readFileSync(path.join(dir, e.file), "utf8"), raw: e.raw ? fs.readFileSync(path.join(dir, e.raw), "utf8") : null };
  } catch {
    return null;
  }
}

function keep(dir, url, page, raw, contentType) {
  const file = libraryFile(url);
  const abs = path.join(dir, file);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  const read = new Date().toISOString();
  fs.writeFileSync(abs, `# ${page.title}\n\nsource: ${url}\nread: ${read.slice(0, 10)}\n\n${page.markdown}\n`);
  const rawFile = path.join(".raw", file.replace(/\.md$/, /html/.test(contentType) ? ".html" : ".txt"));
  fs.mkdirSync(path.dirname(path.join(dir, rawFile)), { recursive: true });
  fs.writeFileSync(path.join(dir, rawFile), raw);
  // .raw/ is the source for extraction, not for searching.
  fs.writeFileSync(path.join(dir, ".ignore"), ".raw/\npages.json\n");
  const words = page.markdown.split(/\s+/).filter(Boolean).length;
  const entry = { url, title: page.title, file: file.replace(/\\/g, "/"), raw: rawFile.replace(/\\/g, "/"), contentType, read, words, outline: page.outline };
  const c = catalogue(dir);
  c[url] = entry;
  fs.writeFileSync(path.join(dir, "pages.json"), JSON.stringify(c, null, 1));
  return entry;
}

/** Read one URL and keep it. → { entry, links } */
export async function readUrl(url, { local = false, dir = libraryDir() } = {}) {
  const r = await fetchGuarded(url, { local });
  if (r.status >= 400) throw new Error(`${r.url} answered HTTP ${r.status}`);
  // Trust the content type; a page served without one is HTML if it opens like HTML.
  // (DCAT-AP 3.0.1 has no <html> tag and begins with several KB of script.)
  const html = /text\/html|application\/xhtml/i.test(r.contentType) || /^\s*(<!doctype html|<html|<head)/i.test(r.body);
  const page = html ? htmlToPage(r.body, r.url) : textToPage(r.body, r.url, r.contentType);
  return { entry: keep(dir, r.url, page, r.body, r.contentType), links: page.links };
}

const SKIP = /\.(png|jpe?g|gif|svg|webp|ico|css|js|mjs|map|pdf|zip|gz|tgz|woff2?|ttf|eot|mp4|mp3|json|xml|ttl|jsonld|owl|rdf|csv|xlsx?)$/i;

/** Pages of a site under `start`'s path, breadth-first. → { pages: [entry], skipped, errors } */
export async function crawl(start, { limit = 30, local = false, dir = libraryDir() } = {}) {
  const base = new URL(start);
  const prefix = base.pathname.endsWith("/") ? base.pathname : base.pathname.replace(/[^/]*$/, "");
  const seen = new Set();
  const queue = [base.toString()];
  const pages = [];
  const errors = [];
  let skipped = 0;
  while (queue.length && pages.length < limit) {
    const url = queue.shift();
    if (seen.has(url)) continue;
    seen.add(url);
    try {
      const { entry, links } = await readUrl(url, { local, dir });
      pages.push(entry);
      for (const l of links) {
        const u = new URL(l);
        if (u.origin !== base.origin || !u.pathname.startsWith(prefix) || SKIP.test(u.pathname) || seen.has(u.toString())) continue;
        queue.push(u.toString());
      }
      await new Promise((r) => setTimeout(r, 150)); // be polite to the site
    } catch (e) {
      errors.push({ url, error: e.message });
    }
  }
  skipped = queue.filter((u) => !seen.has(u)).length;
  return { pages, skipped, errors };
}

// ── search ───────────────────────────────────────────────────────────────

const decodeEntities = (s) => cheerio.load(`<p>${s}</p>`)("p").text();

export function parseDuckDuckGo(html) {
  const $ = cheerio.load(html);
  const out = [];
  $(".result").each((_, r) => {
    const a = $(r).find("a.result__a").first();
    let href = a.attr("href") || "";
    const m = /[?&]uddg=([^&]+)/.exec(href);
    if (m) href = decodeURIComponent(m[1]);
    if (!/^https?:/.test(href) || /duckduckgo\.com\/y\.js/.test(href)) return;
    out.push({ title: a.text().trim(), url: href, snippet: $(r).find(".result__snippet").text().replace(/\s+/g, " ").trim() });
  });
  return out;
}

/** → { engine, results: [{ title, url, snippet }] } */
export async function search(query, { limit = 10, env = process.env } = {}) {
  const q = String(query || "").trim();
  if (!q) return { engine: null, results: [] };
  if (env.SEARXNG_URL) {
    const u = new URL("/search", env.SEARXNG_URL);
    u.searchParams.set("q", q);
    u.searchParams.set("format", "json");
    const r = await fetch(u, { headers: { "User-Agent": UA }, signal: AbortSignal.timeout(20_000) });
    const j = await r.json();
    return { engine: "searxng", results: (j.results || []).slice(0, limit).map((x) => ({ title: x.title, url: x.url, snippet: x.content || "" })) };
  }
  if (env.BRAVE_SEARCH_KEY) {
    const u = new URL("https://api.search.brave.com/res/v1/web/search");
    u.searchParams.set("q", q);
    const r = await fetch(u, { headers: { Accept: "application/json", "X-Subscription-Token": env.BRAVE_SEARCH_KEY }, signal: AbortSignal.timeout(20_000) });
    if (!r.ok) throw new Error(`brave answered HTTP ${r.status}`);
    const j = await r.json();
    return { engine: "brave", results: (j.web?.results || []).slice(0, limit).map((x) => ({ title: decodeEntities(x.title), url: x.url, snippet: decodeEntities(x.description || "") })) };
  }
  const r = await fetch("https://html.duckduckgo.com/html/", {
    method: "POST",
    headers: { "User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ q }).toString(),
    signal: AbortSignal.timeout(20_000),
  });
  if (!r.ok) throw new Error(`duckduckgo answered HTTP ${r.status}`);
  return { engine: "duckduckgo", results: parseDuckDuckGo(await r.text()).slice(0, limit) };
}

/** Index the library with spraypaint, so what was read can be searched with a verdict. */
export async function reindexLibrary(dir = libraryDir()) {
  const bin = findBinary("spraypaint", "SPRAYPAINT_CLI");
  if (!bin) return { ok: false, error: "spraypaint is not installed, so what you read cannot be searched with a verdict" };
  fs.mkdirSync(path.join(dir, ".spraypaint"), { recursive: true });
  const r = await run(bin, ["index", "--root", dir, "--json"], { timeoutMs: 600_000 });
  return r.code === 0 ? { ok: true, ...(parseJsonLoose(r.stdout) || {}) } : { ok: false, error: r.stderr.trim().slice(0, 300) };
}
