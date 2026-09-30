/* ============================================================================
 * Server-side retrieval for the player — four-sided-triangle's individuator.
 *
 * The retrieval half of the RAG player runs here, never in the browser: the
 * individuator (vendor/individuate, sdk-ts at 6090e51) reads local files and
 * keeps a persistent receiver graph on disk. It never answers with a bare
 * passage — every answer carries a grounding status (grounded, single- or
 * two-sourced, contested, declined), which the player passes on unchanged.
 *
 * Folders: RAG_FOLDERS (comma-separated, from the server's environment) are
 * always readable. Folders a client names in its RAG settings are honoured
 * only when the request comes from this machine — a deployed instance must not
 * be told by a browser to read arbitrary paths off its disk.
 *
 * One individuator per (project, folder set), cached for the process; the
 * receiver graph persists under .receivers/<project>.json so it accretes
 * across restarts (the paper's append-only receiver).
 * ========================================================================== */

import path from "node:path";
import fs from "node:fs";
import {
  createIndividuator,
  JsonFilePersistAdapter,
  LocalFileSource,
} from "@four-sided-triangle/individuate";

const RECEIVER_DIR = path.join(process.cwd(), ".receivers");
const _cache = new Map();

export function envFolders() {
  return (process.env.RAG_FOLDERS || "").split(",").map((s) => s.trim()).filter(Boolean);
}

const LOOPBACK = new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"]);

// One forwarded address, bare: quotes, IPv6 brackets and any port removed.
function bareAddress(raw) {
  let a = raw.trim().replace(/^"|"$/g, "");
  const v6 = /^\[([^\]]+)\](?::\d+)?$/.exec(a);
  if (v6) return v6[1];
  if (/^[\d.]+:\d+$/.test(a)) a = a.slice(0, a.lastIndexOf(":"));
  return a;
}

// The addresses a request says it was forwarded for, from every forwarding
// header — or null when a forwarding header is present but yields none, so
// an unreadable header fails closed instead of reading as "no proxy".
function forwardedFor(h) {
  const raw = [];
  let present = false;
  for (const v of [h["x-forwarded-for"], h["x-real-ip"]]) {
    if (v) { present = true; raw.push(...String(v).split(",")); }
  }
  if (h.forwarded) {
    present = true;
    for (const element of String(h.forwarded).split(",")) {
      for (const pair of element.split(";")) {
        const [k, ...v] = pair.split("=");
        if (k.trim().toLowerCase() === "for") raw.push(v.join("="));
      }
    }
  }
  const out = raw.map(bareAddress).filter(Boolean);
  return present && out.length === 0 ? null : out;
}

/**
 * Whether a Next.js API request came from this machine. The socket being
 * loopback is not enough: behind a reverse proxy every request arrives from
 * loopback. So every address in the forwarding chain must be loopback too.
 * Next's own server relays requests with X-Forwarded-For: 127.0.0.1, which
 * passes; Caddy replaces a client's forwarding headers with the client's
 * real address, which does not — otherwise any visitor could name a folder
 * on the server for it to read. (A proxy that passes a client's
 * X-Forwarded-For through unchanged would defeat this; Caddy and nginx's
 * $proxy_add_x_forwarded_for do not.)
 */
export function isLocalRequest(req) {
  const addr = req.socket?.remoteAddress || "";
  if (!LOOPBACK.has(addr)) return false;
  const chain = forwardedFor(req.headers || {});
  return chain !== null && chain.every((a) => LOOPBACK.has(a));
}

const safeName = (s) => String(s || "default").replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 64) || "default";

/**
 * The folders a request may read: the environment's, plus the client's own
 * when local. Folders that do not exist are reported, not silently dropped.
 */
export function resolveFolders(requested, local) {
  const wanted = [...envFolders(), ...(local ? requested || [] : [])];
  const seen = new Set();
  const ok = [];
  const missing = [];
  for (const f of wanted) {
    const abs = path.resolve(String(f));
    if (seen.has(abs)) continue;
    seen.add(abs);
    try {
      if (fs.statSync(abs).isDirectory()) ok.push(abs);
      else missing.push(abs);
    } catch {
      missing.push(abs);
    }
  }
  const refused = !local ? (requested || []).length : 0;
  return { folders: ok, missing, refused };
}

function individuatorFor(project, folders, extensions) {
  const key = JSON.stringify([project, folders, extensions]);
  let ind = _cache.get(key);
  if (!ind) {
    ind = createIndividuator({
      receiverId: safeName(project),
      sources: folders.map((root) => new LocalFileSource({ root, extensions })),
      persist: new JsonFilePersistAdapter(path.join(RECEIVER_DIR, `${safeName(project)}.json`)),
    });
    _cache.set(key, ind);
  }
  return ind;
}

// Trim what goes back to the browser: claims can be long passages.
function clip(s, n = 600) {
  const t = String(s ?? "");
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

/**
 * Ask the project's receiver about `query` over `folders`.
 * Returns { status, claim?, support?, floor?, warning?, classes?, reason?, folders }.
 */
export async function retrieve({ query, project, folders, extensions }) {
  if (!folders.length) {
    return { status: "declined", reason: "no retrieval folders configured (RAG settings, bottom edge)", folders };
  }
  const ind = individuatorFor(project, folders, extensions && extensions.length ? extensions : [".md", ".txt", ".tex"]);
  const a = await ind.ask(query);
  const out = { status: a.status, folders };
  if ("claim" in a) out.claim = clip(a.claim);
  if ("support" in a) out.support = a.support.map((c) => ({ source: c.source, power: Number(c.power.toFixed(3)) }));
  if ("floor" in a) out.floor = a.floor;
  if ("warning" in a) out.warning = a.warning;
  if ("classes" in a) out.classes = a.classes.map((c) => ({ key: c.key, representative: clip(c.representative, 300) }));
  if ("reason" in a) out.reason = a.reason;
  return out;
}
