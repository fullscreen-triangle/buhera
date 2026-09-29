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

/** Whether a Next.js API request came from this machine. */
export function isLocalRequest(req) {
  const addr = req.socket?.remoteAddress || "";
  return addr === "127.0.0.1" || addr === "::1" || addr === "::ffff:127.0.0.1";
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
