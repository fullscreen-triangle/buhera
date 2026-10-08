/* ============================================================================
 * The landing document on disk, its record, and what its cells run.
 *
 * Where: BUHERA_NOTEBOOK_DIR, else ~/.buhera/notebook —
 *   <name>.md          the document (lib/notebook/document.js)
 *   record.jsonl       one line per run, appended, never rewritten
 *   work/              the working directory scripts run in
 *
 * Who: this machine always. From elsewhere only the owner — a signed-in
 * session (lib/server/session.js) whose account is BUHERA_OWNER — because a
 * run spends model credit and reads what you have kept. Scripts and file
 * search touch the computer the server runs on, so they run only for a
 * request from that computer: on the hosted site they would run on the
 * server, not on your PC.
 * ========================================================================== */

import fs from "fs";
import os from "os";
import path from "path";
import { spawn } from "child_process";
import { allowed } from "@/lib/server/session";
import { childEnv } from "@/lib/server/exec-sandbox";
import { findBinary, run } from "@/lib/server/spawn";
import { askArgs, parseJsonLoose } from "@/lib/server/spraypaint";
import { libraryDir, libraryPages } from "@/lib/server/web";

export function notebookDir(env = process.env) {
  const d = env.BUHERA_NOTEBOOK_DIR || path.join(os.homedir(), ".buhera", "notebook");
  return d.replace(/^~(?=$|[\\/])/, os.homedir());
}

const NAME = /^[a-z0-9][a-z0-9-]{0,63}$/;
export const validName = (n) => NAME.test(String(n || ""));

export function loadDocument(name, dir = notebookDir()) {
  const f = path.join(dir, `${name}.md`);
  return fs.existsSync(f) ? fs.readFileSync(f, "utf8") : null;
}

export function saveDocument(name, markdown, dir = notebookDir()) {
  fs.mkdirSync(dir, { recursive: true });
  const f = path.join(dir, `${name}.md`);
  // Write beside, then rename: a crash mid-write never leaves half a document.
  fs.writeFileSync(`${f}.tmp`, markdown);
  fs.renameSync(`${f}.tmp`, f);
}

export function appendRecord(entry, dir = notebookDir()) {
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.join(dir, "record.jsonl"), JSON.stringify(entry) + "\n");
}

/** → { count, entries: the last `tail`, newest first } */
export function readRecord({ tail = 50, document } = {}, dir = notebookDir()) {
  const f = path.join(dir, "record.jsonl");
  if (!fs.existsSync(f)) return { count: 0, entries: [] };
  const all = fs.readFileSync(f, "utf8").split("\n").filter(Boolean).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
  const mine = document ? all.filter((e) => e.document === document) : all;
  return { count: mine.length, entries: mine.slice(-tail).reverse() };
}

/** The account a gateway session belongs to. Only call after the gateway accepted the token. */
export function tokenSubject(authorization) {
  const m = /^Bearer\s+v1\.([A-Za-z0-9_-]+)\./.exec(authorization || "");
  if (!m) return null;
  try {
    const payload = Buffer.from(m[1], "base64url").toString("utf8");
    return payload.split(":")[1] || null;
  } catch {
    return null;
  }
}

/** → { ok: true, local, subject } or { ok: false, status, error } */
export async function owner(req, env = process.env) {
  const who = await allowed(req);
  if (!who.ok || who.local) return who;
  const subject = tokenSubject(req.headers?.authorization);
  const owners = String(env.BUHERA_OWNER || "").split(/[\s,]+/).filter(Boolean);
  if (!owners.length) {
    return { ok: false, status: 403, error: `this document runs only for its owner, and none is set on this server: add BUHERA_OWNER=${subject || "<your account id>"} to its .env.local if that is you` };
  }
  if (!subject || !owners.includes(subject)) return { ok: false, status: 403, error: "this document runs only for its owner" };
  return { ok: true, local: false, subject };
}

// ─── Scripts ────────────────────────────────────────────────────────────────

const DEFAULT_SCRIPT_S = 60;
const MAX_SCRIPT_S = 600;
const MAX_OUTPUT = 1024 * 1024;
const lf = (s) => s.replace(/\r\n/g, "\n");

function bashPath() {
  if (process.platform !== "win32") return "/bin/bash";
  const git = ["C:\\Program Files\\Git\\bin\\bash.exe", "C:\\Program Files (x86)\\Git\\bin\\bash.exe"].find((p) => fs.existsSync(p));
  return git || "bash";
}

/** The interpreter and file extension for a script kind. */
export function interpreter(kind, env = process.env) {
  const win = process.platform === "win32";
  switch (kind) {
    case "python": return { cmd: env.NOTEBOOK_PYTHON || (win ? "python" : "python3"), args: ["-u"], ext: ".py" };
    case "node": return { cmd: process.execPath, args: [], ext: ".mjs" };
    case "bash": return { cmd: env.NOTEBOOK_BASH || bashPath(), args: [], ext: ".sh" };
    case "powershell": return { cmd: win ? "powershell.exe" : "pwsh", args: ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"], ext: ".ps1" };
    default: return null;
  }
}

/**
 * Run a script in notebook/work. The child gets an allowlisted environment,
 * never the server's own (which holds API keys and mail passwords), so a
 * script cannot print them into the document. Never throws.
 * → { stdout, stderr, code, ms, timed_out, truncated }
 */
export function runScript(kind, code, { timeoutS, onChunk, dir = notebookDir() } = {}) {
  const it = interpreter(kind);
  if (!it) return Promise.resolve({ stdout: "", stderr: `no interpreter for ${kind}`, code: -1, ms: 0 });
  const work = path.join(dir, "work");
  fs.mkdirSync(work, { recursive: true });
  const file = path.join(work, `.cell-${process.pid}-${Date.now()}${it.ext}`);
  fs.writeFileSync(file, code.endsWith("\n") ? code : `${code}\n`);
  const limit = Math.min(MAX_SCRIPT_S, Math.max(1, Number(timeoutS) || DEFAULT_SCRIPT_S)) * 1000;
  const env = childEnv({ PYTHONIOENCODING: "utf-8", PYTHONUTF8: "1", NOTEBOOK_WORK: work });

  return new Promise((resolve) => {
    const t0 = Date.now();
    const out = [];
    const err = [];
    let bytes = 0;
    let truncated = false;
    let timedOut = false;
    const done = (code) => {
      clearTimeout(timer);
      fs.rm(file, { force: true }, () => {});
      resolve({ stdout: lf(Buffer.concat(out).toString("utf8")), stderr: lf(Buffer.concat(err).toString("utf8")), code, ms: Date.now() - t0, timed_out: timedOut, truncated });
    };
    let child;
    try {
      child = spawn(it.cmd, [...it.args, file], { cwd: work, env, windowsHide: true });
    } catch (e) {
      fs.rm(file, { force: true }, () => {});
      resolve({ stdout: "", stderr: `could not start ${it.cmd}: ${e.message}`, code: -1, ms: 0 });
      return;
    }
    const timer = setTimeout(() => { timedOut = true; child.kill(); }, limit);
    const take = (sink, stream) => (chunk) => {
      bytes += chunk.length;
      if (bytes > MAX_OUTPUT) { if (!truncated) { truncated = true; child.kill(); } return; }
      sink.push(chunk);
      onChunk?.(stream, lf(chunk.toString("utf8")));
    };
    child.stdout.on("data", take(out, "stdout"));
    child.stderr.on("data", take(err, "stderr"));
    child.on("error", (e) => { err.push(Buffer.from(`could not start ${it.cmd}: ${e.message}\n`)); done(-1); });
    child.on("close", (code) => done(code ?? -1));
  });
}

/** A script's result as an output block. */
export function scriptOutput(r) {
  const parts = [];
  if (r.stdout.trim()) parts.push(r.stdout.replace(/\s+$/, ""));
  if (r.stderr.trim()) parts.push(`stderr:\n${r.stderr.replace(/\s+$/, "")}`);
  const tail = [`exit ${r.code}`, `${(r.ms / 1000).toFixed(1)} s`];
  if (r.timed_out) tail.push("stopped: took too long (set timeout=<seconds> after the cell's language)");
  if (r.truncated) tail.push("stopped: more than 1 MB of output");
  parts.push(tail.join(" · "));
  return parts.join("\n\n");
}

// ─── Files on this computer ─────────────────────────────────────────────────

const SKIP_DIRS = new Set(["node_modules", ".git", "target", ".next", "dist", "build", "__pycache__", ".venv", "venv", ".cargo", ".cargo-home", ".claude", ".purpose", ".spraypaint", "site-packages", "AppData", "$RECYCLE.BIN"]);
const TEXT = /\.(md|txt|tex|bib|py|js|mjs|ts|tsx|jsx|rs|toml|json|ya?ml|csv|tsv|html?|css|sh|ps1|r|jl|ipynb|org|rst|xml|ttl|sql|c|h|cpp|hpp|java|go|rb|ndo|hj|shk|ss|vh)$/i;

export function searchRoots(env = process.env) {
  const raw = env.NOTEBOOK_SEARCH_ROOTS;
  const roots = raw ? raw.split(path.delimiter) : [path.join(os.homedir(), "Documents")];
  return roots.map((r) => r.trim().replace(/^~(?=$|[\\/])/, os.homedir())).filter((r) => r && fs.existsSync(r));
}

/**
 * Files whose path holds every word, then lines that hold every word.
 * Bounded in files visited and time; says so when it stopped early.
 * → { roots, names: [path], lines: [{ path, line, text }], visited, stopped }
 */
export async function searchFiles(query, { roots = searchRoots(), maxFiles = 60_000, maxMs = 20_000, limit = 40 } = {}) {
  const words = String(query || "").toLowerCase().split(/\s+/).filter(Boolean);
  const names = [];
  const lines = [];
  let visited = 0;
  let stopped = null;
  const t0 = Date.now();
  const stack = [...roots];
  while (stack.length) {
    if (visited >= maxFiles) { stopped = `stopped after ${maxFiles} files`; break; }
    if (Date.now() - t0 > maxMs) { stopped = `stopped after ${maxMs / 1000} s`; break; }
    const d = stack.pop();
    let entries;
    try { entries = await fs.promises.readdir(d, { withFileTypes: true }); } catch { continue; }
    const texts = [];
    for (const e of entries) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) { if (!SKIP_DIRS.has(e.name) && !e.name.startsWith(".")) stack.push(p); continue; }
      if (!e.isFile()) continue;
      visited++;
      const lower = p.toLowerCase();
      if (names.length < limit && words.every((w) => lower.includes(w))) names.push(p);
      if (TEXT.test(e.name)) texts.push(p);
    }
    // A folder's text files are read together: the walk waits on the disk, not the CPU.
    for (let k = 0; k < texts.length && lines.length < limit; k += 32) {
      const found = await Promise.all(texts.slice(k, k + 32).map(async (p) => {
        try {
          const st = await fs.promises.stat(p);
          if (st.size > 1024 * 1024) return [];
          const text = await fs.promises.readFile(p, "utf8");
          const low = text.toLowerCase();
          if (!words.every((w) => low.includes(w))) return [];
          return text.split(/\r?\n/).flatMap((l, i) => (words.every((w) => l.toLowerCase().includes(w)) ? [{ path: p, line: i + 1, text: l.trim().slice(0, 240) }] : []));
        } catch {
          return [];
        }
      }));
      for (const hits of found) for (const x of hits) if (lines.length < limit) lines.push(x);
    }
  }
  return { roots, names, lines, visited, stopped, ms: Date.now() - t0 };
}

export function filesOutput(query, r) {
  const out = [`files "${query}" · ${r.names.length} by name · ${r.lines.length} lines · ${r.visited} files looked at in ${r.roots.join(", ")}${r.stopped ? ` · ${r.stopped}` : ""}`];
  if (r.names.length) out.push("", "**by name**", ...r.names.map((p) => `- \`${p}\``));
  if (r.lines.length) out.push("", "**by content**", ...r.lines.map((x) => `- \`${x.path}:${x.line}\` — ${x.text.replace(/`/g, "'")}`));
  if (!r.names.length && !r.lines.length) out.push("", "nothing holds every word. A miss here is not proof: only text files under 1 MB are read, and some folders are skipped.");
  return out.join("\n");
}

// ─── What you have read ─────────────────────────────────────────────────────

/** spraypaint over the library. → { ok, verdict, reason, results: [{ path, url, title, lines, text }] } */
export async function findInLibrary(query, { dir = libraryDir(), budget = 6 } = {}) {
  if (!fs.existsSync(path.join(dir, ".spraypaint", "index.json"))) return { ok: false, error: "nothing has been read yet — read a page first" };
  const bin = findBinary("spraypaint", "SPRAYPAINT_CLI");
  if (!bin) return { ok: false, error: "spraypaint is not installed on this server" };
  const r = await run(bin, askArgs({ query, root: dir, budget, dryRun: false }), { timeoutMs: 120_000 });
  if (r.code !== 0) return { ok: false, error: `spraypaint exited with ${r.code}: ${r.stderr.trim().slice(0, 200)}` };
  const parsed = parseJsonLoose(r.stdout);
  if (!parsed) return { ok: false, error: "could not read spraypaint's output" };
  const byFile = new Map(libraryPages(dir).map((p) => [p.file, p]));
  const results = (parsed.results || []).map((x) => {
    const page = byFile.get(x.path);
    const from = x.evidence_start_line ?? x.start_line;
    const to = x.evidence_end_line ?? x.end_line;
    return { path: x.path, url: page?.url || null, title: page?.title || x.path, lines: `${from}-${to}`, matched: x.matched_terms || [], text: String(x.snippet || "").trim() };
  });
  return { ok: true, verdict: parsed.coverage?.verdict || null, reason: parsed.coverage?.reason || "", missing: (parsed.coverage?.terms || []).filter((t) => t.df === 0).map((t) => t.term), results };
}

export function findOutput(query, r) {
  if (!r.ok) return r.error;
  const out = [`find "${query}" — **${r.verdict || "no verdict"}**${r.reason ? `: ${r.reason}` : ""}`];
  if (r.missing?.length) out.push(`not in anything you have read: ${r.missing.join(", ")}`);
  for (const x of r.results) {
    out.push("", `**${x.title}** ${x.url ? `— [${x.url}](${x.url})` : ""} · lines ${x.lines}${x.matched.length ? ` · matched ${x.matched.join(", ")}` : ""}`);
    out.push("", ...x.text.split("\n").slice(0, 6).map((l) => `> ${l}`));
  }
  return out.join("\n");
}

export function webOutput(query, r) {
  const out = [`web "${query}" · ${r.results.length} results · ${r.engine} · ${new Date().toISOString().slice(0, 16).replace("T", " ")} UTC`];
  for (const x of r.results) out.push("", `- [${x.title || x.url}](${x.url})${x.snippet ? ` — ${x.snippet}` : ""}`);
  if (!r.results.length) out.push("", "no results.");
  return out.join("\n");
}
