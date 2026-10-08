// API route for the landing document (lib/notebook/document.js).
//
//   GET  /api/notebook?name=today              → { markdown, record, can }
//   POST /api/notebook { action: "save", name, markdown }
//   POST /api/notebook { action: "run", name, kind, info, source }
//        → text/event-stream: { type: "text" | "step" | "stdout" | "stderr", text } … { type: "done", ok, output, ms }
//   POST /api/notebook { action: "record", name }  → the last runs, newest first
//
// Only the owner (lib/server/notebook.js). Scripts and file search run only
// for a request from the computer the server runs on.

import crypto from "crypto";
import { isScript, KINDS, STARTER, infoTimeout } from "@/lib/notebook/document";
import {
  appendRecord, filesOutput, findInLibrary, findOutput, loadDocument, owner, readRecord,
  runScript, saveDocument, scriptOutput, searchFiles, searchRoots, validName, webOutput,
} from "@/lib/server/notebook";
import { askClaude, askHuggingFace, pickProvider, providers } from "@/lib/server/notebook-ask";
import { readUrl, reindexLibrary, search } from "@/lib/server/web";

export const config = { api: { bodyParser: { sizeLimit: "8mb" }, responseLimit: false } };

function can(local) {
  const p = providers();
  return { local, claude: p.claude, huggingface: p.huggingface, scripts: local, files: local, roots: local ? searchRoots() : [] };
}

const ON_PC = "this runs on your PC, so only from long-grass running there (npm run dev in long-grass/, then open localhost:3000). The hosted site cannot reach your PC.";

export default async function handler(req, res) {
  const who = await owner(req);
  if (!who.ok) return res.status(who.status).json({ ok: false, error: who.error });
  const fail = (status, error) => res.status(status).json({ ok: false, error });

  const name = String((req.method === "GET" ? req.query.name : req.body?.name) || "today");
  if (!validName(name)) return fail(400, "a document name is lowercase letters, digits and dashes");

  if (req.method === "GET") {
    const markdown = loadDocument(name);
    return res.status(200).json({ ok: true, name, markdown: markdown ?? STARTER, fresh: markdown === null, record: readRecord({ tail: 30, document: name }), can: can(who.local) });
  }
  if (req.method !== "POST") return fail(405, "method not allowed");

  const { action } = req.body ?? {};
  if (action === "save") {
    const md = req.body.markdown;
    if (typeof md !== "string") return fail(400, "markdown is required");
    saveDocument(name, md);
    return res.status(200).json({ ok: true, saved: new Date().toISOString() });
  }
  if (action === "record") return res.status(200).json({ ok: true, record: readRecord({ tail: 200, document: name }) });
  if (action !== "run") return fail(400, `unknown action "${action}"`);

  const { kind, info = "", source } = req.body;
  if (!KINDS[kind]) return fail(400, `no cell kind "${kind}"`);
  if (typeof source !== "string" || !source.trim()) return fail(400, "the cell is empty");
  if ((isScript(kind) || kind === "files") && !who.local) return fail(403, ON_PC);

  // From here on the answer is a stream of events.
  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    // no-transform keeps Next's gzip from holding the stream back; X-Accel-Buffering does the same for proxies.
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    Connection: "keep-alive",
  });
  res.flushHeaders?.();
  const abort = new AbortController();
  res.on("close", () => { if (!res.writableEnded) abort.abort(); });
  const emit = (e) => { if (!res.writableEnded) res.write(`data: ${JSON.stringify(e)}\n\n`); };

  const t0 = Date.now();
  const q = source.trim();
  let result;
  try {
    switch (kind) {
      case "ask": {
        const provider = pickProvider(info);
        if (!provider) throw new Error("no model is set up on this server: add ANTHROPIC_API_KEY (or ANTHROPIC_BASE_URL with ANTHROPIC_AUTH_TOKEN) for Claude, or HUGGINGFACE_API_KEY, to long-grass/.env.local and restart it");
        const r = provider === "claude"
          ? await askClaude(q, { local: who.local, emit, signal: abort.signal })
          : await askHuggingFace(q, { local: who.local, emit, signal: abort.signal });
        result = { ok: true, output: r.output, meta: { provider: r.provider, model: r.model, usage: r.usage } };
        break;
      }
      case "web": {
        const r = await search(q, { limit: 10 });
        result = { ok: true, output: webOutput(q, r), meta: { engine: r.engine, results: r.results.length } };
        break;
      }
      case "read": {
        const urls = q.split(/\s+/).filter((u) => /^https?:\/\//i.test(u));
        if (!urls.length) throw new Error("give the full address, starting with https://");
        const lines = [];
        for (const u of urls.slice(0, 10)) {
          emit({ type: "step", text: `read ${u}` });
          try {
            const { entry } = await readUrl(u, { local: who.local });
            lines.push(`- [${entry.title}](${entry.url}) · ${entry.words.toLocaleString("en")} words · ${entry.outline?.length || 0} sections — kept in your library`);
          } catch (e) {
            lines.push(`- ${u} — could not be read: ${e.message}`);
          }
        }
        emit({ type: "step", text: "indexing the library" });
        const idx = await reindexLibrary();
        if (!idx.ok) lines.push("", `not searchable yet: ${idx.error}`);
        result = { ok: true, output: lines.join("\n") };
        break;
      }
      case "find": {
        const r = await findInLibrary(q);
        result = { ok: r.ok, output: findOutput(q, r), meta: { verdict: r.verdict || null } };
        break;
      }
      case "files": {
        const r = await searchFiles(q);
        result = { ok: true, output: filesOutput(q, r) };
        break;
      }
      default: {
        const r = await runScript(kind, source, { timeoutS: infoTimeout(info), onChunk: (stream, text) => emit({ type: stream, text }) });
        result = { ok: r.code === 0, output: scriptOutput(r), meta: { exit: r.code } };
      }
    }
  } catch (e) {
    result = { ok: false, output: abort.signal.aborted ? "stopped." : `could not run: ${e.message || e}` };
  }

  const ms = Date.now() - t0;
  appendRecord({
    at: new Date().toISOString(),
    document: name,
    kind,
    info,
    source,
    source_sha: crypto.createHash("sha256").update(source).digest("hex").slice(0, 16),
    ok: result.ok,
    ms,
    output: result.output,
    ...(result.meta ? { meta: result.meta } : {}),
    from: who.local ? "this computer" : who.subject,
  });
  emit({ type: "done", ok: result.ok, output: result.output, ms });
  res.end();
}
