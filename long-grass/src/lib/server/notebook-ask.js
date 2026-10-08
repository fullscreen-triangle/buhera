/* ============================================================================
 * An `ask` cell: a question answered from fresh searches, with sources.
 *
 * Claude (ANTHROPIC_API_KEY, or ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN for
 * a gateway) answers by calling tools that run here: a search engine, reading
 * a page into the library, the library itself, and — only when the request
 * comes from the computer the server runs on — that computer's files. It
 * never runs scripts: a script it proposes is written as a cell for you to
 * run. Without a Claude credential, a Hugging Face model answers instead,
 * from searches made before it is asked (it has no tools).
 *
 * The credential is read from the environment only. A profile from
 * `ant auth login` on this machine is deliberately not used.
 * ========================================================================== */

import Anthropic from "@anthropic-ai/sdk";
import fs from "fs";
import path from "path";
import { readUrl, search, libraryDir, reindexLibrary } from "@/lib/server/web";
import { findInLibrary, searchFiles, searchRoots } from "@/lib/server/notebook";

const MODEL = () => process.env.NOTEBOOK_MODEL || "claude-opus-5-5";
const EFFORT = () => process.env.NOTEBOOK_EFFORT || "medium";
const HF_MODEL = () => process.env.HF_NOTEBOOK_MODEL || "Qwen/Qwen2.5-72B-Instruct";
const MAX_TURNS = 14;
const PAGE_CHARS = 14_000;

export function claudeConfigured(env = process.env) {
  return !!(env.ANTHROPIC_API_KEY || env.ANTHROPIC_AUTH_TOKEN);
}

export function providers(env = process.env) {
  return { claude: claudeConfigured(env), huggingface: !!env.HUGGINGFACE_API_KEY };
}

/** `ask hf` / `ask claude` in the cell's info picks one; else Claude when configured. */
export function pickProvider(info, env = process.env) {
  const p = providers(env);
  const want = /\b(hf|huggingface)\b/i.test(info || "") ? "huggingface" : /\bclaude\b/i.test(info || "") ? "claude" : env.NOTEBOOK_PROVIDER || null;
  if (want && p[want]) return want;
  if (want) return null;
  return p.claude ? "claude" : p.huggingface ? "huggingface" : null;
}

function client(env = process.env) {
  return new Anthropic({
    apiKey: env.ANTHROPIC_API_KEY || null,
    authToken: env.ANTHROPIC_API_KEY ? null : env.ANTHROPIC_AUTH_TOKEN || null,
    baseURL: env.ANTHROPIC_BASE_URL || undefined,
    maxRetries: 2,
  });
}

const SYSTEM = `You answer questions for one researcher from a document they run, like a lab notebook. Each question is answered fresh: search, read, then answer.

Use the tools. Search the web for anything current or factual you are not certain of, and read the most relevant pages before relying on them; a search snippet alone is not a source. Search the library (pages the researcher has read before) when the question concerns something they have studied. When files on their computer are available, search them when the question concerns their own work.

Answer in Markdown, concisely, for an expert. Cite every claim that came from a source with a link to the page or the file path. Say plainly what you could not find or verify instead of filling the gap. If a script would answer the question better, write it in a fenced block with its language (python, bash, powershell or node) and say that it can be run as a cell; you cannot run it yourself.`;

function tools(local) {
  const t = [
    { name: "web_search", description: "Search the web with a search engine. Returns titles, addresses and snippets.", input_schema: { type: "object", properties: { query: { type: "string", description: "the search words" } }, required: ["query"] } },
    { name: "read_page", description: "Read a web page (http or https) and return its text as Markdown. The page is kept in the researcher's library.", input_schema: { type: "object", properties: { url: { type: "string" } }, required: ["url"] } },
    { name: "search_library", description: "Search pages the researcher has read before. Returns a coverage verdict (covered: one passage holds every word; partial; declined: the words are not there) and passages with their source.", input_schema: { type: "object", properties: { query: { type: "string", description: "the words the answer would contain" } }, required: ["query"] } },
  ];
  if (local) {
    t.push(
      { name: "search_files", description: "Search files on the researcher's computer by path and by content; every word must appear. Returns paths, and lines with their line numbers.", input_schema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } },
      { name: "read_file", description: "Read a text file on the researcher's computer (a path returned by search_files).", input_schema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } },
    );
  }
  // Inputs stream as they are generated; each is checked before it is used.
  return t.map((x) => ({ ...x, eager_input_streaming: true }));
}

const str = (v) => (typeof v === "string" && v.trim() ? v.trim() : null);

/** Run one tool. → { text, note, sources: [{ title, url }], read } */
async function runTool(name, input, { local }) {
  const need = (k) => { const v = str(input?.[k]); if (!v) throw new Error(`${k} is required`); return v; };
  switch (name) {
    case "web_search": {
      const q = need("query");
      const r = await search(q, { limit: 8 });
      return { note: `web "${q}"`, text: JSON.stringify(r.results), sources: [] };
    }
    case "read_page": {
      const url = need("url");
      if (!/^https?:\/\//i.test(url)) throw new Error("only http and https addresses can be read");
      // Never local: a page asking for an address on this network is refused.
      const { entry } = await readUrl(url, { local: false });
      const md = fs.readFileSync(path.join(libraryDir(), entry.file), "utf8");
      const text = md.length > PAGE_CHARS ? `${md.slice(0, PAGE_CHARS)}\n\n[… ${md.length - PAGE_CHARS} more characters not shown]` : md;
      return { note: `read ${url}`, text, sources: [{ title: entry.title, url: entry.url }], read: true };
    }
    case "search_library": {
      const q = need("query");
      const r = await findInLibrary(q);
      return { note: `library "${q}"`, text: JSON.stringify(r), sources: [] };
    }
    case "search_files": {
      if (!local) throw new Error("files are searched only on the researcher's own computer");
      const q = need("query");
      const r = await searchFiles(q);
      return { note: `files "${q}"`, text: JSON.stringify(r), sources: [] };
    }
    case "read_file": {
      if (!local) throw new Error("files are read only on the researcher's own computer");
      const p = path.resolve(need("path"));
      const inside = searchRoots().some((root) => { const rel = path.relative(path.resolve(root), p); return rel && !rel.startsWith("..") && !path.isAbsolute(rel); });
      if (!inside) throw new Error(`only files under ${searchRoots().join(", ")} can be read`);
      const st = fs.statSync(p);
      if (st.size > 400_000) throw new Error("the file is larger than 400 KB");
      const text = fs.readFileSync(p, "utf8");
      return { note: `file ${p}`, text: text.length > PAGE_CHARS ? `${text.slice(0, PAGE_CHARS)}\n[… truncated]` : text, sources: [{ title: path.basename(p), url: null, path: p }] };
    }
    default:
      throw new Error(`no tool named ${name}`);
  }
}

function footer(notes, model) {
  return `\n\n---\n${model} · looked at: ${notes.length ? notes.map((n) => `\`${n.replace(/`/g, "'")}\``).join(" · ") : "nothing — answered from what it knows"}`;
}

/**
 * Ask Claude. `emit(event)` receives { type: "text", text } as the answer
 * streams and { type: "step", text } for each tool call.
 * → { output, provider, model, usage }
 */
export async function askClaude(question, { local, emit = () => {}, signal } = {}) {
  const c = client();
  // NOTEBOOK_FALLBACKS=on|off overrides, for a gateway known to pass betas through.
  const fb = process.env.NOTEBOOK_FALLBACKS;
  const direct = fb ? fb === "on" : !process.env.ANTHROPIC_BASE_URL;
  const today = new Date().toISOString().slice(0, 10);
  const messages = [{ role: "user", content: `Today is ${today}.\n\n${question}` }];
  const notes = [];
  const usage = { input_tokens: 0, output_tokens: 0 };
  let answer = "";
  let readAny = false;

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    const params = {
      model: MODEL(),
      max_tokens: 32000,
      system: SYSTEM,
      thinking: { type: "adaptive" },
      output_config: { effort: EFFORT() },
      tools: tools(local),
      messages,
    };
    // A request declined by a safety classifier is retried on another model,
    // server-side. Only when talking to Anthropic directly: a gateway may not
    // pass the beta through.
    const stream = direct
      ? c.beta.messages.stream({ ...params, betas: ["server-side-fallback-2026-07-01"], fallbacks: "default" }, { signal })
      : c.beta.messages.stream(params, { signal });
    let turnText = "";
    stream.on("text", (delta) => { turnText += delta; emit({ type: "text", text: delta }); });

    let message;
    try {
      message = await stream.finalMessage();
    } catch (err) {
      if (err instanceof Anthropic.APIError) throw err;
      // A tool input that could not be parsed: ask again, once per turn.
      notes.push("(a tool call could not be read and was retried)");
      continue;
    }
    usage.input_tokens += message.usage?.input_tokens || 0;
    usage.output_tokens += message.usage?.output_tokens || 0;
    answer += turnText;

    if (message.stop_reason === "refusal") {
      const why = message.stop_details?.explanation || message.stop_details?.category || "no reason given";
      return { output: `${answer}\n\n*Declined by the model: ${why}.*${footer(notes, message.model)}`, provider: "claude", model: message.model, usage };
    }
    const uses = message.content.filter((b) => b.type === "tool_use");
    if (message.stop_reason === "max_tokens" && uses.length) {
      return { output: `${answer}\n\n*Stopped: the answer ran past its length limit while calling a tool.*${footer(notes, message.model)}`, provider: "claude", model: message.model, usage };
    }
    if (!uses.length || message.stop_reason === "end_turn") {
      if (readAny) await reindexLibrary();
      // The answer is the last turn's text; what came before it was talk between searches.
      const final = turnText.trim() || answer.trim();
      return { output: final + footer(notes, message.model), provider: "claude", model: message.model, usage };
    }

    messages.push({ role: "assistant", content: message.content });
    if (turnText) { answer += "\n\n"; emit({ type: "text", text: "\n\n" }); }
    const results = await Promise.all(uses.map(async (u) => {
      try {
        const r = await runTool(u.name, u.input, { local });
        notes.push(r.note);
        if (r.read) readAny = true;
        emit({ type: "step", text: r.note });
        return { type: "tool_result", tool_use_id: u.id, content: r.text };
      } catch (e) {
        emit({ type: "step", text: `${u.name}: ${e.message}` });
        return { type: "tool_result", tool_use_id: u.id, is_error: true, content: e.message };
      }
    }));
    messages.push({ role: "user", content: results });
  }
  if (readAny) await reindexLibrary();
  return { output: `${answer.trim()}\n\n*Stopped after ${MAX_TURNS} rounds of searching.*${footer(notes, MODEL())}`, provider: "claude", model: MODEL(), usage };
}

/** Ask a Hugging Face model, giving it fresh search results first. */
export async function askHuggingFace(question, { local, emit = () => {}, signal } = {}) {
  const notes = [];
  const context = [];
  emit({ type: "step", text: `web "${question.slice(0, 80)}"` });
  try {
    const r = await search(question, { limit: 8 });
    notes.push(`web "${question.slice(0, 80)}"`);
    context.push("WEB SEARCH RESULTS", ...r.results.map((x, i) => `[${i + 1}] ${x.title} — ${x.url}\n${x.snippet}`));
  } catch (e) {
    context.push(`(web search failed: ${e.message})`);
  }
  const lib = await findInLibrary(question).catch(() => ({ ok: false }));
  if (lib.ok && lib.results.length) {
    notes.push(`library "${question.slice(0, 80)}" (${lib.verdict})`);
    context.push("", `FROM PAGES THE RESEARCHER HAS READ (verdict: ${lib.verdict})`, ...lib.results.slice(0, 4).map((x) => `${x.title} — ${x.url || x.path}\n${x.text.slice(0, 800)}`));
  }
  if (local) {
    const f = await searchFiles(question.split(/\s+/).slice(0, 3).join(" "), { maxMs: 8000 }).catch(() => null);
    if (f?.lines.length) {
      notes.push("files");
      context.push("", "LINES FROM THE RESEARCHER'S FILES", ...f.lines.slice(0, 10).map((x) => `${x.path}:${x.line} ${x.text}`));
    }
  }
  emit({ type: "step", text: `asking ${HF_MODEL()}` });
  const res = await fetch(process.env.HUGGINGFACE_BASE_URL || "https://router.huggingface.co/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}` },
    body: JSON.stringify({
      model: HF_MODEL(),
      max_tokens: 2048,
      temperature: 0.3,
      messages: [
        { role: "system", content: `${SYSTEM}\n\nYou have no tools in this conversation: the searches below were made for you. Cite them by their address.` },
        { role: "user", content: `Today is ${new Date().toISOString().slice(0, 10)}.\n\n${context.join("\n")}\n\nQUESTION\n${question}` },
      ],
    }),
    signal,
  });
  if (!res.ok) throw new Error(`Hugging Face answered HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const body = await res.json();
  const text = String(body?.choices?.[0]?.message?.content || "").trim();
  emit({ type: "text", text });
  return { output: text + footer(notes, HF_MODEL()), provider: "huggingface", model: HF_MODEL(), usage: body?.usage || null };
}
