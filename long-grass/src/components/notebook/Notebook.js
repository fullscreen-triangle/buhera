/* ============================================================================
 * The landing page: a document you run.
 *
 * Prose and cells (lib/notebook/document.js). A cell runs on the server
 * (pages/api/notebook.js) and streams back what it does; its result replaces
 * the output block beneath it, and the run is added to the record. The
 * command line at the top adds a cell and runs it at once.
 * ========================================================================== */

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { authHeaders } from "@/lib/auth/headers";
import { KINDS, command, isScript, parse, serialize } from "@/lib/notebook/document";
import { MarkdownView } from "@/components/surface/Reader";

let seq = 0;
const withIds = (blocks) => blocks.map((b) => ({ ...b, id: `b${++seq}` }));
const SCRIPT_FENCE = /```(python|py|bash|sh|powershell|node|js)\n([\s\S]*?)```/g;

function useAutosize(ref, value) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + 2}px`;
  }, [ref, value]);
}

/** Read a text/event-stream body. */
async function readEvents(res, onEvent) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (line) { try { onEvent(JSON.parse(line.slice(6))); } catch { /* a partial event; the next read completes it */ } }
    }
  }
}

function Prose({ block, onChange, onRemove }) {
  const [editing, setEditing] = useState(!block.text.trim());
  const [text, setText] = useState(block.text);
  const ref = useRef(null);
  useAutosize(ref, text);
  useEffect(() => { if (editing) ref.current?.focus(); }, [editing]);
  const commit = () => {
    setEditing(false);
    if (!text.trim()) onRemove();
    else if (text !== block.text) onChange({ text });
  };
  if (editing) {
    return (
      <textarea ref={ref} value={text} onChange={(e) => setText(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); commit(); } }}
        placeholder="write — Markdown; Ctrl+Enter or click away to finish"
        className="w-full resize-none bg-transparent border border-gray-800 rounded p-2 text-sm text-gray-200 leading-relaxed focus:outline-none focus:border-gray-600" />
    );
  }
  return (
    <div onClick={() => setEditing(true)} className="cursor-text rounded -mx-2 px-2 hover:bg-white/[0.015]" title="click to edit">
      <MarkdownView markdown={block.text} />
    </div>
  );
}

function Output({ kind, text }) {
  if (isScript(kind)) return <pre className="text-xs font-mono whitespace-pre-wrap break-words text-gray-300 leading-relaxed">{text}</pre>;
  return <MarkdownView markdown={text} />;
}

function Cell({ block, run, can, onChange, onRemove, onMove, onRun, onStop, onAddCell }) {
  const ref = useRef(null);
  useAutosize(ref, block.source);
  const meta = KINDS[block.kind];
  const blocked = (isScript(block.kind) || block.kind === "files") && can && !can.local
    ? "runs on your PC — open long-grass there (localhost:3000)" : null;
  const scripts = useMemo(() => {
    if (block.kind !== "ask" || !block.output) return [];
    return [...block.output.matchAll(SCRIPT_FENCE)].map((m) => ({ lang: m[1], code: m[2].replace(/\n$/, "") }));
  }, [block.kind, block.output]);
  const shown = run?.running ? null : block.output;

  return (
    <div className="group my-5">
      <div className="flex items-center gap-2 text-[11px] text-gray-500 mb-1">
        <select value={block.kind} onChange={(e) => onChange({ kind: e.target.value })}
          className="bg-transparent text-teal-300/90 font-mono focus:outline-none cursor-pointer" title={meta?.hint}>
          {Object.keys(KINDS).map((k) => <option key={k} value={k} className="bg-gray-950">{k}</option>)}
        </select>
        <input value={block.info} onChange={(e) => onChange({ info: e.target.value })} placeholder={block.kind === "ask" ? "claude | hf" : isScript(block.kind) ? "timeout=60" : ""}
          className="w-28 bg-transparent font-mono text-gray-500 placeholder:text-gray-800 focus:outline-none" />
        <span className="flex-1" />
        {blocked && <span className="text-amber-300/70">{blocked}</span>}
        {run?.running
          ? <button onClick={onStop} className="text-rose-300 hover:text-rose-200">■ stop</button>
          : <button onClick={onRun} disabled={!!blocked} className="text-gray-300 hover:text-white disabled:text-gray-700">▶ run</button>}
        <span className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">
          <button onClick={() => onMove(-1)} title="move up" className="hover:text-gray-300">↑</button>
          <button onClick={() => onMove(1)} title="move down" className="hover:text-gray-300">↓</button>
          {block.output !== null && <button onClick={() => onChange({ output: null })} className="hover:text-gray-300">clear</button>}
          <button onClick={onRemove} className="hover:text-rose-300">delete</button>
        </span>
      </div>
      <textarea ref={ref} value={block.source} spellCheck={block.kind === "ask"}
        onChange={(e) => onChange({ source: e.target.value })}
        onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); if (!blocked) onRun(); } }}
        className={`w-full resize-none rounded border border-gray-900 bg-white/[0.025] p-3 text-[13px] leading-relaxed text-gray-100 focus:outline-none focus:border-gray-700 ${block.kind === "ask" ? "font-sans" : "font-mono"}`} />

      {run && (run.running || run.error) && (
        <div className="mt-2 pl-3 border-l-2 border-teal-800/60">
          {run.steps.length > 0 && (
            <div className="text-[11px] font-mono text-gray-500 space-y-0.5 mb-1">
              {run.steps.map((s, i) => <div key={i}>· {s}</div>)}
            </div>
          )}
          {run.live && (isScript(block.kind)
            ? <pre className="text-xs font-mono whitespace-pre-wrap break-words text-gray-300">{run.live}</pre>
            : <div className="text-sm text-gray-300 whitespace-pre-wrap">{run.live}</div>)}
          {run.running && <div className="text-[11px] text-teal-400/80 mt-1 animate-pulse">running · {Math.round((Date.now() - run.t0) / 1000)} s</div>}
          {run.error && <div className="text-xs text-rose-300/90">{run.error}</div>}
        </div>
      )}

      {shown !== null && shown !== undefined && (
        <div className="mt-2 pl-3 border-l-2 border-gray-800">
          <Output kind={block.kind} text={shown} />
          {run?.last && <div className="text-[10px] text-gray-600 mt-1">ran {run.last.at} · {(run.last.ms / 1000).toFixed(1)} s</div>}
          {scripts.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-3 text-[11px]">
              {scripts.map((s, i) => (
                <button key={i} onClick={() => onAddCell(s.lang, s.code)} className="text-teal-300/90 hover:text-teal-200">
                  + add the {s.lang} {scripts.length > 1 ? `(${i + 1}) ` : ""}as a cell
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Between({ onText, onCell }) {
  return (
    <div className="h-4 -my-2 flex items-center justify-center gap-4 text-[11px] text-gray-700 opacity-0 hover:opacity-100 transition-opacity">
      <button onClick={onText} className="hover:text-gray-400">+ text</button>
      <button onClick={onCell} className="hover:text-gray-400">+ cell</button>
    </div>
  );
}

function Record({ name, onClose }) {
  const [rec, setRec] = useState(null);
  const [open, setOpen] = useState(null);
  useEffect(() => {
    fetch("/api/notebook", { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() }, body: JSON.stringify({ action: "record", name }) })
      .then((r) => r.json()).then(setRec).catch((e) => setRec({ ok: false, error: e.message }));
  }, [name]);
  return (
    <aside className="fixed top-0 right-0 h-full w-[26rem] max-w-full bg-gray-950 border-l border-gray-900 overflow-y-auto p-4 z-20">
      <div className="flex items-baseline justify-between mb-1">
        <h2 className="text-sm text-gray-200">record</h2>
        <button onClick={onClose} className="text-xs text-gray-500 hover:text-gray-300">close</button>
      </div>
      <p className="text-[11px] text-gray-600 mb-3">every run of this document, newest first. Clearing an output does not remove it here.</p>
      {!rec && <p className="text-xs text-gray-500">reading…</p>}
      {rec && !rec.ok && <p className="text-xs text-rose-300">{rec.error}</p>}
      {rec?.ok && <p className="text-[11px] text-gray-500 mb-2">{rec.record.count} runs</p>}
      {rec?.ok && rec.record.entries.map((e, i) => (
        <div key={i} className="border-b border-gray-900 py-2">
          <button onClick={() => setOpen(open === i ? null : i)} className="w-full text-left">
            <div className="flex gap-2 text-[11px]">
              <span className="text-gray-500 font-mono">{e.at.slice(5, 16).replace("T", " ")}</span>
              <span className="text-teal-300/80 font-mono">{e.kind}</span>
              <span className={e.ok ? "text-gray-500" : "text-rose-300/80"}>{e.ok ? "" : "failed · "}{(e.ms / 1000).toFixed(1)} s</span>
            </div>
            <div className="text-xs text-gray-300 truncate">{e.source.split("\n")[0]}</div>
          </button>
          {open === i && <div className="mt-2 text-xs"><Output kind={e.kind} text={e.output} /></div>}
        </div>
      ))}
    </aside>
  );
}

export default function Notebook({ name = "today" }) {
  const [blocks, setBlocks] = useState(null);
  const [can, setCan] = useState(null);
  const [error, setError] = useState(null);
  const [runs, setRuns] = useState({});
  const [saved, setSaved] = useState("");
  const [count, setCount] = useState(0);
  const [line, setLine] = useState("");
  const [showRecord, setShowRecord] = useState(false);
  const [, tick] = useState(0);
  const controllers = useRef({});
  const dirty = useRef(false);
  const latest = useRef(null);
  latest.current = blocks;

  useEffect(() => {
    fetch(`/api/notebook?name=${encodeURIComponent(name)}`, { headers: authHeaders() })
      .then((r) => r.json())
      .then((j) => {
        if (!j.ok) { setError(j.error); return; }
        setBlocks(withIds(parse(j.markdown)));
        setCan(j.can);
        setCount(j.record.count);
        setSaved(j.fresh ? "new — saved on your first change" : "saved");
      })
      .catch((e) => setError(e.message));
  }, [name]);

  // While anything runs, redraw once a second for the elapsed time.
  const anyRunning = Object.values(runs).some((r) => r.running);
  useEffect(() => {
    if (!anyRunning) return undefined;
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [anyRunning]);

  const save = useCallback(async () => {
    if (!latest.current) return;
    dirty.current = false;
    setSaved("saving…");
    try {
      const r = await fetch("/api/notebook", { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() }, body: JSON.stringify({ action: "save", name, markdown: serialize(latest.current) }) });
      const j = await r.json();
      setSaved(j.ok ? "saved" : `not saved: ${j.error}`);
    } catch (e) {
      setSaved(`not saved: ${e.message}`);
    }
  }, [name]);

  // Save a moment after the last change.
  useEffect(() => {
    if (!dirty.current) return undefined;
    const t = setTimeout(save, 900);
    return () => clearTimeout(t);
  }, [blocks, save]);

  const update = (fn) => { dirty.current = true; setBlocks((bs) => fn(bs)); };
  const change = (id, patch) => update((bs) => bs.map((b) => (b.id === id ? { ...b, ...patch } : b)));
  const remove = (id) => update((bs) => bs.filter((b) => b.id !== id));
  const move = (id, d) => update((bs) => {
    const i = bs.findIndex((b) => b.id === id);
    const j = i + d;
    if (i < 0 || j < 0 || j >= bs.length) return bs;
    const next = [...bs];
    [next[i], next[j]] = [next[j], next[i]];
    return next;
  });
  const insertAt = (index, block) => update((bs) => [...bs.slice(0, index), block, ...bs.slice(index)]);
  const newCell = (kind, source = "") => ({ id: `b${++seq}`, type: "cell", kind, info: "", source, output: null });
  const newText = () => ({ id: `b${++seq}`, type: "prose", text: "" });

  const setRun = (id, fn) => setRuns((rs) => ({ ...rs, [id]: fn(rs[id]) }));

  const runCell = useCallback(async (id, given) => {
    const block = given || latest.current?.find((b) => b.id === id);
    if (!block || !block.source.trim()) return;
    const ctl = new AbortController();
    controllers.current[id] = ctl;
    setRun(id, (r) => ({ ...r, running: true, t0: Date.now(), steps: [], live: "", error: null }));
    try {
      const res = await fetch("/api/notebook", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ action: "run", name, kind: block.kind, info: block.info, source: block.source }),
        signal: ctl.signal,
      });
      if (!res.ok || !(res.headers.get("content-type") || "").includes("event-stream")) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || `HTTP ${res.status}`);
      }
      await readEvents(res, (e) => {
        if (e.type === "text" || e.type === "stdout" || e.type === "stderr") setRun(id, (r) => ({ ...r, live: (r.live || "") + e.text }));
        else if (e.type === "step") setRun(id, (r) => ({ ...r, steps: [...r.steps, e.text] }));
        else if (e.type === "done") {
          change(id, { output: e.output });
          setCount((n) => n + 1);
          setRun(id, (r) => ({ ...r, running: false, error: null, last: { at: new Date().toLocaleTimeString(), ms: e.ms } }));
        }
      });
      setRun(id, (r) => (r?.running ? { ...r, running: false, error: "the connection closed before the run finished" } : r));
    } catch (e) {
      setRun(id, (r) => ({ ...r, running: false, error: ctl.signal.aborted ? "stopped" : e.message }));
    } finally {
      delete controllers.current[id];
    }
  }, [name]); // eslint-disable-line react-hooks/exhaustive-deps

  const submit = () => {
    const c = command(line);
    if (!c) return;
    const cell = newCell(c.kind, c.source);
    insertAt(latest.current.length, cell);
    setLine("");
    runCell(cell.id, cell);
    setTimeout(() => document.getElementById(cell.id)?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
  };

  const preview = command(line);

  if (error) {
    return (
      <main className="min-h-screen bg-black text-gray-300 flex items-center justify-center p-6">
        <div className="max-w-lg text-sm">
          <p className="text-rose-300 mb-3">{error}</p>
          <Link href="/surface" className="text-teal-300 hover:underline">open the blank surface instead</Link>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-black text-gray-300">
      <header className="sticky top-0 z-10 bg-black/90 backdrop-blur border-b border-gray-900">
        <div className="max-w-3xl mx-auto px-5 pt-3 pb-2">
          <div className="flex items-baseline gap-3 text-[11px] text-gray-500 mb-2">
            <span className="text-gray-200 text-sm">buhera</span>
            <span className="font-mono">{name}.md</span>
            <span>{saved}</span>
            <span className="flex-1" />
            <button onClick={() => setShowRecord(true)} className="hover:text-gray-300">record · {count}</button>
            <Link href="/surface" className="hover:text-gray-300">surface</Link>
            <Link href="/tutorials" className="hover:text-gray-300">tutorials</Link>
          </div>
          <div className="flex items-center gap-2 rounded border border-gray-800 focus-within:border-gray-600 px-3 py-2">
            <span className="text-[11px] font-mono text-teal-300/80 w-14 shrink-0">{preview?.kind || "ask"}</span>
            <input value={line} onChange={(e) => setLine(e.target.value)} autoFocus
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); submit(); } }}
              placeholder="ask anything · web … · read https://… · find … · files … · $ command · py code"
              className="flex-1 bg-transparent text-sm text-gray-100 placeholder:text-gray-700 focus:outline-none" />
          </div>
          {can && (
            <div className="text-[10px] text-gray-600 mt-1.5">
              {can.claude ? "Claude" : can.huggingface ? "Hugging Face (no Claude key on this server)" : "no model set up on this server"}
              {" · "}{can.local ? `on this computer — scripts and files under ${can.roots.join(", ") || "(no search roots)"}` : "hosted — scripts and file search run only from long-grass on your PC"}
            </div>
          )}
        </div>
      </header>

      <div className="max-w-3xl mx-auto px-5 py-6 pb-40">
        {!blocks && <p className="text-xs text-gray-500">opening…</p>}
        {blocks && blocks.map((b, i) => (
          <div key={b.id} id={b.id}>
            <Between onText={() => insertAt(i, newText())} onCell={() => insertAt(i, newCell("ask"))} />
            {b.type === "prose"
              ? <Prose block={b} onChange={(p) => change(b.id, p)} onRemove={() => remove(b.id)} />
              : <Cell block={b} run={runs[b.id]} can={can}
                  onChange={(p) => change(b.id, p)} onRemove={() => remove(b.id)} onMove={(d) => move(b.id, d)}
                  onRun={() => runCell(b.id)} onStop={() => controllers.current[b.id]?.abort()}
                  onAddCell={(lang, code) => insertAt(i + 1, newCell(command(`${lang} x`)?.kind || "bash", code))} />}
          </div>
        ))}
        {blocks && <Between onText={() => insertAt(blocks.length, newText())} onCell={() => insertAt(blocks.length, newCell("ask"))} />}
      </div>

      {showRecord && <Record name={name} onClose={() => setShowRecord(false)} />}
    </main>
  );
}
