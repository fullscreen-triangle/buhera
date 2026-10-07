/* ============================================================================
 * Reader — the web, read on the surface: a search's results, a page, a site,
 * the library.
 *
 * A page frame holds only where the page is and its outline; the text is
 * read from the library when the frame is drawn, so a long specification
 * does not fill the book. The text is Markdown made from the page, rendered
 * by our own parser into elements — nothing from a fetched page is ever
 * inserted as HTML. Every section has "+ note": your words, with the
 * passage and its address (url#anchor), kept on a plan.
 * ========================================================================== */

import { useEffect, useMemo, useRef, useState } from "react";
import { parseTutorial } from "@/lib/tutorial-markdown";
import { postJSON } from "@/lib/auth/headers";
import { useStep } from "@/components/surface/actions";
import { Button } from "@/components/surface/controls";
import { KeepOnPlan, NoteOnPlan } from "@/components/surface/Planning";

const ANCHOR = /\s*\{#([^}]+)\}\s*$/;
// Markdown's backslash escapes (turndown writes `0..\*`), shown as the character.
const unescape = (s) => String(s || "").replace(/\\([\\`*_{}[\]()#+\-.!|~<>])/g, "$1");

function Inline({ tokens }) {
  return (tokens || []).map((t, i) => {
    if (t.kind === "code") return <code key={i} className="px-1 rounded bg-white/[0.04] text-teal-100/90 text-[0.92em]">{t.text}</code>;
    if (t.kind === "bold") return <strong key={i} className="text-gray-100">{unescape(t.text)}</strong>;
    if (t.kind === "link") {
      const safe = /^https?:\/\//.test(t.href);
      return safe
        ? <a key={i} href={t.href} target="_blank" rel="noreferrer noopener" className="text-teal-300/90 hover:underline">{unescape(t.text)}</a>
        : <span key={i}>{unescape(t.text)}</span>;
    }
    return <span key={i}>{unescape(t.text)}</span>;
  });
}

/** Markdown as elements. Headings get `data-anchor`; `onHeading` adds controls beside h2/h3. */
export function MarkdownView({ markdown, onHeading, skipTitle = false }) {
  const blocks = useMemo(() => parseTutorial(markdown || ""), [markdown]);
  // Each heading's section: the blocks until the next heading at its level or above, for quoting.
  const textAfter = (i) => {
    const out = [];
    for (let k = i + 1; k < blocks.length && out.join(" ").length < 600; k++) {
      const b = blocks[k];
      if (/^h[1-4]$/.test(b.type)) break;
      if (b.type === "p") out.push(unescape(b.inline.map((t) => t.text).join("")));
    }
    return out.join(" ").slice(0, 600);
  };
  let firstH1 = true;
  return (
    <div className="text-sm text-gray-300 leading-relaxed">
      {blocks.map((b, i) => {
        if (/^h[1-4]$/.test(b.type)) {
          const m = ANCHOR.exec(b.text);
          const text = b.text.replace(ANCHOR, "");
          if (b.type === "h1" && skipTitle && firstH1) { firstH1 = false; return null; }
          const size = { h1: "text-xl mt-6", h2: "text-lg mt-8", h3: "text-base mt-6", h4: "text-sm mt-4" }[b.type];
          return (
            <div key={i} data-anchor={m?.[1] || undefined} className={`${size} mb-2 flex flex-wrap items-baseline gap-x-3`}>
              <span className="text-gray-100 font-semibold">{text}</span>
              {onHeading && (b.type === "h2" || b.type === "h3") && onHeading({ anchor: m?.[1], text, passage: textAfter(i) })}
            </div>
          );
        }
        if (b.type === "p") return <p key={i} className="my-2"><Inline tokens={b.inline} /></p>;
        if (b.type === "ul") return <ul key={i} className="my-2 ml-5 list-disc space-y-0.5">{b.items.map((it, k) => <li key={k}><Inline tokens={it} /></li>)}</ul>;
        if (b.type === "blockquote") return <blockquote key={i} className="my-2 pl-3 border-l-2 border-gray-800 text-gray-400"><Inline tokens={b.inline} /></blockquote>;
        if (b.type === "code") return <pre key={i} className="my-3 p-3 rounded border border-gray-900 bg-white/[0.02] text-xs font-mono whitespace-pre-wrap break-all text-gray-300">{b.code}</pre>;
        if (b.type === "hr") return <hr key={i} className="my-6 border-gray-900" />;
        if (b.type === "table") {
          return (
            <div key={i} className="my-3 overflow-x-auto">
              <table className="text-xs border-collapse">
                <thead><tr>{b.header.map((h, k) => <th key={k} className="text-left font-normal text-gray-500 pr-4 pb-1 border-b border-gray-800 align-bottom"><Inline tokens={h} /></th>)}</tr></thead>
                <tbody>{b.rows.map((r, k) => <tr key={k} className="border-b border-gray-900 align-top">{r.map((c, j) => <td key={j} className="pr-4 py-1"><Inline tokens={c} /></td>)}</tr>)}</tbody>
              </table>
            </div>
          );
        }
        return null;
      })}
    </div>
  );
}

const day = (iso) => (iso ? String(iso).slice(0, 10) : "");

export function WebPage({ url, title, read, words, outline = [] }) {
  const step = useStep();
  const root = useRef(null);
  const [page, setPage] = useState(null);
  const [error, setError] = useState(null);
  const [showOutline, setShowOutline] = useState(outline.length > 8);

  useEffect(() => {
    let alive = true;
    postJSON("/api/web", { action: "page", url }).then((r) => {
      if (!alive) return;
      if (r.ok) setPage(r); else setError(r.error);
    });
    return () => { alive = false; };
  }, [url]);

  const go = (anchor) => root.current?.querySelector(`[data-anchor="${CSS.escape(anchor)}"]`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  const body = page ? page.markdown.replace(/^# .*\n\nsource: .*\nread: .*\n\n/, "") : "";

  return (
    <div ref={root}>
      <div className="text-lg text-gray-100">{title}</div>
      <div className="text-[11px] text-gray-500 mt-1 flex flex-wrap gap-x-3">
        <a href={url} target="_blank" rel="noreferrer noopener" className="font-mono text-teal-300/80 hover:underline break-all">{url}</a>
        <span>read {day(read)}</span>
        {words ? <span>{words.toLocaleString()} words</span> : null}
        <span>{outline.length} sections</span>
      </div>
      {step && (
        <div className="flex flex-wrap gap-2 mt-3">
          <Button onClick={() => step(`diagram ${url}`, "spec", { kind: "diagram", url, view: "classes" })}>class diagram</Button>
          <Button onClick={() => step(`workflow ${url}`, "spec", { kind: "diagram", url, view: "flow" })}>workflow</Button>
          <Button onClick={() => step(`what ${title} defines`, "spec", { kind: "model", url })}>what it defines</Button>
          <KeepOnPlan found={{ source: "web", cite: url, title, snippet: `${words || "?"} words, read ${day(read)}` }} />
        </div>
      )}

      {outline.length > 0 && (
        <div className="mt-4">
          <button type="button" className="text-[11px] uppercase tracking-wider text-gray-600 hover:text-gray-300" onClick={() => setShowOutline((v) => !v)}>
            {showOutline ? "▾" : "▸"} outline
          </button>
          {showOutline && (
            <div className="mt-1 columns-2 md:columns-1 gap-8 text-xs">
              {outline.map((o, i) => (
                <button key={i} type="button" onClick={() => go(o.anchor)} style={{ paddingLeft: `${(o.level - 1) * 0.75}rem` }}
                  className={`block text-left w-full truncate hover:text-white ${o.level <= 2 ? "text-gray-300" : "text-gray-500"}`}>
                  {o.text}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="mt-6 border-t border-gray-900 pt-2">
        {error && <p className="text-sm text-rose-300/80">{error}</p>}
        {!page && !error && <p className="text-xs text-gray-600 animate-pulse">opening the page from the library…</p>}
        {page && (
          <MarkdownView markdown={body} onHeading={({ anchor, text, passage }) => (
            <NoteOnPlan cite={anchor ? `${url}#${anchor}` : url} title={`${title} — ${text}`} snippet={passage} />
          )} />
        )}
      </div>
    </div>
  );
}

export function WebSearch({ query, engine, results = [] }) {
  const step = useStep();
  return (
    <div>
      <div className="text-xs text-gray-500 mb-3">web <span className="text-white">“{query}”</span> · {results.length} results · {engine}</div>
      {results.length === 0 && <p className="text-gray-500">no results.</p>}
      {results.map((r) => (
        <div key={r.url} className="py-2 border-t border-gray-900">
          <div className="flex flex-wrap items-baseline gap-x-3">
            <a href={r.url} target="_blank" rel="noreferrer noopener" className="text-sm text-gray-200 hover:text-white">{r.title || r.url}</a>
            {step && <button type="button" className="text-[11px] text-gray-500 hover:text-teal-300" onClick={() => step(`read ${r.url}`, "web", { kind: "read", url: r.url })}>read</button>}
            <KeepOnPlan found={{ source: "web", cite: r.url, title: r.title, snippet: r.snippet, query }} />
          </div>
          <div className="text-[11px] font-mono text-teal-300/60 break-all">{r.url}</div>
          {r.snippet && <div className="text-xs text-gray-500 mt-0.5">{r.snippet}</div>}
        </div>
      ))}
    </div>
  );
}

function PageRow({ p }) {
  const step = useStep();
  return (
    <div className="flex flex-wrap items-baseline gap-x-3 py-1 border-t border-gray-900">
      <button type="button" className="text-sm text-gray-200 hover:text-white text-left" onClick={() => step?.(`read ${p.url}`, "web", { kind: "page", url: p.url })}>{p.title}</button>
      <span className="text-[11px] text-gray-600">{(p.words || 0).toLocaleString()} words · {day(p.read)}</span>
      <span className="text-[11px] font-mono text-gray-700 break-all">{p.url}</span>
    </div>
  );
}

export function WebSite({ start, limit, pages = [], skipped = 0, errors = [], indexed }) {
  return (
    <div>
      <div className="text-xs text-gray-500 mb-3">
        read {pages.length} page{pages.length === 1 ? "" : "s"} under <span className="font-mono text-gray-300 break-all">{start}</span>
        {skipped > 0 ? ` · ${skipped} more linked pages left unread (limit ${limit})` : ""}
        {indexed ? " · searchable with `find`" : ""}
      </div>
      {pages.map((p) => <PageRow key={p.url} p={p} />)}
      {errors.map((e) => <div key={e.url} className="text-xs text-rose-300/80 break-all">{e.url}: {e.error}</div>)}
    </div>
  );
}

export function WebLibrary({ pages = [] }) {
  const step = useStep();
  const [q, setQ] = useState("");
  return (
    <div>
      <div className="text-xs text-gray-500 mb-3">{pages.length} page{pages.length === 1 ? "" : "s"} read</div>
      {step && pages.length > 0 && (
        <div className="flex gap-2 mb-4">
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="search what you have read…" spellCheck={false}
            onKeyDown={(e) => { if (e.key === "Enter" && q.trim()) step(`find ${q} in what I read`, "web", { kind: "ask", query: q }); }}
            className="flex-1 bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-sm text-gray-200 px-1" />
          <Button disabled={!q.trim()} onClick={() => step(`find ${q} in what I read`, "web", { kind: "ask", query: q })}>search</Button>
        </div>
      )}
      {pages.length === 0 && <p className="text-gray-500 italic">nothing read yet — write <span className="not-italic text-gray-400">read https://…</span></p>}
      {pages.map((p) => <PageRow key={p.url} p={p} />)}
    </div>
  );
}
