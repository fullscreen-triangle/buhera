/* ============================================================================
 * Surface — the blank screen.
 *
 * Nothing is drawn but a caret. There is no input box, no placeholder, no
 * banner, no chrome: the user writes, and Enter commits the step. Every step
 * becomes its own page (lib/surface/book.js); going back is flipping pages,
 * and a page is an inert snapshot of what was on screen.
 *
 * Presentation states (blank-screen-interceptor.tex §single-surface):
 *   B   blank page, nothing written          — a caret on black
 *   P   blank page, writing                  — the words, centred
 *   A   viewing a page                       — the page; caret hidden
 *   AP  viewing a page, writing              — the page, words along the foot
 * Writing while on an older page forks to the end: the new page is appended
 * after the latest one; nothing is rewritten.
 *
 * Edges: the pointer at the top, right, bottom or left edge opens that edge's
 * drawer (components/surface/EdgeDrawer.js; the filing is lib/surface/edges.js).
 * Picking a module opens its page as a new step.
 *
 * Flipping: PageUp / PageDown, Alt+← / Alt+→, or ← / → and Home / End when
 * nothing is written; a horizontal swipe on a trackpad.
 *
 * What the next step may know: kernel memory, and the page being viewed when
 * the step began — passed to the resolver (lib/surface/resolve.js), which is
 * where the intent layer plugs in.
 * ========================================================================== */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { bootstrapFederation } from "@/lib/runtime/bootstrap";
import { listModules } from "@/lib/modules/registry";
import { appendPage, loadBook, pageContext, saveBook } from "@/lib/surface/book";
import { createSurfaceRuntime, openModule, resolve } from "@/lib/surface/resolve";
import PageView from "@/components/surface/PageView";
import EdgeDrawer from "@/components/surface/EdgeDrawer";

// Pointer must sit within EDGE_PX of an edge for EDGE_DWELL_MS before the
// drawer opens, so crossing an edge on the way somewhere does not flash one.
const EDGE_PX = 3;
const EDGE_DWELL_MS = 140;

// Routes a line-router "external" envelope navigates to.
const EXTERNAL_ROUTES = { tutorials: "/tutorials", experiment: "/protein-modelling" };

function edgeAt(x, y) {
  if (y <= EDGE_PX) return "top";
  if (y >= window.innerHeight - 1 - EDGE_PX) return "bottom";
  if (x <= EDGE_PX) return "left";
  if (x >= window.innerWidth - 1 - EDGE_PX) return "right";
  return null;
}

export default function Surface() {
  const runtimeRef = useRef(null);
  const inputRef = useRef(null);
  // The book is read synchronously on first render (this component is
  // client-only), so no effect ever sees — and persists — an empty book
  // in place of the stored one.
  const [book, setBook] = useState(() => loadBook());
  const bookRef = useRef(book);                   // latest book, for commit()
  // Open on a blank page, with the history behind it.
  const [view, setView] = useState(() => book.pages.length); // pages.length ⇒ blank
  const [dir, setDir] = useState(1);              // last flip direction, for the turn
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(null);   // words of the step being resolved
  const [edge, setEdge] = useState(null);
  const [registered, setRegistered] = useState([]);
  const [showFolio, setShowFolio] = useState(false);

  const pages = book.pages;
  const onBlank = view >= pages.length;
  const current = onBlank ? null : pages[view];
  const busy = pending != null;

  // ── boot ────────────────────────────────────────────────────────────────
  useEffect(() => {
    const cleanup = bootstrapFederation();
    runtimeRef.current = createSurfaceRuntime();
    setRegistered(listModules());
    return () => cleanup();
  }, []);

  useEffect(() => {
    saveBook(book);
  }, [book]);

  const focusCaret = useCallback(() => {
    if (!edge) inputRef.current?.focus();
  }, [edge]);

  useEffect(() => { focusCaret(); }, [focusCaret, view, busy]);

  // ── flipping ────────────────────────────────────────────────────────────
  const folioTimer = useRef(null);
  const flipTo = useCallback((next) => {
    const clamped = Math.max(0, Math.min(pages.length, next));
    if (clamped === view) return;
    setDir(clamped > view ? 1 : -1);
    setView(clamped);
    setShowFolio(true);
    clearTimeout(folioTimer.current);
    folioTimer.current = setTimeout(() => setShowFolio(false), 1400);
  }, [pages.length, view]);

  // Trackpad swipe: accumulate horizontal wheel delta, flip once per gesture.
  const swipe = useRef({ acc: 0, lock: 0 });
  useEffect(() => {
    const onWheel = (e) => {
      if (edge || Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
      const s = swipe.current;
      const now = Date.now();
      if (now < s.lock) return;
      s.acc += e.deltaX;
      if (Math.abs(s.acc) > 120) {
        flipTo(view + (s.acc > 0 ? 1 : -1));
        s.acc = 0;
        s.lock = now + 500;
      }
    };
    window.addEventListener("wheel", onWheel, { passive: true });
    return () => window.removeEventListener("wheel", onWheel);
  }, [edge, flipTo, view]);

  useEffect(() => {
    const onKey = (e) => {
      if (edge || e.defaultPrevented) return;
      const t = e.target;
      if (t === inputRef.current || (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA"))) return;
      if (e.ctrlKey || e.metaKey) return;
      if (e.key.length === 1 && !e.altKey && !busy) {
        e.preventDefault();
        setDraft((d) => d + e.key);
        inputRef.current?.focus();
      } else if (e.key === "PageUp" || e.key === "ArrowLeft") {
        e.preventDefault(); flipTo(view - 1);
      } else if (e.key === "PageDown" || e.key === "ArrowRight") {
        e.preventDefault(); flipTo(view + 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [edge, busy, flipTo, view]);

  // ── edges ───────────────────────────────────────────────────────────────
  const dwell = useRef({ edge: null, timer: null });
  useEffect(() => {
    const onMove = (e) => {
      if (edge) return; // an open drawer closes on its own mouseleave
      const at = edgeAt(e.clientX, e.clientY);
      const d = dwell.current;
      if (at === d.edge) return;
      clearTimeout(d.timer);
      d.edge = at;
      if (at) {
        d.timer = setTimeout(() => {
          setRegistered(listModules());
          setEdge(at);
        }, EDGE_DWELL_MS);
      }
    };
    window.addEventListener("mousemove", onMove);
    return () => {
      window.removeEventListener("mousemove", onMove);
      clearTimeout(dwell.current.timer);
    };
  }, [edge]);

  const closeEdge = useCallback(() => {
    dwell.current.edge = null;
    setEdge(null);
  }, []);

  useEffect(() => { if (!edge) inputRef.current?.focus(); }, [edge]);

  // ── committing a step ───────────────────────────────────────────────────
  // Every step appends one page and shows it. When the step began on an
  // older page (not the latest), `from` records it, so the fork is legible
  // on the new page. The resolver still sees whichever page was in view.
  async function commit(source, produce) {
    if (busy) return;
    const latest = bookRef.current.pages.length;
    const from = current && current.n !== latest ? current.n : null;
    setPending(source.type === "module" ? source.moduleId : source.text);
    setDraft("");
    const envelope = await produce();
    setPending(null);
    if (!envelope || envelope.kind === "noop") return;
    if (envelope.kind === "external" && EXTERNAL_ROUTES[envelope.meta]) {
      const w = window.open(EXTERNAL_ROUTES[envelope.meta], "_blank", "noopener");
      if (!w) window.location.href = EXTERNAL_ROUTES[envelope.meta];
    }
    const next = appendPage(bookRef.current, { source, envelope, from });
    bookRef.current = next;
    setBook(next);
    setDir(1);
    setView(next.pages.length - 1);
  }

  function commitUtterance(text) {
    const words = text.trim();
    if (!words) return;
    const page = pageContext(current);
    commit({ type: "utterance", text: words }, () =>
      resolve(words, { runtime: runtimeRef.current, page })
    );
  }

  function pickModule(moduleId) {
    closeEdge();
    commit({ type: "module", moduleId }, () => openModule(moduleId));
  }

  // A module page's action: runnable as written → a new step; a template →
  // put it at the caret for the user to complete.
  function act(instruction, template) {
    if (template) {
      setDraft(instruction);
      requestAnimationFrame(() => inputRef.current?.focus());
      return;
    }
    commitUtterance(instruction);
  }

  // ── keys ────────────────────────────────────────────────────────────────
  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      commitUtterance(draft);
      return;
    }
    if (e.key === "Escape") {
      if (draft) { e.preventDefault(); setDraft(""); }
      return;
    }
    const empty = draft.length === 0;
    if (e.key === "PageUp" || (e.altKey && e.key === "ArrowLeft") || (empty && e.key === "ArrowLeft")) {
      e.preventDefault(); flipTo(view - 1);
    } else if (e.key === "PageDown" || (e.altKey && e.key === "ArrowRight") || (empty && e.key === "ArrowRight")) {
      e.preventDefault(); flipTo(view + 1);
    } else if (empty && e.key === "Home") {
      e.preventDefault(); flipTo(0);
    } else if (empty && e.key === "End") {
      e.preventDefault(); flipTo(pages.length);
    }
  }

  function onInput(e) {
    setDraft(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = e.target.scrollHeight + "px";
  }

  // ── drawing ─────────────────────────────────────────────────────────────
  const writing = draft.length > 0 || busy;
  // The caret shows on a blank page always, and on a page only once writing.
  const caretVisible = onBlank || writing;

  const words = (
    <textarea
      ref={inputRef}
      value={busy ? pending : draft}
      onChange={onInput}
      onKeyDown={onKeyDown}
      readOnly={busy}
      rows={1}
      spellCheck={false}
      autoFocus
      aria-label="write"
      className={`w-full bg-transparent border-none outline-none resize-none font-mono text-sm leading-relaxed ${
        busy ? "text-gray-500 animate-pulse" : "text-gray-200"
      }`}
      style={{ caretColor: caretVisible && !busy ? "#2a9d8f" : "transparent" }}
    />
  );

  const folio = useMemo(
    () => (onBlank ? `${pages.length + 1}` : `${view + 1} / ${pages.length}`),
    [onBlank, pages.length, view]
  );

  return (
    <div
      className="fixed inset-0 bg-black text-gray-300 font-mono text-sm leading-relaxed overflow-hidden"
      onMouseDown={(e) => {
        // Clicking empty surface returns to writing; clicks on page content
        // (selecting text, actions) are left alone.
        if (e.target === e.currentTarget) { e.preventDefault(); focusCaret(); }
      }}
    >
      {onBlank ? (
        // B / P — the words sit where a first line would, nothing else.
        <div key={`blank-${pages.length}`} className={`absolute inset-x-0 top-[38%] mx-auto w-full max-w-3xl px-10 md:px-5 ${dir > 0 ? "turn-fwd" : "turn-back"}`}>
          {words}
        </div>
      ) : (
        // A / AP — the page; words along its foot once writing begins.
        <>
          <div
            key={`page-${current.n}`}
            className={`absolute inset-0 overflow-y-auto no-scrollbar ${dir > 0 ? "turn-fwd" : "turn-back"}`}
            onMouseDown={(e) => { if (e.target === e.currentTarget) { e.preventDefault(); focusCaret(); } }}
          >
            <div className={`mx-auto w-full max-w-4xl px-10 pt-16 md:px-5 ${writing ? "pb-40" : "pb-24"} ${writing ? "opacity-60" : ""} transition-opacity`}>
              <PageView page={current} onAct={act} />
            </div>
          </div>
          <div className={`absolute inset-x-0 bottom-0 ${writing ? "bg-gradient-to-t from-black via-black/95 to-transparent pt-10" : "pointer-events-none"}`}>
            <div className="mx-auto w-full max-w-4xl px-10 pb-8 md:px-5">{words}</div>
          </div>
        </>
      )}

      <div
        className={`fixed bottom-3 right-4 text-[10px] text-gray-700 pointer-events-none transition-opacity duration-500 ${
          showFolio ? "opacity-100" : "opacity-0"
        }`}
      >
        {folio}
      </div>

      <EdgeDrawer edge={edge} registered={registered} onPick={pickModule} onClose={closeEdge} />

      <style jsx global>{`
        .no-scrollbar { scrollbar-width: none; }
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .turn-fwd  { animation: turn-fwd 0.28s ease; }
        .turn-back { animation: turn-back 0.28s ease; }
        @keyframes turn-fwd  { from { opacity: 0; transform: translateX(24px); }  to { opacity: 1; transform: none; } }
        @keyframes turn-back { from { opacity: 0; transform: translateX(-24px); } to { opacity: 1; transform: none; } }
        .drawer-down  { animation: drawer-down 0.18s ease; }
        .drawer-up    { animation: drawer-up 0.18s ease; }
        .drawer-left  { animation: drawer-left 0.18s ease; }
        .drawer-right { animation: drawer-right 0.18s ease; }
        @keyframes drawer-down  { from { transform: translateY(-100%); } to { transform: none; } }
        @keyframes drawer-up    { from { transform: translateY(100%); }  to { transform: none; } }
        @keyframes drawer-left  { from { transform: translateX(100%); }  to { transform: none; } }
        @keyframes drawer-right { from { transform: translateX(-100%); } to { transform: none; } }
      `}</style>
    </div>
  );
}
