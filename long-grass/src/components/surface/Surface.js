/* ============================================================================
 * Surface — the blank screen.
 *
 * Nothing is drawn but a caret. The user writes — in any syntax, any words —
 * and Enter commits the step to the player (lib/surface/player.js): a script
 * in a DSL runs as written; anything else becomes a vaHera script written by
 * retrieval and the user's personal model, which runs and lays a line in the
 * runtime graph.
 *
 * Every step becomes a frame. The frames stack into one continuous strip that
 * the user scrolls through; the live screen — blank, with the caret — is
 * always the last frame. A frame is an inert snapshot: the surface never reads
 * it, labels it or decides which one matters. The user does, by cutting:
 * press the scroll wheel (or Alt) and drag over any part of any frame, and
 * that part lifts onto the live screen as a live piece (components/surface/
 * Pieces.js). Pieces from many frames make one screen — many "applications"
 * at once, with no windows.
 *
 * Edges: the pointer at an edge opens that edge's drawer (EdgeDrawer.js,
 * filing in lib/surface/edges.js); picking an item opens its frame, the item's
 * mark flying from the drawer to the frame's head.
 *
 * Preferences (text size, spacing, column, motion) apply here, at the root.
 * ========================================================================== */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, MotionConfig, motion } from "framer-motion";
import { bootstrapFederation } from "@/lib/runtime/bootstrap";
import { listModules } from "@/lib/modules/registry";
import { appendPage, loadBook, pageContext, saveBook } from "@/lib/surface/book";
import { edgeOf } from "@/lib/surface/edges";
import { createSurfaceRuntime, openModule, resolve, setResolver } from "@/lib/surface/resolve";
import { play } from "@/lib/surface/player";
import { useSettings } from "@/lib/surface/settings";
import { ModuleHead } from "@/components/surface/PageView";
import EdgeDrawer from "@/components/surface/EdgeDrawer";
import { SurfaceActions } from "@/components/surface/actions";
import { FrameContent, PieceLayer } from "@/components/surface/Pieces";
import { loadPieces, pieceFromDrag, savePieces } from "@/lib/surface/pieces";

const EDGE_PX = 3;
const EDGE_DWELL_MS = 140;
const EXTERNAL_ROUTES = { tutorials: "/tutorials", experiment: "/protein-modelling" };
const GLIDE = { type: "spring", stiffness: 320, damping: 36, mass: 0.9 };
const COLUMN = { narrow: "max-w-3xl", normal: "max-w-4xl", wide: "max-w-6xl" };

function edgeAt(x, y) {
  if (y <= EDGE_PX) return "top";
  if (y >= window.innerHeight - 1 - EDGE_PX) return "bottom";
  if (x <= EDGE_PX) return "left";
  if (x >= window.innerWidth - 1 - EDGE_PX) return "right";
  return null;
}

// ── one frame of the strip ───────────────────────────────────────────────
//
// Mounted only near the viewport; far away it keeps its measured height as a
// placeholder, so a long book stays light and the scroll position stays put.

function Frame({ page, column, onAct, fly, stripRef, registerContent }) {
  const ref = useRef(null);
  const contentRef = useRef(null);
  const [near, setNear] = useState(true);
  const [height, setHeight] = useState(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting && el.offsetHeight) setHeight(el.offsetHeight);
      setNear(e.isIntersecting);
    }, { root: stripRef.current, rootMargin: "150% 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [stripRef]);

  useEffect(() => {
    registerContent(page.n, contentRef.current);
    return () => registerContent(page.n, null);
  });

  return (
    <section ref={ref} data-frame={page.n} className="relative min-h-screen border-t border-gray-900/80"
      style={!near && height ? { height } : undefined}>
      <span className="absolute top-2 right-4 text-[10px] text-gray-800 select-none" aria-hidden="true">{page.n}</span>
      {near && (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={GLIDE}
          className={`mx-auto w-full ${column}`}>
          <FrameContent page={page} onAct={onAct} fly={fly} contentRef={contentRef} />
        </motion.div>
      )}
    </section>
  );
}

export default function Surface() {
  const settings = useSettings();
  const prefs = settings.preferences;
  const column = COLUMN[prefs.width] || COLUMN.normal;

  const runtimeRef = useRef(null);
  const inputRef = useRef(null);
  const stripRef = useRef(null);
  const liveRef = useRef(null);
  const contents = useRef(new Map()); // frame n → its content element (for cutting)

  const [book, setBook] = useState(() => loadBook());
  const bookRef = useRef(book);
  const [pieces, setPieces] = useState(() => loadPieces());
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(null);   // { type, text?, moduleId? } being resolved
  const [edge, setEdge] = useState(null);
  const [registered, setRegistered] = useState([]);
  const [fly, setFly] = useState(null);
  const [atLive, setAtLive] = useState(true);     // the live screen fills the view
  const [cut, setCut] = useState(null);           // { n, start, end } while dragging a cut

  const pages = book.pages;
  const busy = pending != null;

  // ── boot ────────────────────────────────────────────────────────────────
  useEffect(() => {
    const cleanup = bootstrapFederation();
    runtimeRef.current = createSurfaceRuntime();
    setResolver(play); // the player is the default resolver on the blank screen
    setRegistered(listModules());
    // Open at the live screen, the history above it.
    requestAnimationFrame(() => liveRef.current?.scrollIntoView({ block: "start" }));
    return () => { setResolver(null); cleanup(); };
  }, []);

  useEffect(() => { saveBook(book); }, [book]);
  useEffect(() => { savePieces(pieces); }, [pieces]);

  // Is the live screen what the user is looking at?
  useEffect(() => {
    const el = liveRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setAtLive(e.intersectionRatio >= 0.55), {
      root: stripRef.current, threshold: [0, 0.55, 1],
    });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const registerContent = useCallback((n, el) => {
    if (el) contents.current.set(n, el); else contents.current.delete(n);
  }, []);

  const focusCaret = useCallback(() => { if (!edge) inputRef.current?.focus({ preventScroll: true }); }, [edge]);
  useEffect(() => { focusCaret(); }, [focusCaret, busy]);

  const scrollToFrame = useCallback((n, smooth = true) => {
    const el = n == null ? liveRef.current : stripRef.current?.querySelector(`[data-frame="${n}"]`);
    el?.scrollIntoView({ block: "start", behavior: smooth && prefs.motion ? "smooth" : "auto" });
  }, [prefs.motion]);

  // The frame whose top is nearest the viewport's top (for keyboard stepping).
  function frameInView() {
    const strip = stripRef.current;
    if (!strip) return null;
    const frames = [...strip.querySelectorAll("[data-frame]")];
    let best = null;
    for (const f of frames) {
      const d = Math.abs(f.getBoundingClientRect().top);
      if (!best || d < best.d) best = { n: Number(f.dataset.frame), d };
    }
    const liveD = Math.abs(liveRef.current?.getBoundingClientRect().top ?? Infinity);
    return best && best.d < liveD ? best.n : null; // null = the live screen
  }

  function step(delta) {
    const at = frameInView();
    const ns = pages.map((p) => p.n);
    const idx = at == null ? ns.length : ns.indexOf(at);
    const next = Math.max(0, Math.min(ns.length, idx + delta));
    scrollToFrame(next === ns.length ? null : ns[next]);
  }

  // ── edges ───────────────────────────────────────────────────────────────
  const dwell = useRef({ edge: null, timer: null });
  useEffect(() => {
    const d = dwell.current;
    const onMove = (e) => {
      if (cut) return;
      const at = edgeAt(e.clientX, e.clientY);
      if (edge && (!at || at === edge)) return;
      if (at === d.edge) return;
      clearTimeout(d.timer);
      d.edge = at;
      if (at) d.timer = setTimeout(() => { setRegistered(listModules()); setEdge(at); }, EDGE_DWELL_MS);
    };
    window.addEventListener("mousemove", onMove);
    return () => { window.removeEventListener("mousemove", onMove); clearTimeout(d.timer); };
  }, [edge, cut]);

  const closeEdge = useCallback(() => { dwell.current.edge = null; setEdge(null); }, []);
  useEffect(() => { if (!edge) focusCaret(); }, [edge, focusCaret]);

  // ── committing a step ───────────────────────────────────────────────────
  async function commit(source, produce) {
    if (busy) return null;
    setPending(source);
    setDraft("");
    const envelope = await produce();
    setPending(null);
    if (!envelope || envelope.kind === "noop") return null;
    if (envelope.kind === "external" && EXTERNAL_ROUTES[envelope.meta]) {
      const w = window.open(EXTERNAL_ROUTES[envelope.meta], "_blank", "noopener");
      if (!w) window.location.href = EXTERNAL_ROUTES[envelope.meta];
    }
    const next = appendPage(bookRef.current, { source, envelope });
    bookRef.current = next;
    setBook(next);
    const n = next.pages.length;
    requestAnimationFrame(() => requestAnimationFrame(() => scrollToFrame(n)));
    return n;
  }

  // What the user put on the screen is what the next step may know: the
  // frames their pieces came from (else the latest frame).
  function contextPage() {
    const last = pieces.length ? pages.find((p) => p.n === pieces[pieces.length - 1].n) : pages[pages.length - 1];
    return pageContext(last || null);
  }

  function commitUtterance(text) {
    const words = text.trim();
    if (!words) return;
    commit({ type: "utterance", text: words }, () => resolve(words, { runtime: runtimeRef.current, page: contextPage() }));
  }

  function pickModule(moduleId) {
    if (busy) return;
    const layoutId = `fly-${Date.now()}`;
    setFly({ layoutId, moduleId });
    requestAnimationFrame(async () => {
      closeEdge();
      const n = await commit({ type: "module", moduleId }, () => openModule(moduleId));
      setFly((f) => (f && f.layoutId === layoutId ? (n ? { ...f, page: n } : null) : f));
      setTimeout(() => setFly((f) => (f && f.layoutId === layoutId ? null : f)), 900);
    });
  }

  function act(instruction, template) {
    if (template) { setDraft(instruction); requestAnimationFrame(() => inputRef.current?.focus()); return; }
    commitUtterance(instruction);
  }

  const actions = useMemo(() => ({
    derive: (label, produce) => commit({ type: "utterance", text: label }, produce),
    write: (text) => commitUtterance(text),
    draft: (text) => { setDraft(text); requestAnimationFrame(() => inputRef.current?.focus()); },
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [busy, pieces, pages]);

  // ── cutting ─────────────────────────────────────────────────────────────
  function onStripPointerDown(e) {
    const { pointer } = settings;
    const wheel = e.button === 1 && pointer.cutWithWheel;
    const alt = e.button === 0 && e.altKey && pointer.cutWithAlt;
    if (!wheel && !alt) return;
    const frameEl = e.target.closest?.("[data-frame]");
    if (!frameEl || e.target.closest("[data-piece]")) return;
    e.preventDefault(); // no middle-click autoscroll, no text selection
    const n = Number(frameEl.dataset.frame);
    const p = { x: e.clientX, y: e.clientY };
    setCut({ n, start: p, end: p });
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
  function onStripPointerMove(e) {
    if (cut) setCut((c) => c && { ...c, end: { x: e.clientX, y: e.clientY } });
  }
  function onStripPointerUp() {
    if (!cut) return;
    const el = contents.current.get(cut.n);
    const piece = el && pieceFromDrag({ n: cut.n, box: el.getBoundingClientRect(), start: cut.start, end: cut.end, placed: pieces.length });
    setCut(null);
    if (piece) setPieces((ps) => [...ps, piece]);
  }

  const movePiece = useCallback((id, at) => setPieces((ps) => ps.map((p) => (p.id === id ? { ...p, at } : p))), []);
  const removePiece = useCallback((id) => setPieces((ps) => ps.filter((p) => p.id !== id)), []);

  // ── keys ────────────────────────────────────────────────────────────────
  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commitUtterance(draft); return; }
    if (e.key === "Escape") { if (draft) { e.preventDefault(); setDraft(""); } return; }
    const empty = draft.length === 0;
    if (e.key === "PageUp" || (empty && e.key === "ArrowUp" && e.altKey)) { e.preventDefault(); step(-1); }
    else if (e.key === "PageDown" || (empty && e.key === "ArrowDown" && e.altKey)) { e.preventDefault(); step(1); }
    else if (empty && e.key === "Home") { e.preventDefault(); if (pages.length) scrollToFrame(pages[0].n); }
    else if (empty && e.key === "End") { e.preventDefault(); scrollToFrame(null); }
  }

  // Keys that land while focus is elsewhere start writing.
  useEffect(() => {
    const onKey = (e) => {
      if (edge || e.defaultPrevented || e.ctrlKey || e.metaKey) return;
      const t = e.target;
      if (t === inputRef.current || (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA"))) return;
      if (e.key.length === 1 && !e.altKey && !busy) {
        e.preventDefault();
        setDraft((d) => d + e.key);
        inputRef.current?.focus({ preventScroll: true });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [edge, busy]);

  function onInput(e) {
    setDraft(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = e.target.scrollHeight + "px";
  }

  // ── drawing ─────────────────────────────────────────────────────────────
  const pendingWords = pending?.type === "utterance" ? pending.text : "";
  const writing = draft.length > 0 || !!pendingWords;
  const caretVisible = (atLive || writing) && !busy;
  const openingModule = pending?.type === "module";
  const flyForPage = (p) =>
    fly && p.source?.type === "module" && p.source.moduleId === fly.moduleId &&
    (fly.page === p.n || (fly.page == null && p.n === pages.length)) ? fly.layoutId : null;

  const cutBox = cut && {
    left: Math.min(cut.start.x, cut.end.x),
    top: Math.min(cut.start.y, cut.end.y),
    width: Math.abs(cut.end.x - cut.start.x),
    height: Math.abs(cut.end.y - cut.start.y),
  };

  return (
    <MotionConfig reducedMotion={prefs.motion ? "never" : "always"}>
      <SurfaceActions.Provider value={actions}>
        <div
          data-surface-root
          className="fixed inset-0 bg-black text-gray-300 font-mono overflow-hidden"
          style={{ fontSize: `${14 * prefs.textScale}px`, lineHeight: 1.625 * prefs.spacing }}
        >
          {/* the strip: every frame, then the live screen */}
          <div
            ref={stripRef}
            data-strip
            className={`absolute inset-0 overflow-y-auto no-scrollbar ${cut ? "cursor-crosshair select-none" : ""}`}
            style={{ overflowAnchor: "auto" }}
            onPointerDown={onStripPointerDown}
            onPointerMove={onStripPointerMove}
            onPointerUp={onStripPointerUp}
            onAuxClick={(e) => { if (e.button === 1) e.preventDefault(); }}
          >
            {pages.map((p) => (
              <Frame key={p.n} page={p} column={column} onAct={act} fly={flyForPage(p)}
                stripRef={stripRef} registerContent={registerContent} />
            ))}

            {openingModule && (
              <section className="relative min-h-screen border-t border-gray-900/80">
                <div className={`mx-auto w-full ${column} px-10 pt-14 md:px-5`}>
                  <ModuleHead id={pending.moduleId} edge={edgeOf(pending.moduleId)} fly={fly?.layoutId} pending />
                </div>
              </section>
            )}

            {/* the live screen: blank, the caret, and the pieces laid on it */}
            <section
              ref={liveRef}
              data-live
              className="relative min-h-screen border-t border-gray-900/80"
              onMouseDown={(e) => { if (e.target === e.currentTarget) { e.preventDefault(); focusCaret(); } }}
            >
              <PieceLayer pieces={pieces} pages={pages} onMove={movePiece} onRemove={removePiece} onAct={act} />
            </section>
          </div>

          {/* the writing layer: one caret — on the live screen's first line, or
              along the foot of whatever frame is in view while writing */}
          <motion.div
            layout
            transition={GLIDE}
            className={
              atLive && !writing
                ? "absolute inset-x-0 top-[38%] pointer-events-none"
                : `absolute inset-x-0 bottom-0 ${writing ? "bg-gradient-to-t from-black via-black/95 to-transparent pt-10" : "pointer-events-none"}`
            }
            style={{ zIndex: 10 }}
          >
            <motion.div layout="position" transition={GLIDE} className={`mx-auto w-full ${column} px-10 md:px-5 ${atLive && !writing ? "" : "pb-8"}`}>
              <textarea
                ref={inputRef}
                value={pendingWords || draft}
                onChange={onInput}
                onKeyDown={onKeyDown}
                readOnly={busy}
                rows={1}
                spellCheck={false}
                autoFocus
                aria-label="write"
                className={`pointer-events-auto w-full bg-transparent border-none outline-none resize-none font-mono ${busy ? "text-gray-500 animate-pulse" : "text-gray-200"}`}
                style={{ caretColor: caretVisible ? "#2a9d8f" : "transparent", fontSize: "1em", lineHeight: "inherit" }}
              />
            </motion.div>
          </motion.div>

          {cutBox && (
            <div className="fixed pointer-events-none border border-teal-400/80 bg-teal-400/10 rounded-sm" style={{ ...cutBox, zIndex: 40 }} />
          )}

          <AnimatePresence>
            {edge && (
              <EdgeDrawer key={edge} edge={edge} registered={registered} onPick={pickModule} onClose={closeEdge} fly={fly} />
            )}
          </AnimatePresence>

          <style jsx global>{`
            .no-scrollbar { scrollbar-width: none; }
            .no-scrollbar::-webkit-scrollbar { display: none; }
            @media print {
              [data-edge-drawer] { display: none !important; }
              [data-surface-root] { position: static !important; overflow: visible !important; }
              [data-surface-root] > div { position: static !important; overflow: visible !important; }
            }
          `}</style>
        </div>
      </SurfaceActions.Provider>
    </MotionConfig>
  );
}
