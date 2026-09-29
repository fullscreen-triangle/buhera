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
 *   P   blank page, writing                  — the words, where a first line sits
 *   A   viewing a page                       — the page; caret hidden
 *   AP  viewing a page, writing              — the page, words along its foot
 * Writing while on an older page forks to the end: the new page is appended
 * after the latest one; nothing is rewritten.
 *
 * Edges: the pointer at the top, right, bottom or left edge opens that edge's
 * drawer (components/surface/EdgeDrawer.js; the filing is lib/surface/edges.js).
 * Picking a module opens its page as a new step; the module's mark flies from
 * the drawer to the head of that page (a shared layout id, `fly`).
 *
 * Motion: pages turn — the old page leaves the way the new one arrives from;
 * the caret glides between the blank page's first line and a page's foot;
 * drawers slide from their edge. All of it is layout motion, never content:
 * nothing appears on a page that was not in its snapshot.
 *
 * Flipping: PageUp / PageDown, Alt+← / Alt+→, or ← / → and Home / End when
 * nothing is written; a horizontal swipe on a trackpad.
 *
 * What the next step may know: kernel memory, and the page being viewed when
 * the step began — passed to the resolver (lib/surface/resolve.js), which is
 * where the intent layer plugs in.
 * ========================================================================== */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { bootstrapFederation } from "@/lib/runtime/bootstrap";
import { listModules } from "@/lib/modules/registry";
import { appendPage, loadBook, pageContext, saveBook } from "@/lib/surface/book";
import { edgeOf } from "@/lib/surface/edges";
import { createSurfaceRuntime, openModule, resolve } from "@/lib/surface/resolve";
import PageView, { ModuleHead } from "@/components/surface/PageView";
import EdgeDrawer from "@/components/surface/EdgeDrawer";
import { SurfaceActions } from "@/components/surface/actions";

// Pointer must sit within EDGE_PX of an edge for EDGE_DWELL_MS before the
// drawer opens, so crossing an edge on the way somewhere does not flash one.
const EDGE_PX = 3;
const EDGE_DWELL_MS = 140;

// Routes a line-router "external" envelope navigates to.
const EXTERNAL_ROUTES = { tutorials: "/tutorials", experiment: "/protein-modelling" };

const GLIDE = { type: "spring", stiffness: 320, damping: 36, mass: 0.9 };

// A page turn: dir 1 arrives from the right and leaves to the left; -1 the
// reverse; 0 (a module opening, carried by its flying mark) just fades.
const TURN = {
  enter: (dir) => ({ opacity: 0, x: dir * 56 }),
  center: { opacity: 1, x: 0, transition: { x: GLIDE, opacity: { duration: 0.22 } } },
  exit: (dir) => ({ opacity: 0, x: dir * -56, transition: { x: GLIDE, opacity: { duration: 0.16 } } }),
};

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
  const [dir, setDir] = useState(1);              // direction of the last turn
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(null);   // { type, text? , moduleId? } being resolved
  const [edge, setEdge] = useState(null);
  const [registered, setRegistered] = useState([]);
  const [showFolio, setShowFolio] = useState(false);
  const [fly, setFly] = useState(null);           // { layoutId, moduleId, page? } — a mark in flight

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

  // Keys that land while focus is elsewhere (after selecting text on a page):
  // printable keys start writing, flip keys flip. Drawer open ⇒ hands off.
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
    const d = dwell.current; // one object for the component's lifetime
    const onMove = (e) => {
      const at = edgeAt(e.clientX, e.clientY);
      // An open drawer closes on its own (pointer leaves, click outside,
      // Escape); reaching a DIFFERENT edge while it is open switches to it.
      if (edge && (!at || at === edge)) return;
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
      clearTimeout(d.timer);
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
  // Returns the new page's number, or null if nothing was committed.
  async function commit(source, produce) {
    if (busy) return null;
    const latest = bookRef.current.pages.length;
    const from = current && current.n !== latest ? current.n : null;
    setPending(source);
    setDraft("");
    const envelope = await produce();
    setPending(null);
    if (!envelope || envelope.kind === "noop") return null;
    if (envelope.kind === "external" && EXTERNAL_ROUTES[envelope.meta]) {
      const w = window.open(EXTERNAL_ROUTES[envelope.meta], "_blank", "noopener");
      if (!w) window.location.href = EXTERNAL_ROUTES[envelope.meta];
    }
    const next = appendPage(bookRef.current, { source, envelope, from });
    bookRef.current = next;
    setBook(next);
    setDir(source.type === "module" ? 0 : 1);
    setView(next.pages.length - 1);
    return next.pages.length;
  }

  function commitUtterance(text) {
    const words = text.trim();
    if (!words) return;
    const page = pageContext(current);
    commit({ type: "utterance", text: words }, () =>
      resolve(words, { runtime: runtimeRef.current, page })
    );
  }

  // Picking a module: first mark the picked tile as in flight (so it carries
  // the shared layout id), then — next frame — close the drawer and open the
  // module. The pending head and then the page head take the same id, so the
  // mark travels drawer → head instead of vanishing and reappearing.
  function pickModule(moduleId) {
    if (busy) return;
    const layoutId = `fly-${Date.now()}`;
    setFly({ layoutId, moduleId });
    requestAnimationFrame(async () => {
      closeEdge();
      const n = await commit({ type: "module", moduleId }, () => openModule(moduleId));
      setFly((f) => (f && f.layoutId === layoutId ? (n ? { ...f, page: n } : null) : f));
      // Once landed, retire the id. A mark left carrying it would share it with
      // the next drawer's row for the same module, and the page head would
      // fly back into the drawer.
      setTimeout(() => setFly((f) => (f && f.layoutId === layoutId ? null : f)), 900);
    });
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

  // What a page's contents may start: a new step with given words.
  const actions = useMemo(() => ({
    derive: (label, produce) => commit({ type: "utterance", text: label }, produce),
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [current, busy]);

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
  const pendingWords = pending?.type === "utterance" ? pending.text : "";
  const writing = draft.length > 0 || !!pendingWords;
  // The caret shows on a blank page always, and on a page only once writing.
  const caretVisible = (onBlank || writing) && !busy;
  const openingModule = pending?.type === "module";

  // One key per thing that can occupy the page layer.
  const layerKey = openingModule ? `opening-${fly?.layoutId}` : onBlank ? `blank-${pages.length}` : `page-${current.n}`;
  // The page a mark flew to keeps its layout id — including the first render
  // after it lands, before `fly.page` is recorded (then it is the latest page).
  const flyForPage =
    current && fly && current.source?.type === "module" && current.source.moduleId === fly.moduleId &&
    (fly.page === current.n || (fly.page == null && current.n === pages.length))
      ? fly.layoutId
      : null;

  const folio = useMemo(
    () => (onBlank ? `${pages.length + 1}` : `${view + 1} / ${pages.length}`),
    [onBlank, pages.length, view]
  );

  return (
    <SurfaceActions.Provider value={actions}>
      <div
        className="fixed inset-0 bg-black text-gray-300 font-mono text-sm leading-relaxed overflow-hidden"
        onMouseDown={(e) => {
          // Clicking empty surface returns to writing; clicks on page content
          // (selecting text, actions, charts) are left alone.
          if (e.target === e.currentTarget) { e.preventDefault(); focusCaret(); }
        }}
      >
        {/* the page layer — one page at a time, turning */}
        <AnimatePresence initial={false} custom={dir}>
          <motion.div
            key={layerKey}
            custom={dir}
            variants={TURN}
            initial="enter"
            animate="center"
            exit="exit"
            className="absolute inset-0 overflow-y-auto no-scrollbar"
            onMouseDown={(e) => { if (e.target === e.currentTarget) { e.preventDefault(); focusCaret(); } }}
          >
            {openingModule ? (
              <div className="mx-auto w-full max-w-4xl px-10 pt-16 md:px-5">
                <ModuleHead id={pending.moduleId} edge={edgeOf(pending.moduleId)} fly={fly?.layoutId} pending />
              </div>
            ) : current ? (
              <div className={`mx-auto w-full max-w-4xl px-10 pt-16 md:px-5 transition-opacity duration-300 ${writing ? "pb-40 opacity-60" : "pb-24"}`}>
                <PageView page={current} onAct={act} fly={flyForPage} />
              </div>
            ) : null}
          </motion.div>
        </AnimatePresence>

        {/* the writing layer — one caret, gliding between the blank page's
            first line and a page's foot */}
        <motion.div
          layout
          transition={GLIDE}
          className={
            onBlank
              ? "absolute inset-x-0 top-[38%]"
              : `absolute inset-x-0 bottom-0 ${writing ? "bg-gradient-to-t from-black via-black/95 to-transparent pt-10" : "pointer-events-none"}`
          }
        >
          <motion.div layout="position" transition={GLIDE}
            className={`mx-auto w-full px-10 md:px-5 ${onBlank ? "max-w-3xl" : "max-w-4xl pb-8"}`}>
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
              className={`w-full bg-transparent border-none outline-none resize-none font-mono text-sm leading-relaxed ${
                busy ? "text-gray-500 animate-pulse" : "text-gray-200"
              }`}
              style={{ caretColor: caretVisible ? "#2a9d8f" : "transparent" }}
            />
          </motion.div>
        </motion.div>

        <div
          className={`fixed bottom-3 right-4 text-[10px] text-gray-700 pointer-events-none transition-opacity duration-500 ${
            showFolio ? "opacity-100" : "opacity-0"
          }`}
          aria-hidden="true"
        >
          {folio}
        </div>

        <AnimatePresence>
          {edge && (
            <EdgeDrawer key={edge} edge={edge} registered={registered} onPick={pickModule} onClose={closeEdge} fly={fly} />
          )}
        </AnimatePresence>

        <style jsx global>{`
          .no-scrollbar { scrollbar-width: none; }
          .no-scrollbar::-webkit-scrollbar { display: none; }
        `}</style>
      </div>
    </SurfaceActions.Provider>
  );
}
