/* ============================================================================
 * EdgeDrawer — what appears when the pointer reaches a screen edge.
 *
 *   top / bottom   a band across the screen with the edge's modules
 *   right          a panel of the compute / server modules
 *   left           a thin column: search at the top, every module below
 *
 * A drawer lies over the page without touching it. It slides in from its
 * edge and back out when the pointer leaves or on Escape (the surface wraps
 * it in AnimatePresence). Picking a module starts a new step that opens the
 * module's page; the picked module's mark carries a shared layout id
 * (`fly`), so it travels from the drawer to the head of the page it opens.
 * ========================================================================== */

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { EDGES, modulesAt, searchModules, edgeOf } from "@/lib/surface/edges";
import ModuleIcon from "@/components/surface/ModuleIcon";

const SLIDE = {
  top: { y: "-100%" },
  bottom: { y: "100%" },
  right: { x: "100%" },
  left: { x: "-100%" },
};
const SPRING = { type: "spring", stiffness: 420, damping: 38, mass: 0.8 };

function firstSentence(s) {
  const t = (s || "").trim();
  const m = t.match(/^(.+?[.:—])\s/);
  return (m ? m[1] : t).replace(/[.:—]$/, "");
}

// The module's mark, carrying the shared layout id while it takes off — only
// until the page it flies to exists (`fly.page` set), never after.
function Mark({ id, fly, size = 16 }) {
  const flying = fly && fly.moduleId === id && fly.page == null;
  return (
    <motion.span layoutId={flying ? fly.layoutId : undefined} className="inline-flex">
      <ModuleIcon id={id} size={size} />
    </motion.span>
  );
}

function ModuleTile({ mod, onPick, fly }) {
  return (
    <button
      type="button"
      onClick={() => onPick(mod.id)}
      className="text-left px-3 py-2 rounded hover:bg-white/5 group min-w-0 flex items-start gap-3 transition-colors"
      title={mod.description || mod.id}
    >
      <span className="mt-0.5 text-gray-500 group-hover:text-teal-300 transition-colors">
        <Mark id={mod.id} fly={fly} />
      </span>
      <span className="min-w-0">
        <span className="block text-gray-200 text-sm group-hover:text-teal-300 transition-colors">{mod.id}</span>
        <span className="block text-gray-600 text-xs truncate">{firstSentence(mod.description)}</span>
      </span>
    </button>
  );
}

function Shell({ edge, className, onLeave, children }) {
  return (
    <motion.div
      initial={SLIDE[edge]}
      animate={{ x: 0, y: 0 }}
      exit={SLIDE[edge]}
      transition={SPRING}
      onMouseLeave={onLeave}
      data-edge-drawer=""
      className={`fixed z-30 bg-black/95 backdrop-blur-sm ${className}`}
    >
      {children}
    </motion.div>
  );
}

function Band({ edge, modules, onPick, onLeave, fly }) {
  const top = edge === "top";
  return (
    <Shell edge={edge} onLeave={onLeave}
      className={`left-0 right-0 border-gray-900 px-10 py-4 md:px-4 ${top ? "top-0 border-b" : "bottom-0 border-t"}`}>
      <div className="text-gray-600 text-[10px] uppercase tracking-widest mb-2">{EDGES[edge].title}</div>
      {modules.length === 0 ? (
        <div className="text-gray-600 text-xs italic">nothing here yet</div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] gap-1">
          {modules.map((m) => <ModuleTile key={m.id} mod={m} onPick={onPick} fly={fly} />)}
        </div>
      )}
    </Shell>
  );
}

function RightPanel({ modules, onPick, onLeave, fly }) {
  return (
    <Shell edge="right" onLeave={onLeave}
      className="top-0 right-0 bottom-0 w-80 sm:w-64 border-l border-gray-900 py-6 px-2 overflow-y-auto no-scrollbar">
      <div className="text-gray-600 text-[10px] uppercase tracking-widest mb-3 px-3">{EDGES.right.title}</div>
      <div className="flex flex-col gap-1">
        {modules.map((m) => <ModuleTile key={m.id} mod={m} onPick={onPick} fly={fly} />)}
      </div>
    </Shell>
  );
}

const EDGE_MARK = { top: "↑", right: "→", bottom: "↓" };

function LeftColumn({ registered, onPick, onLeave, fly }) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef(null);
  const hits = searchModules(registered, query);

  useEffect(() => { inputRef.current?.focus(); }, []);
  useEffect(() => { setActive(0); }, [query]);

  function onKeyDown(e) {
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(hits.length - 1, a + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
    else if (e.key === "Enter" && hits[active]) { e.preventDefault(); onPick(hits[active].id); }
  }

  return (
    <Shell edge="left" onLeave={onLeave} className="top-0 left-0 bottom-0 w-56 border-r border-gray-900 flex flex-col">
      <input
        ref={inputRef}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={onKeyDown}
        spellCheck={false}
        aria-label="search modules"
        className="m-3 mb-2 px-2 py-1 bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-gray-200 text-xs font-mono"
        style={{ caretColor: "#2a9d8f" }}
      />
      <div className="flex-1 overflow-y-auto no-scrollbar pb-4">
        {hits.map((m, i) => {
          const edge = edgeOf(m.id);
          const on = i === active;
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => onPick(m.id)}
              onMouseEnter={() => setActive(i)}
              title={m.description || m.id}
              className={`w-full text-left px-4 py-1 text-xs font-mono flex items-center gap-2 transition-colors ${
                on ? "text-teal-300 bg-white/5" : "text-gray-400"
              }`}
            >
              <span className={on ? "text-teal-300" : "text-gray-600"}><Mark id={m.id} fly={fly} size={14} /></span>
              <span className="truncate flex-1">{m.id}</span>
              {edge && <span className="text-gray-700 ml-2">{EDGE_MARK[edge]}</span>}
            </button>
          );
        })}
        {hits.length === 0 && <div className="px-4 text-gray-700 text-xs italic">no match</div>}
      </div>
    </Shell>
  );
}

export default function EdgeDrawer({ edge, registered, onPick, onClose, fly }) {
  // Leaving closes a drawer (the Shell’s mouseleave); so do Escape and a
  // pointer pressed anywhere outside it.
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    const onDown = (e) => { if (!e.target.closest?.("[data-edge-drawer]")) onClose(); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown, true);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown, true);
    };
  }, [onClose]);

  if (edge === "left") return <LeftColumn registered={registered} onPick={onPick} onLeave={onClose} fly={fly} />;
  if (edge === "right") return <RightPanel modules={modulesAt("right", registered)} onPick={onPick} onLeave={onClose} fly={fly} />;
  return <Band edge={edge} modules={modulesAt(edge, registered)} onPick={onPick} onLeave={onClose} fly={fly} />;
}
