/* ============================================================================
 * EdgeDrawer — what appears when the pointer reaches a screen edge.
 *
 *   top / bottom   a band across the screen with the edge's modules
 *   right          a panel of the compute / server modules
 *   left           a thin column: search at the top, every module below
 *
 * A drawer lies over the page without touching it. Picking a module closes
 * the drawer and starts a new step (onPick), which opens that module's page.
 * The drawer closes when the pointer leaves it, or on Escape.
 * ========================================================================== */

import { useEffect, useRef, useState } from "react";
import { EDGES, modulesAt, searchModules, edgeOf } from "@/lib/surface/edges";

function firstSentence(s) {
  const t = (s || "").trim();
  const m = t.match(/^(.+?[.:—])\s/);
  return (m ? m[1] : t).replace(/[.:—]$/, "");
}

function ModuleTile({ mod, onPick }) {
  return (
    <button
      type="button"
      onClick={() => onPick(mod.id)}
      className="text-left px-3 py-2 rounded hover:bg-white/5 group min-w-0"
      title={mod.description || mod.id}
    >
      <div className="text-gray-200 text-sm group-hover:text-teal-300">{mod.id}</div>
      <div className="text-gray-600 text-xs truncate">{firstSentence(mod.description)}</div>
    </button>
  );
}

function Band({ edge, modules, onPick, onLeave }) {
  const top = edge === "top";
  return (
    <div
      onMouseLeave={onLeave}
      className={`fixed left-0 right-0 z-30 bg-black/95 border-gray-900 px-10 py-4 md:px-4 ${
        top ? "top-0 border-b drawer-down" : "bottom-0 border-t drawer-up"
      }`}
    >
      <div className="text-gray-600 text-[10px] uppercase tracking-widest mb-2">{EDGES[edge].title}</div>
      {modules.length === 0 ? (
        <div className="text-gray-600 text-xs italic">nothing here yet</div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(14rem,1fr))] gap-1">
          {modules.map((m) => <ModuleTile key={m.id} mod={m} onPick={onPick} />)}
        </div>
      )}
    </div>
  );
}

function RightPanel({ modules, onPick, onLeave }) {
  return (
    <div
      onMouseLeave={onLeave}
      className="fixed top-0 right-0 bottom-0 z-30 w-80 sm:w-64 bg-black/95 border-l border-gray-900 py-6 px-2 overflow-y-auto drawer-left no-scrollbar"
    >
      <div className="text-gray-600 text-[10px] uppercase tracking-widest mb-3 px-3">{EDGES.right.title}</div>
      <div className="flex flex-col gap-1">
        {modules.map((m) => <ModuleTile key={m.id} mod={m} onPick={onPick} />)}
      </div>
    </div>
  );
}

const EDGE_MARK = { top: "↑", right: "→", bottom: "↓" };

function LeftColumn({ registered, onPick, onLeave }) {
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
    <div
      onMouseLeave={onLeave}
      className="fixed top-0 left-0 bottom-0 z-30 w-56 bg-black/95 border-r border-gray-900 flex flex-col drawer-right"
    >
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
          return (
            <button
              key={m.id}
              type="button"
              onClick={() => onPick(m.id)}
              onMouseEnter={() => setActive(i)}
              title={m.description || m.id}
              className={`w-full text-left px-4 py-1 text-xs font-mono flex justify-between ${
                i === active ? "text-teal-300 bg-white/5" : "text-gray-400"
              }`}
            >
              <span className="truncate">{m.id}</span>
              {edge && <span className="text-gray-700 ml-2">{EDGE_MARK[edge]}</span>}
            </button>
          );
        })}
        {hits.length === 0 && <div className="px-4 text-gray-700 text-xs italic">no match</div>}
      </div>
    </div>
  );
}

export default function EdgeDrawer({ edge, registered, onPick, onClose }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!edge) return null;
  if (edge === "left") return <LeftColumn registered={registered} onPick={onPick} onLeave={onClose} />;
  if (edge === "right") return <RightPanel modules={modulesAt("right", registered)} onPick={onPick} onLeave={onClose} />;
  return <Band edge={edge} modules={modulesAt(edge, registered)} onPick={onPick} onLeave={onClose} />;
}
