/* ============================================================================
 * Pieces — parts cut out of frames, placed on the live screen.
 *
 * There are no application windows. Instead, while scrolling through the
 * frames, the user presses the scroll wheel (or Alt) and drags over any part
 * of any frame; that part lifts onto the live screen as a piece. Pieces from
 * many frames sit side by side on one screen — several "applications" at
 * once, with no window manager and no split screen.
 *
 * A piece is LIVE: it is not a picture but a window onto its frame's content,
 * re-rendered at the width the frame had when it was cut and offset so exactly
 * the cut rectangle shows. A chart in a piece still brushes; an action in a
 * piece still starts a step. The frame itself is untouched (pages are inert).
 *
 * A piece: { id, n, rect: {x, y, w, h}, width, at: {x, y} } — `n` the frame,
 * `rect` in that frame's content coordinates, `width` the content width at
 * cutting time, `at` where it sits on the live screen. Persisted locally.
 * ========================================================================== */

import { useEffect, useState } from "react";
import { AnimatePresence, motion, useDragControls } from "framer-motion";
import { X } from "lucide-react";
import PageView from "@/components/surface/PageView";

// Geometry and persistence live in lib/surface/pieces.js (pure, tested).
export { loadPieces, savePieces } from "@/lib/surface/pieces";

function Piece({ piece, page, onMove, onRemove, onAct }) {
  const controls = useDragControls();
  const [hover, setHover] = useState(false);
  return (
    <motion.div
      drag
      dragListener={false}
      dragControls={controls}
      dragMomentum={false}
      onDragEnd={(_, info) => onMove(piece.id, { x: piece.at.x + info.offset.x, y: piece.at.y + info.offset.y })}
      initial={{ opacity: 0, scale: 0.96 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ type: "spring", stiffness: 380, damping: 34 }}
      onHoverStart={() => setHover(true)}
      onHoverEnd={() => setHover(false)}
      className="absolute rounded border border-gray-800 hover:border-gray-600 bg-black overflow-hidden shadow-[0_0_0_1px_rgba(0,0,0,0.6)]"
      style={{ left: piece.at.x, top: piece.at.y, width: piece.rect.w, height: piece.rect.h + 18, zIndex: 5 }}
      data-piece={piece.id}
    >
      {/* the handle: move the piece by it; the rest stays live */}
      <div
        onPointerDown={(e) => { e.stopPropagation(); controls.start(e); }}
        className="h-[18px] flex items-center justify-between px-1.5 cursor-grab active:cursor-grabbing bg-white/[0.03] border-b border-gray-900 select-none"
      >
        <span className="text-[9px] text-gray-600">{page ? `frame ${piece.n}` : `frame ${piece.n} — gone from the book`}</span>
        <button type="button" onPointerDown={(e) => e.stopPropagation()} onClick={() => onRemove(piece.id)}
          className={`text-gray-600 hover:text-rose-300 transition-opacity ${hover ? "opacity-100" : "opacity-0"}`} aria-label="remove piece">
          <X size={11} />
        </button>
      </div>
      <div style={{ width: piece.rect.w, height: piece.rect.h, overflow: "hidden", position: "relative" }}>
        {page && (
          <div style={{ width: piece.width, position: "absolute", left: -piece.rect.x, top: -piece.rect.y }}>
            <FrameContent page={page} onAct={onAct} />
          </div>
        )}
      </div>
    </motion.div>
  );
}

/**
 * A frame's content box. Shared by frames and pieces so a piece's offsets
 * (measured on the frame's content box) land on exactly the same layout.
 */
export function FrameContent({ page, onAct, fly, contentRef }) {
  return (
    <div ref={contentRef} data-frame-content className="px-10 pt-14 pb-16 md:px-5">
      <PageView page={page} onAct={onAct} fly={fly} />
    </div>
  );
}

export function PieceLayer({ pieces, pages, onMove, onRemove, onAct }) {
  const byN = new Map(pages.map((p) => [p.n, p]));
  // Re-read the window size so pieces cut on a wider screen stay reachable.
  const [, setTick] = useState(0);
  useEffect(() => {
    const onResize = () => setTick((t) => t + 1);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return (
    <AnimatePresence>
      {pieces.map((p) => (
        <Piece key={p.id} piece={p} page={byN.get(p.n)} onMove={onMove} onRemove={onRemove} onAct={onAct} />
      ))}
    </AnimatePresence>
  );
}
