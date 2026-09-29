/* ============================================================================
 * Pieces — the pure half: geometry and persistence.
 *
 * A piece is a part of a frame, cut out by the user and laid on the live
 * screen (components/surface/Pieces.js draws it live). Here only the data:
 *
 *   { id, n, rect: {x, y, w, h}, width, at: {x, y} }
 *     n      the frame it was cut from
 *     rect   the cut, in that frame's content-box coordinates
 *     width  the frame's content width when cut (the piece re-renders at it)
 *     at     where it sits on the live screen
 * ========================================================================== */

export const PIECES_KEY = "buhera.surface.pieces";
export const MIN_SIDE = 24;

export function loadPieces(storage = safeStorage()) {
  try {
    const raw = storage && storage.getItem(PIECES_KEY);
    const v = raw ? JSON.parse(raw) : [];
    return Array.isArray(v) ? v.filter(isPiece) : [];
  } catch {
    return [];
  }
}

export function savePieces(pieces, storage = safeStorage()) {
  try { storage && storage.setItem(PIECES_KEY, JSON.stringify(pieces)); } catch { /* best effort */ }
}

function isPiece(p) {
  return p && typeof p.id === "string" && Number.isFinite(p.n) && p.rect && p.at && Number.isFinite(p.width);
}

/**
 * Turn a drag (viewport coordinates) over a frame's content box into a piece,
 * clipped to the box, or null if too small to mean anything.
 * @param {{ n, box: {left, top, right, bottom, width}, start, end, placed }} a
 */
export function pieceFromDrag({ n, box, start, end, placed = 0 }) {
  const x0 = Math.max(box.left, Math.min(start.x, end.x));
  const y0 = Math.max(box.top, Math.min(start.y, end.y));
  const x1 = Math.min(box.right, Math.max(start.x, end.x));
  const y1 = Math.min(box.bottom, Math.max(start.y, end.y));
  const w = x1 - x0;
  const h = y1 - y0;
  if (w < MIN_SIDE || h < MIN_SIDE) return null;
  const k = placed % 8; // cascade new pieces so none lands exactly on another
  return {
    id: `piece-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    n,
    rect: { x: x0 - box.left, y: y0 - box.top, w, h },
    width: box.width,
    at: { x: 48 + k * 28, y: 64 + k * 28 },
  };
}

function safeStorage() {
  try { return typeof window !== "undefined" ? window.localStorage : null; } catch { return null; }
}
