/* ============================================================================
 * The page book — the surface's only history.
 *
 * Every step the user takes on the blank surface commits one page. A page is
 * a snapshot: the words that started the step and the result that came back,
 * serialised at commit time and frozen. It is semantically inert — it never
 * re-runs, holds no live references, and carries nothing that is not drawn
 * on it. Going back is flipping to an earlier page, not re-executing it.
 *
 * The book is append-only. Starting a step while viewing an older page forks
 * to the end: the new page is appended after the latest one, and the page the
 * step was started from is recorded as `from` (so the fork is legible), but no
 * page is ever rewritten or truncated.
 *
 * Pure data — no React, no DOM beyond best-effort localStorage persistence —
 * so the Node test runner can exercise it directly.
 *
 * Page shape:
 *   {
 *     n:        1-based page number (position in the book)
 *     at:       commit time, epoch ms
 *     from:     page number the step was started from, or null (blank page)
 *     source:   { type: "utterance", text }
 *             | { type: "module", moduleId, edge }   (picked from an edge)
 *     envelope: the runner envelope (see lib/runtime/run-input.js), frozen
 *   }
 * ========================================================================== */

export const STORAGE_KEY = "buhera.surface.book";

// Cap on what we try to persist. A page carrying a huge payload (a full
// mass-spec record set, say) should not evict the rest of localStorage; past
// the cap we drop the oldest pages from the *persisted* copy only.
const PERSIST_BUDGET_BYTES = 2_000_000;

/**
 * Serialise-and-freeze. The JSON round trip is the definition of "inert":
 * functions, class instances, typed-array identity and live references do
 * not survive, which is exactly what may not live on a page. Values that
 * cannot be serialised at all become a text envelope saying so, rather than
 * silently vanishing.
 */
export function snapshot(value) {
  let copy;
  try {
    copy = JSON.parse(JSON.stringify(value ?? null));
  } catch (err) {
    copy = { kind: "text", lines: [`(this result could not be snapshotted: ${err.message || err})`] };
  }
  return deepFreeze(copy);
}

function deepFreeze(v) {
  if (v && typeof v === "object" && !Object.isFrozen(v)) {
    Object.freeze(v);
    for (const k of Object.keys(v)) deepFreeze(v[k]);
  }
  return v;
}

/** An empty book. */
export function emptyBook() {
  return Object.freeze({ pages: Object.freeze([]) });
}

/**
 * Append one committed step. Returns a new book; the old one is untouched.
 *
 * @param {{pages: object[]}} book
 * @param {{ source: object, envelope: object, from?: number|null, at?: number }} step
 */
export function appendPage(book, { source, envelope, from = null, at = Date.now() }) {
  const page = snapshot({
    n: book.pages.length + 1,
    at,
    from: from ?? null,
    source,
    envelope,
  });
  return Object.freeze({ pages: Object.freeze([...book.pages, page]) });
}

/**
 * What the next step is allowed to know about the page it was started from:
 * the page's visible content, and nothing else. The resolver receives this
 * alongside kernel memory. Returns null for the blank page.
 */
export function pageContext(page) {
  if (!page) return null;
  return { n: page.n, source: page.source, envelope: page.envelope };
}

// --------------------------------------------------------------------------
// Persistence (best effort; the book is fully usable without it).
// --------------------------------------------------------------------------

/** Load the persisted book, or an empty one. Never throws. */
export function loadBook(storage = safeStorage()) {
  if (!storage) return emptyBook();
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return emptyBook();
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.pages)) return emptyBook();
    // A trimmed persisted copy starts mid-book: renumber it, and drop `from`
    // links, which would otherwise point at pages that are no longer here.
    const trimmed = parsed.pages.length > 0 && parsed.pages[0].n !== 1;
    const pages = trimmed
      ? parsed.pages.map((p, i) => ({ ...p, n: i + 1, from: null }))
      : parsed.pages;
    return Object.freeze({ pages: Object.freeze(pages.map((p) => deepFreeze(p))) });
  } catch {
    return emptyBook();
  }
}

/** Persist the book, trimming oldest pages to fit the budget. Never throws. */
export function saveBook(book, storage = safeStorage()) {
  if (!storage) return false;
  let pages = book.pages;
  while (pages.length) {
    const raw = JSON.stringify({ pages });
    if (raw.length <= PERSIST_BUDGET_BYTES) {
      try {
        storage.setItem(STORAGE_KEY, raw);
        return true;
      } catch {
        // quota exceeded — fall through and trim
      }
    }
    pages = pages.slice(1);
  }
  try { storage.removeItem(STORAGE_KEY); } catch { /* noop */ }
  return false;
}

function safeStorage() {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}
