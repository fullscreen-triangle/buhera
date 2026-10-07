/* ============================================================================
 * Surface settings — the user's standing choices, in one store.
 *
 * Everything the right and bottom edges configure lives here: how the screen
 * reads (text size, spacing, code visibility, the cut gesture), which model
 * speaks for the user, where retrieval looks, which project is active. Plans
 * (scripts the player produced or a person wrote) and reports (the account a
 * completed run leaves) are kept here too — they are the user's, not a page's.
 *
 * One object, persisted to localStorage, observable: components subscribe with
 * useSettings() and re-render on change; modules read with getSettings(). No
 * content from pages is ever stored here (pages are the book's; see book.js).
 * ========================================================================== */

import { useSyncExternalStore } from "react";

export const SETTINGS_KEY = "buhera.surface.settings";
const MAX_PLANS = 60;
const MAX_REPORTS = 40;

export const DEFAULTS = Object.freeze({
  preferences: {
    textScale: 1,        // multiplies the surface's base font size
    spacing: 1,          // multiplies line height and block spacing
    width: "normal",     // "narrow" | "normal" | "wide" column
    motion: true,        // page turns, drawer slides, flying marks
  },
  code: {
    showScript: true,    // show the vaHera a step ran, above its result
    showCodeBlocks: true,// show code inside module results (interceptor, generated DSL)
    showTrace: false,    // show the player's retrieval + generation trace
  },
  pointer: {
    cutWithWheel: true,  // press the scroll wheel and drag to cut
    cutWithAlt: true,    // Alt + drag does the same (trackpads)
  },
  model: {
    providers: [],       // [] = every configured provider drafts; else only these
    temperature: 0.3,
    instructions: "",    // the user's standing notes to their model ("prefer SI units")
  },
  rag: {
    enabled: true,
    folders: [],         // local folders the retrieval reads (served by /api/player)
    extensions: [".md", ".txt", ".tex"],
  },
  lattice: {
    repos: [],           // repositories on this machine whose tasks can go to AppHub
  },
  project: "default",    // the active project: names the receiver and scopes plans
  plans: [],             // [{ id, at, project, source, script, by }]
  reports: [],           // [{ id, at, project, source, script, summary, graph, retrieval }]
});

let _state = load();
const _listeners = new Set();

function load() {
  try {
    if (typeof window === "undefined") return structuredClone(DEFAULTS);
    const raw = window.localStorage.getItem(SETTINGS_KEY);
    return merge(DEFAULTS, raw ? JSON.parse(raw) : {});
  } catch {
    return structuredClone(DEFAULTS);
  }
}

// Deep-merge stored values over the defaults, so a new setting added in code
// appears with its default for a user whose stored object predates it.
function merge(base, over) {
  if (Array.isArray(base)) return Array.isArray(over) ? over : structuredClone(base);
  if (base && typeof base === "object") {
    const out = {};
    for (const k of Object.keys(base)) out[k] = merge(base[k], over?.[k]);
    return out;
  }
  return over === undefined ? base : over;
}

function persist() {
  try {
    if (typeof window !== "undefined") window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(_state));
  } catch { /* storage full or blocked: settings still apply for this session */ }
}

export function getSettings() {
  return _state;
}

/** Replace a section with `patch` merged into it (e.g. update("preferences", { textScale: 1.1 })). */
export function update(section, patch) {
  const cur = _state[section];
  const next = cur && typeof cur === "object" && !Array.isArray(cur) ? { ...cur, ...patch } : patch;
  _state = { ..._state, [section]: next };
  persist();
  for (const l of _listeners) l();
}

export function subscribe(listener) {
  _listeners.add(listener);
  return () => _listeners.delete(listener);
}

/** React: the whole settings object, re-rendering on change. */
export function useSettings() {
  return useSyncExternalStore(subscribe, getSettings, getSettings);
}

const newId = (p) => `${p}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;

/** Keep a script as a plan. Returns the plan. */
export function addPlan({ source, script, by }) {
  const plan = { id: newId("plan"), at: Date.now(), project: _state.project, source, script, by };
  update("plans", [plan, ..._state.plans].slice(0, MAX_PLANS));
  return plan;
}

/** Keep the account of a completed run. Returns the report. */
export function addReport(report) {
  const r = { id: newId("report"), at: Date.now(), project: _state.project, ...report };
  update("reports", [r, ..._state.reports].slice(0, MAX_REPORTS));
  return r;
}

/** Reset one section to its default. */
export function resetSection(section) {
  update(section, structuredClone(DEFAULTS[section]));
}
