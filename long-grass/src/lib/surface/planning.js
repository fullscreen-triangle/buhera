/* ============================================================================
 * Planning — the items you plan: experiments and tasks, each with steps, the
 * things you found that bear on it, and the AppHub jobs that run it.
 *
 *   item  { id, project, kind: "experiment" | "task", title, status, due,
 *           created, updated, steps: [{ id, text, done }],
 *           refs: [{ id, source, cite, title, snippet, verdict, query, at,
 *                    note?, mermaid? }],
 *           jobs: [{ repo, unit, remote, at }], notes }
 *   status  idea → planned → running → done  (or dropped)
 *
 * A ref is something found by `find` (a mail, a passage in your files, a web
 * answer) kept on the item with where it came from (`cite`) and, for a
 * search that gives one, its coverage verdict — so a plan records not just
 * what you read but how sure the search was that it was about the thing.
 * A ref with a `note` is your own words about a passage (cite: url#anchor);
 * one with `mermaid` is a diagram. The Markdown export gives each its section.
 *
 * Kept in this browser (localStorage), like the surface's settings; scoped to
 * the active project. One store, observable with usePlanning().
 * ========================================================================== */

import { useSyncExternalStore } from "react";
import { getSettings } from "@/lib/surface/settings";

export const PLANNING_KEY = "buhera.surface.planning";
export const STATUSES = ["idea", "planned", "running", "done", "dropped"];

// An experiment starts with the steps one usually has; every one can be
// edited or removed. A task starts empty.
export const EXPERIMENT_STEPS = [
  "state the question",
  "find what is already known — mail, notes, papers",
  "materials and method",
  "book time: instrument, AppHub session",
  "run",
  "analyse",
  "write the report",
];

const newId = (p) => `${p}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;

let _items = load();
const _listeners = new Set();

function load() {
  try {
    if (typeof window === "undefined") return [];
    const raw = window.localStorage.getItem(PLANNING_KEY);
    const v = raw ? JSON.parse(raw) : [];
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function commit(next) {
  _items = next;
  try {
    if (typeof window !== "undefined") window.localStorage.setItem(PLANNING_KEY, JSON.stringify(_items));
  } catch { /* storage full or blocked: the plan still holds for this session */ }
  for (const l of _listeners) l();
}

export function getItems() {
  return _items;
}

export function subscribe(listener) {
  _listeners.add(listener);
  return () => _listeners.delete(listener);
}

export function usePlanning() {
  return useSyncExternalStore(subscribe, getItems, getItems);
}

const project = () => getSettings().project || "default";

/** The active project's items, newest first, open ones before closed. */
export function itemsInProject(items = _items, p = project()) {
  const closed = (i) => i.status === "done" || i.status === "dropped";
  return items.filter((i) => i.project === p).sort((a, b) => Number(closed(a)) - Number(closed(b)) || b.updated - a.updated);
}

export function addItem({ title, kind = "task", due = null, steps } = {}) {
  const now = Date.now();
  const item = {
    id: newId("item"),
    project: project(),
    kind: kind === "experiment" ? "experiment" : "task",
    title: String(title || "").trim() || (kind === "experiment" ? "untitled experiment" : "untitled task"),
    status: "idea",
    due,
    created: now,
    updated: now,
    steps: (steps || (kind === "experiment" ? EXPERIMENT_STEPS : [])).map((text) => ({ id: newId("step"), text, done: false })),
    refs: [],
    jobs: [],
    notes: "",
  };
  commit([item, ..._items]);
  return item;
}

function change(id, fn) {
  let found = null;
  commit(_items.map((i) => {
    if (i.id !== id) return i;
    found = { ...fn(i), updated: Date.now() };
    return found;
  }));
  return found;
}

export function updateItem(id, patch) {
  return change(id, (i) => ({ ...i, ...patch }));
}

export function removeItem(id) {
  commit(_items.filter((i) => i.id !== id));
}

export function addStep(id, text) {
  return change(id, (i) => ({ ...i, steps: [...i.steps, { id: newId("step"), text: String(text).trim(), done: false }] }));
}

export function toggleStep(id, stepId) {
  return change(id, (i) => ({ ...i, steps: i.steps.map((s) => (s.id === stepId ? { ...s, done: !s.done } : s)) }));
}

export function removeStep(id, stepId) {
  return change(id, (i) => ({ ...i, steps: i.steps.filter((s) => s.id !== stepId) }));
}

/**
 * Keep a found thing on an item. The same citation is kept once — except a
 * note or a diagram, which are your own words and may cite the same place.
 */
export function addRef(id, ref) {
  return change(id, (i) => {
    if (!ref.note && !ref.mermaid && i.refs.some((r) => r.cite === ref.cite && !r.note && !r.mermaid)) return i;
    return { ...i, refs: [...i.refs, { id: newId("ref"), at: Date.now(), ...ref }] };
  });
}

export function removeRef(id, refId) {
  return change(id, (i) => ({ ...i, refs: i.refs.filter((r) => r.id !== refId) }));
}

export function addJob(id, job) {
  return change(id, (i) => {
    if (i.jobs.some((j) => j.repo === job.repo && j.unit === job.unit)) return i;
    return { ...i, jobs: [...i.jobs, { at: Date.now(), ...job }], status: i.status === "idea" || i.status === "planned" ? "running" : i.status };
  });
}

/** Items whose title, notes, steps or refs contain every word of a query. */
export function matchItems(query, items = _items) {
  const words = String(query || "").toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  return items.filter((i) => {
    const hay = [i.title, i.notes, ...i.steps.map((s) => s.text), ...i.refs.map((r) => `${r.title} ${r.snippet}`)].join(" ").toLowerCase();
    return words.every((w) => hay.includes(w));
  });
}

/** One item as Markdown, to keep, send or print. */
export function toMarkdown(item) {
  const day = (t) => new Date(t).toISOString().slice(0, 10);
  const out = [`# ${item.title}`, "", `${item.kind} · ${item.status} · project ${item.project}${item.due ? ` · due ${item.due}` : ""} · started ${day(item.created)}`, ""];
  if (item.notes?.trim()) out.push(item.notes.trim(), "");
  if (item.steps.length) {
    out.push("## Steps", "");
    for (const s of item.steps) out.push(`- [${s.done ? "x" : " "}] ${s.text}`);
    out.push("");
  }
  const notes = item.refs.filter((r) => r.note);
  const diagrams = item.refs.filter((r) => r.mermaid);
  const found = item.refs.filter((r) => !r.note && !r.mermaid);
  if (notes.length) {
    out.push("## Notes", "");
    for (const r of notes) {
      out.push(`### ${r.title || r.cite}`, "", r.note.trim(), "");
      if (r.snippet) out.push(`> ${String(r.snippet).replace(/\s+/g, " ").slice(0, 400)}`, "");
      out.push(`— ${r.cite}`, "");
    }
  }
  if (diagrams.length) {
    out.push("## Diagrams", "");
    for (const r of diagrams) {
      out.push(`### ${r.title || "diagram"}`, "", r.snippet ? `${r.snippet}\n` : "", "```mermaid", r.mermaid.trim(), "```", "");
    }
  }
  if (found.length) {
    out.push("## What we found", "");
    for (const r of found) {
      out.push(`- **${r.title || r.cite}** — ${r.source}, \`${r.cite}\`${r.verdict ? ` (search verdict: ${r.verdict})` : ""}`);
      if (r.snippet) out.push(`  > ${String(r.snippet).replace(/\s+/g, " ").slice(0, 300)}`);
    }
    out.push("");
  }
  if (item.jobs.length) {
    out.push("## Jobs on AppHub", "");
    for (const j of item.jobs) out.push(`- unit \`${j.unit}\` in \`${j.repo}\``);
    out.push("");
  }
  return out.join("\n");
}
