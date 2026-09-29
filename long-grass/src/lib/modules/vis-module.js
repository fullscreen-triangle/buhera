/* ============================================================================
 * vis — the surface's own charts.
 *
 * Every other module draws the charts its author built in. vis draws charts
 * for data it has never seen: it reads the tables out of whatever it is given
 * (lib/vis/table.js), and the screen decides which charts they get
 * (lib/vis/decide.js). The page draws them with d3, linked through one
 * crossfilter (components/vis/VisBoard.js), so a selection in any chart
 * rescopes the rest.
 *
 * Sources:
 *   { page, focus? }   the page being viewed (what "chart this" charts);
 *                      `focus` picks a table by name when the page has several
 *   { data, title? }   any value — rows, a numeric array, a module's output
 *   "memory"           vaHera kernel memory: every stored object's S-coord
 *   "audit"            the federation's audit log: every act, its module, time
 *   "pages"            the surface's own page book
 *
 * Output: { kind: "vis", title, source, dataset, charts, notes }.
 * Stores nothing (empty-dictionary principle): the dataset lives only in the
 * page that shows it.
 * ========================================================================== */

import { tabulate } from "@/lib/vis/table";
import { decide } from "@/lib/vis/decide";
import { getAuditLog } from "@/lib/modules/registry";
import { getKernel } from "@/lib/modules/vahera-module";
import { loadBook } from "@/lib/surface/book";

function done(output_delta, ok = true) {
  return { ok, output_delta, residue: ok ? 1 : 0, completed: true };
}

function nothing(message) {
  return done({ kind: "text", lines: [message] }, false);
}

// ── sources ──────────────────────────────────────────────────────────────

function memoryRows() {
  return [...getKernel().store.values()].map((o) => ({
    name: o.metadata?.name ?? o.address.slice(0, 8),
    kind: o.metadata?.kind ?? "object",
    tier: o.tier,
    S_k: o.coord.k,
    S_t: o.coord.t,
    S_e: o.coord.e,
    address: o.address,
  }));
}

function auditRows() {
  return getAuditLog().map((e) => ({
    act: e.act_id,
    module: e.module_id,
    ok: !!e.result?.ok,
    duration_ms: e.wall_clock_ms,
    at: Date.parse(e.timestamp) || null,
  }));
}

function pageRows() {
  return loadBook().pages.map((p) => ({
    page: `p${p.n}`,
    at: p.at,
    started_by: p.source?.type === "module" ? "module" : "writing",
    shows: p.envelope?.kind === "artifact" ? p.envelope.result?.kind ?? "artifact" : p.envelope?.kind ?? "?",
    forked: p.from != null,
  }));
}

const NAMED = {
  memory: { title: "kernel memory", rows: memoryRows, empty: "kernel memory is empty — store something first." },
  audit: { title: "audit log", rows: auditRows, empty: "the audit log is empty — nothing has been dispatched yet." },
  pages: { title: "page book", rows: pageRows, empty: "no pages yet." },
};

function describeSource(instruction) {
  if (typeof instruction === "string") {
    const key = instruction.trim().toLowerCase();
    if (NAMED[key]) return { key, title: NAMED[key].title, value: NAMED[key].rows(), empty: NAMED[key].empty };
    return null;
  }
  if (instruction && typeof instruction === "object") {
    if (instruction.page) {
      const p = instruction.page;
      const words = p.source?.type === "module" ? p.source.moduleId : p.source?.text;
      return {
        key: "page",
        title: `page ${p.n}${words ? ` — ${words}` : ""}`,
        value: p.envelope,
        focus: instruction.focus || "",
        empty: `page ${p.n} holds no table to chart.`,
      };
    }
    if ("data" in instruction) {
      return { key: "data", title: instruction.title || "data", value: instruction.data, focus: instruction.focus || "", empty: "that data holds no table to chart." };
    }
  }
  return null;
}

// ── the module ───────────────────────────────────────────────────────────

export const visModule = {
  id: "vis",

  describe() {
    return {
      id: "vis",
      description:
        "Visualisation: charts on demand for data it has never seen. It reads " +
        "the tables out of what it is given, decides which charts they get and " +
        "why, and links them through one crossfilter so a selection in any " +
        "chart rescopes the rest. On a page, write \"chart\" to chart that page.",
      instructions: [
        'dispatch("vis", "memory")',
        'dispatch("vis", "audit")',
        'dispatch("vis", "pages")',
        'dispatch("vis", { data: [{ x: 1, y: 2 }, { x: 2, y: 3 }, { x: 3, y: 5 }] })',
        'dispatch("vis", { data: <any value>, focus: "<table name>" })',
      ],
    };
  },

  async execute(instruction) {
    const src = describeSource(instruction);
    if (!src) {
      return nothing('vis: give it something to chart — "memory", "audit", "pages", { data }, or { page }.');
    }
    const dataset = tabulate(src.value, { focus: src.focus || "" });
    if (!dataset) return nothing(src.empty);

    const { charts, notes } = decide(dataset);
    if (dataset.truncated) notes.unshift(`first ${dataset.rows.length.toLocaleString("en-US")} of ${dataset.total.toLocaleString("en-US")} rows.`);
    if (dataset.others.length) {
      notes.push(`other tables here: ${dataset.others.map((o) => `${o.path} (${o.rows})`).join(", ")} — "chart <name>" picks one.`);
    }

    return done({
      kind: "vis",
      title: src.title,
      source: src.key,
      dataset: { path: dataset.path, fields: dataset.fields, rows: dataset.rows, total: dataset.total },
      charts,
      notes,
    });
  },

  outputCell() {
    return { kind: "vis_cell" };
  },
};
