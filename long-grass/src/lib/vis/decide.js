/* ============================================================================
 * vis/decide — the screen decides which charts a dataset gets.
 *
 * Nobody picks a chart type. Given a typed dataset (vis/table.js), a fixed
 * rule set — the form heuristic "the data's job picks the chart" — chooses
 * the charts, in order, and writes down why for each one. The reasons are
 * drawn on the page: a chart the reader cannot account for is a chart they
 * cannot trust.
 *
 *   job in the data                         → chart
 *   ─────────────────────────────────────────────────────────────────────
 *   a single row of numbers                 → tiles       (a number, not a chart)
 *   an ordered numeric series               → series      (line over index)
 *   a time field                            → timeline    (value, or count, over time)
 *   two or more numeric fields              → scatter     (the most correlated pair)
 *   one label per row + a number            → ranked      (magnitude per item, ≤ 40)
 *   a numeric field with enough rows        → histogram   (its distribution)
 *   a field with a few categories           → bars        (count per category)
 *
 * Before choosing: a number that is an exact linear function of another
 * (|r| ≥ 0.999 — ω = 2πf) is the same quantity and is charted once; a 1…n
 * counter is typed an identifier (vis/table.js), never a measure; constants
 * and free text are reported in the notes, never charted.
 *
 * At most MAX_CHARTS charts. Every chart that filters (histogram, bars,
 * ranked, scatter, timeline) joins one crossfilter in the renderer, so a
 * selection in any of them rescopes all the others.
 *
 * Pure: the output is a list of plain specs; vis/VisBoard draws them.
 * ========================================================================== */

import { correlation } from "@/lib/vis/table";

export const MAX_CHARTS = 6;
const MAX_RANKED = 40;
const MIN_HIST_ROWS = 8;
const MAX_BARS = 12; // categories past this fold into "other"
const REDUNDANT_R = 0.999;

const fmt = (n) => (Math.abs(n) >= 1000 || Number.isInteger(n) ? n.toLocaleString("en-US", { maximumFractionDigits: 0 }) : n.toPrecision(3));

// Continuous fields with the most distinct values carry the most shape.
function byResolution(a, b) {
  return b.distinct - a.distinct || a.name.localeCompare(b.name);
}

/**
 * Decide the charts for a dataset.
 * @param {{ rows: object[], fields: object[] }} dataset
 * @returns {{ charts: object[], notes: string[] }}
 */
export function decide(dataset) {
  const { rows, fields } = dataset;
  const n = rows.length;
  const charts = [];
  const notes = [];
  const add = (spec) => { if (charts.length < MAX_CHARTS) charts.push({ id: `c${charts.length + 1}`, ...spec }); };

  // A number that is an exact linear function of another (ω = 2πf, a value
  // and its percentage) is the same quantity twice. Chart it once and say so;
  // otherwise the "most related pair" is always the trivial one.
  const quant = [];
  for (const f of fields.filter((f) => f.type === "quantitative").sort(byResolution)) {
    const twin = quant.find((q) => Math.abs(correlation(rows, q.name, f.name)) >= REDUNDANT_R);
    if (twin) notes.push(`${f.name} moves exactly with ${twin.name} (|r| ≥ ${REDUNDANT_R}) — the same quantity, charted once as ${twin.name}.`);
    else quant.push(f);
  }
  const temporal = fields.filter((f) => f.type === "temporal");
  const cats = fields.filter((f) => (f.type === "categorical" || f.type === "ordinal") && f.distinct >= 2);
  const labels = fields.filter((f) => f.type === "identifier" || (f.type === "categorical" && f.distinct === n && n >= 2));

  for (const f of fields.filter((f) => f.type === "constant")) {
    notes.push(`${f.name} is ${String(f.value)} in every row — not charted.`);
  }
  for (const f of fields.filter((f) => f.type === "text")) {
    notes.push(`${f.name} is free text — in the table, not charted.`);
  }

  if (n === 0) return { charts, notes: [...notes, "no rows."] };

  // A single row: the numbers are the answer.
  if (n === 1) {
    const nums = fields.filter((f) => typeof rows[0][f.name] === "number");
    if (nums.length) {
      add({ type: "tiles", title: "values", fields: { values: nums.map((f) => f.name) },
        reason: "one row — each number stands alone; a chart of one mark says less than the number." });
    }
    return { charts, notes };
  }

  // An ordered series (a numeric array on the page).
  const isSeries = fields.some((f) => f.name === "index") && fields.some((f) => f.name === "value" && f.type === "quantitative");
  if (isSeries) {
    add({ type: "series", title: "value over index", fields: { x: "index", y: "value" },
      reason: `an ordered series of ${n} numbers — its shape in order.` });
  }

  // Time.
  if (temporal.length) {
    const t = temporal[0];
    const v = quant[0];
    add(v
      ? { type: "timeline", title: `${v.name} over ${t.name}`, fields: { time: t.name, value: v.name, unit: t.unit },
          reason: `${t.name} is a time — ${v.name}, the most varied number, laid along it.` }
      : { type: "timeline", title: `rows over ${t.name}`, fields: { time: t.name, unit: t.unit },
          reason: `${t.name} is a time and there is no number to follow — how many rows fall when.` });
  }

  // Two numbers: the pair that moves together most.
  if (quant.length >= 2 && n >= 5 && !isSeries) {
    const pool = quant.slice(0, 5);
    let best = null;
    for (let i = 0; i < pool.length; i++) {
      for (let j = i + 1; j < pool.length; j++) {
        const r = correlation(rows, pool[i].name, pool[j].name);
        if (!best || Math.abs(r) > Math.abs(best.r)) best = { x: pool[i].name, y: pool[j].name, r };
      }
    }
    // Colour by a category only when it has ≤ 3 values: past three, scatter
    // hues stop being told apart (validated all-pairs cap).
    const color = cats.find((c) => c.distinct <= 3 && c.type === "categorical");
    add({
      type: "scatter",
      title: `${best.y} against ${best.x}`,
      fields: { x: best.x, y: best.y, color: color?.name || null, categories: color?.categories || null },
      reason: `${best.x} and ${best.y} are the most related pair of numbers (r = ${best.r.toFixed(2)})` +
        (color ? `, coloured by ${color.name}.` : "."),
    });
  }

  // One label per row and a number: magnitude per item.
  if (labels.length && quant.length && n <= MAX_RANKED) {
    const label = labels[0];
    const v = quant[0];
    add({ type: "ranked", title: `${v.name} by ${label.name}`, fields: { label: label.name, value: v.name },
      reason: `each row is one ${label.name} — ${v.name} compared across them, largest first.` });
  }

  // Distributions.
  if (n >= MIN_HIST_ROWS) {
    for (const f of quant.slice(0, 3)) {
      if (isSeries && f.name === "index") continue;
      add({ type: "histogram", title: `${f.name}`, fields: { field: f.name, min: f.min, max: f.max },
        reason: `${f.name} is numeric with ${f.distinct} distinct values (${fmt(f.min)} – ${fmt(f.max)}) — its distribution.` });
    }
  } else if (quant.length && !charts.some((c) => c.type === "ranked")) {
    notes.push(`${n} rows is too few for a distribution — see the table.`);
  }

  // Categories.
  for (const f of cats) {
    if (f.distinct === n) continue; // a label, not a grouping
    const folded = f.distinct > MAX_BARS;
    add({ type: "bars", title: `rows by ${f.name}`, fields: { field: f.name, ordinal: f.type === "ordinal", top: MAX_BARS },
      reason: `${f.name} sorts rows into ${f.distinct} groups — how many in each` +
        (folded ? `; the smallest ${f.distinct - MAX_BARS + 1} fold into "other".` : ".") });
  }

  if (!charts.length) notes.push("nothing in this table has a shape to chart — see the table.");
  return { charts, notes };
}
