/* ============================================================================
 * vis/table — find the tables inside any value, and type their fields.
 *
 * The vis module never asks "what data do you want charted?". It is handed
 * whatever is on screen — a page's envelope, a module's output, kernel
 * memory — and reads the tables out of it itself:
 *
 *   • an array of objects             → rows (nested objects flattened: coord.k)
 *   • an array of numbers             → a series { index, value }
 *   • an array of [label, value] rows → rows { label, value }
 *   • an object of name → number      → rows { key, value }   (≥ 3 entries)
 *
 * Each field is then typed from its values alone:
 *
 *   quantitative  numbers with enough distinct values to have a distribution
 *   ordinal       small-cardinality integers (shell n, tier index)
 *   categorical   strings/booleans with few distinct values
 *   temporal      ISO dates, or epoch-ms numbers under a time-like name
 *   identifier    one distinct value per row (a name, an address) — labels
 *   text          long free text — never charted
 *   constant      one value everywhere — reported, never charted
 *
 * Pure: no DOM, no d3. The Node test runner exercises it directly.
 * ========================================================================== */

export const MAX_ROWS = 5000;
const MAX_DEPTH = 6;
const MAX_FLAT_DEPTH = 2;

const isPlainObject = (v) => v != null && typeof v === "object" && !Array.isArray(v);
const isPrimitive = (v) => v == null || ["number", "string", "boolean"].includes(typeof v);

// ── finding tables ───────────────────────────────────────────────────────

/** Flatten one row object to primitive leaves (`coord.k`), two levels deep. */
export function flattenRow(obj, prefix = "", depth = 0, out = {}) {
  for (const [k, v] of Object.entries(obj)) {
    if (k.startsWith("_")) continue;
    const key = prefix ? `${prefix}.${k}` : k;
    if (isPrimitive(v)) {
      if (typeof v === "number" && !isFinite(v)) continue;
      out[key] = v;
    } else if (isPlainObject(v) && depth < MAX_FLAT_DEPTH) {
      flattenRow(v, key, depth + 1, out);
    }
  }
  return out;
}

function tableFromArray(arr) {
  if (arr.length === 0) return null;
  // One object is still a table (one row: its numbers become tiles); a
  // series or a set of label/value pairs needs more than one entry.
  if (arr.length === 1 && !isPlainObject(arr[0])) return null;
  if (arr.every((v) => typeof v === "number" && isFinite(v))) {
    if (arr.length < 3) return null;
    return arr.map((value, index) => ({ index, value }));
  }
  if (arr.every((v) => Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && isPrimitive(v[1]))) {
    // label → value rows: a table only if every value is a bare number.
    const rows = arr.map(([label, value]) => ({ label: label.trim(), value: numeric(value) }));
    return rows.every((r) => r.value != null) ? rows : null;
  }
  if (arr.every(isPlainObject)) {
    // A list of results (every item a typed artifact carrying nested
    // payloads) is structure, not data: walk into it, do not chart it.
    const results = arr.every(
      (o) => typeof o.kind === "string" && Object.values(o).some((v) => v && typeof v === "object")
    );
    if (results) return null;
    const rows = arr.map((o) => flattenRow(o));
    return rows.some((r) => Object.keys(r).length > 0) ? rows : null;
  }
  return null;
}

function tableFromMap(obj) {
  const entries = Object.entries(obj);
  if (entries.length < 3 || !entries.every(([, v]) => typeof v === "number" && isFinite(v))) return null;
  return entries.map(([key, value]) => ({ key, value }));
}

// A bare number, or a string that is exactly one ("1.5", "2,300"). Strings
// with units ("3.0 GB", "42 ms") are NOT numbers here: stripping the unit
// would put gigabytes and bytes on one axis.
function numeric(v) {
  if (typeof v === "number") return isFinite(v) ? v : null;
  if (typeof v !== "string") return null;
  const t = v.trim();
  if (!/\d/.test(t) || !/^-?(\d{1,3}(,\d{3})+|\d+)?(\.\d+)?(e[-+]?\d+)?$/i.test(t)) return null;
  const n = Number(t.replace(/,/g, ""));
  return isFinite(n) ? n : null;
}

/**
 * Every table found in `value`, best first. Each: { path, rows }.
 * `path` names where it was found ("results", "summary.perClass").
 */
export function findTables(value) {
  const found = [];
  const seen = new Set();
  (function walk(v, path, depth) {
    if (v == null || typeof v !== "object" || depth > MAX_DEPTH || seen.has(v)) return;
    seen.add(v);
    if (Array.isArray(v)) {
      const rows = tableFromArray(v);
      if (rows) found.push({ path: path || "data", rows });
      // Arrays of objects may hold tables of their own (multi results).
      v.forEach((item, i) => walk(item, path ? `${path}[${i}]` : `[${i}]`, depth + 1));
      return;
    }
    const map = tableFromMap(v);
    if (map) found.push({ path: path || "data", rows: map });
    for (const [k, child] of Object.entries(v)) {
      if (k.startsWith("_")) continue;
      walk(child, path ? `${path}.${k}` : k, depth + 1);
    }
  })(value, "", 0);

  const scored = found.map((t) => ({ ...t, score: tableScore(t.rows) }));
  scored.sort((a, b) => b.score - a.score);
  return scored.filter((t) => t.score > 0).map(({ path, rows }) => ({ path, rows }));
}

// Rows × chartable fields: a table worth charting has both.
function tableScore(rows) {
  if (!rows.length) return 0;
  const fields = profileFields(rows);
  const chartable = fields.filter((f) => CHARTABLE.has(f.type)).length;
  return chartable === 0 ? 0 : Math.log2(rows.length + 1) * chartable;
}

// ── typing fields ────────────────────────────────────────────────────────

export const CHARTABLE = new Set(["quantitative", "ordinal", "categorical", "temporal"]);

const TIME_NAME = /(^|[._])(at|time|timestamp|date|ts|when|created|updated)$|_ms$|_at$/i;
const ISO_DATE = /^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$/;
const EPOCH_MS = [1e11, 1e13]; // 1973 – 2286

/**
 * Type every field of `rows`. Returns [{ name, type, distinct, missing,
 * min?, max?, mean?, sd?, categories? }] in first-seen key order.
 */
export function profileFields(rows) {
  const keys = [];
  const seenKey = new Set();
  for (const r of rows) for (const k of Object.keys(r)) if (!seenKey.has(k)) { seenKey.add(k); keys.push(k); }
  return keys.map((name) => profileField(name, rows.map((r) => r[name])));
}

function profileField(name, values) {
  const present = values.filter((v) => v != null && v !== "");
  const missing = values.length - present.length;
  const distinctSet = new Set(present.map((v) => (typeof v === "object" ? JSON.stringify(v) : v)));
  const distinct = distinctSet.size;
  const base = { name, distinct, missing, count: present.length };

  if (present.length === 0) return { ...base, type: "empty" };
  // "The same in every row" only means something when there is more than one.
  if (distinct === 1 && values.length > 1) return { ...base, type: "constant", value: present[0] };
  if (values.length === 1) {
    return typeof present[0] === "number"
      ? { ...base, ...stats(present), type: "quantitative" }
      : { ...base, type: "identifier" };
  }

  if (present.every((v) => typeof v === "number")) {
    const st = stats(present);
    const epochLike = st.min >= EPOCH_MS[0] && st.max <= EPOCH_MS[1];
    if (epochLike && TIME_NAME.test(name)) return { ...base, ...st, type: "temporal", unit: "ms" };
    const integers = present.every(Number.isInteger);
    // 1, 2, 3 … n, one per row: a counter (act id, index), not a measure.
    if (integers && present.length >= 3 && distinct === present.length && st.max - st.min + 1 === present.length) {
      return { ...base, ...st, type: "identifier", sequence: true };
    }
    if (integers && distinct <= 12 && distinct < present.length / 3) {
      return { ...base, ...st, type: "ordinal", categories: [...distinctSet].sort((a, b) => a - b) };
    }
    return { ...base, ...st, type: "quantitative" };
  }

  if (present.every((v) => typeof v === "boolean")) {
    return { ...base, type: "categorical", categories: [...distinctSet].map(String) };
  }

  if (present.every((v) => typeof v === "string")) {
    if (present.every((s) => ISO_DATE.test(s.trim()) && !isNaN(Date.parse(s)))) {
      const t = present.map((s) => Date.parse(s));
      return { ...base, ...stats(t), type: "temporal", unit: "iso" };
    }
    const avgLen = present.reduce((s, v) => s + v.length, 0) / present.length;
    if (avgLen > 48) return { ...base, type: "text" };
    if (distinct === present.length && present.length > 3) return { ...base, type: "identifier" };
    if (distinct <= 24) return { ...base, type: "categorical", categories: [...distinctSet] };
    return { ...base, type: distinct > present.length * 0.8 ? "identifier" : "categorical", categories: [...distinctSet] };
  }

  return { ...base, type: "text" }; // mixed types: show in the table, never chart
}

function stats(nums) {
  let min = Infinity, max = -Infinity, sum = 0;
  for (const n of nums) { if (n < min) min = n; if (n > max) max = n; sum += n; }
  const mean = sum / nums.length;
  let ss = 0;
  for (const n of nums) ss += (n - mean) ** 2;
  return { min, max, mean, sd: Math.sqrt(ss / nums.length) };
}

/** Pearson r between two numeric fields over rows where both are present. */
export function correlation(rows, a, b) {
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (const r of rows) {
    const x = r[a], y = r[b];
    if (typeof x !== "number" || typeof y !== "number") continue;
    n++; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
  }
  if (n < 3) return 0;
  const cov = sab / n - (sa / n) * (sb / n);
  const va = saa / n - (sa / n) ** 2;
  const vb = sbb / n - (sb / n) ** 2;
  return va > 0 && vb > 0 ? cov / Math.sqrt(va * vb) : 0;
}

/**
 * Build a dataset from any value: the best table (or the one whose path
 * contains `focus`), capped at MAX_ROWS, with typed fields and the list of
 * other tables that were on offer.
 */
export function tabulate(value, { focus = "" } = {}) {
  const tables = findTables(value);
  if (!tables.length) return null;
  const want = focus.trim().toLowerCase();
  const chosen = (want && tables.find((t) => t.path.toLowerCase().includes(want))) || tables[0];
  const total = chosen.rows.length;
  const rows = total > MAX_ROWS ? chosen.rows.slice(0, MAX_ROWS) : chosen.rows;
  return {
    path: chosen.path,
    rows,
    total,
    truncated: total > rows.length,
    fields: profileFields(rows),
    others: tables.filter((t) => t !== chosen).map((t) => ({ path: t.path, rows: t.rows.length })),
  };
}
