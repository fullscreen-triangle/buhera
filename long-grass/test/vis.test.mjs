// Tests for the vis module's pure half: reading tables out of arbitrary
// values, typing their fields, and the screen's choice of charts.
import test from "node:test";
import assert from "node:assert/strict";

import { findTables, profileFields, tabulate, correlation, MAX_ROWS } from "../src/lib/vis/table.js";
import { decide, MAX_CHARTS } from "../src/lib/vis/decide.js";
import { visInstruction } from "../src/lib/surface/verbs.js";

const typeOf = (rows, name) => profileFields(rows).find((f) => f.name === name).type;
const types = (spec) => spec.charts.map((c) => c.type);

test("tables are found inside page envelopes, nested objects flattened", () => {
  const env = {
    kind: "artifact",
    result: { kind: "find", query: "q", items: [
      { name: "a", distance: 0.1, coord: { k: 0.2, t: 0.3 } },
      { name: "b", distance: 0.4, coord: { k: 0.5, t: 0.6 } },
    ] },
  };
  const [t] = findTables(env);
  assert.equal(t.path, "result.items");
  assert.deepEqual(Object.keys(t.rows[0]), ["name", "distance", "coord.k", "coord.t"]);
});

test("numeric arrays become series; name→number maps become rows", () => {
  const [s] = findTables({ chart: { values: [3, 1, 4, 1, 5, 9] } });
  assert.deepEqual(s.rows[2], { index: 2, value: 4 });
  const [m] = findTables({ perClass: { PC: 10, PE: 4, TG: 7 } });
  assert.deepEqual(m.rows, [{ key: "PC", value: 10 }, { key: "PE", value: 4 }, { key: "TG", value: 7 }]);
});

test("label→value rows chart only when every value is a bare number", () => {
  assert.equal(findTables({ rows: [["used", "0 B"], ["quota", "3.0 GB"]] }).length, 0, "units are not numbers");
  const [t] = findTables({ rows: [["a", "12"], ["b", "1,300"], ["c", 7]] });
  assert.deepEqual(t.rows.map((r) => r.value), [12, 1300, 7]);
});

test("a list of typed results is walked into, not charted itself", () => {
  const env = { kind: "multi", results: [
    { kind: "find", items: [{ name: "x", d: 1 }, { name: "y", d: 2 }] },
    { kind: "stats", stats: { objects: 2 } },
  ] };
  const paths = findTables(env).map((t) => t.path);
  assert.ok(paths.includes("results[0].items"));
  assert.ok(!paths.includes("results"));
});

test("field typing reads values, not names — except time", () => {
  const rows = Array.from({ length: 30 }, (_, i) => ({
    mz: 100 + i * 1.7,
    shell: (i % 3) + 1,
    cls: ["PC", "PE"][i % 2],
    id: `r${i}`,
    at: 1.7e12 + i * 1000,
    note: `observation ${i}: ` + "x".repeat(60),
    same: 5,
    when: `2026-09-${String((i % 28) + 1).padStart(2, "0")}`,
  }));
  assert.equal(typeOf(rows, "mz"), "quantitative");
  assert.equal(typeOf(rows, "shell"), "ordinal");
  assert.equal(typeOf(rows, "cls"), "categorical");
  assert.equal(typeOf(rows, "id"), "identifier");
  assert.equal(typeOf(rows, "at"), "temporal");
  assert.equal(typeOf(rows, "note"), "text");
  assert.equal(typeOf(rows, "same"), "constant");
  assert.equal(typeOf(rows, "when"), "temporal");
  // An epoch-sized number without a time-like name stays a number.
  assert.equal(typeOf(rows.map((r) => ({ big: r.at })), "big"), "quantitative");
});

test("tabulate caps rows and honours focus", () => {
  const big = Array.from({ length: MAX_ROWS + 10 }, (_, i) => ({ v: i * 1.5 }));
  const ds = tabulate({ big, small: [{ w: 2 }, { w: 5 }, { w: 3 }] });
  assert.equal(ds.path, "big");
  assert.equal(ds.rows.length, MAX_ROWS);
  assert.equal(ds.truncated, true);
  assert.equal(tabulate({ big, small: [{ w: 2 }, { w: 5 }, { w: 3 }] }, { focus: "small" }).path, "small");
  assert.equal(tabulate({ kind: "text", lines: ["hello"] }), null);
});

test("correlation is Pearson r", () => {
  const rows = [1, 2, 3, 4, 5].map((x) => ({ x, y: 2 * x + 1, z: -x }));
  assert.ok(Math.abs(correlation(rows, "x", "y") - 1) < 1e-9);
  assert.ok(Math.abs(correlation(rows, "x", "z") + 1) < 1e-9);
});

test("the screen picks relation, then distributions, then groups", () => {
  const rows = Array.from({ length: 40 }, (_, i) => ({ a: i * 0.5 + (i % 3), b: i * 2 + (i % 5), c: Math.sqrt(i * 7), grp: ["x", "y"][i % 2] }));
  const ds = tabulate(rows);
  const spec = decide(ds);
  assert.deepEqual(types(spec), ["scatter", "histogram", "histogram", "histogram", "bars"]);
  const sc = spec.charts[0];
  assert.deepEqual([sc.fields.x, sc.fields.y].sort(), ["a", "b"], "the most correlated pair");
  assert.equal(sc.fields.color, "grp", "≤ 3 categories colour the scatter");
  assert.ok(spec.charts.every((c) => c.reason && c.id));
});

test("labelled rows get ranked magnitudes; one row gets tiles; series get a line", () => {
  const ranked = decide(tabulate([{ name: "a", d: 0.1 }, { name: "b", d: 0.5 }, { name: "c", d: 0.3 }, { name: "e", d: 0.9 }]));
  assert.deepEqual(types(ranked), ["ranked"]);
  assert.ok(!ranked.notes.some((n) => /too few/.test(n)), "the ranked chart already shows every value");
  const bare = decide(tabulate([{ d: 0.1 }, { d: 0.5 }, { d: 0.3 }]));
  assert.match(bare.notes.join(" "), /too few for a distribution/);

  assert.deepEqual(types(decide(tabulate([{ objects: 3, pve: 2 }]))), ["tiles"]);

  const series = decide(tabulate({ values: [1, 3, 2, 5, 4, 6, 5, 8, 7, 9] }));
  assert.equal(series.charts[0].type, "series");
});

test("time leads, charts are capped, many categories fold", () => {
  const rows = Array.from({ length: 60 }, (_, i) => ({
    at: 1.7e12 + i * 6e4, a: i, b: i % 7, c: Math.sin(i), d: i * i, e: (i * 13) % 17,
    cat: `c${i % 20}`,
  }));
  const spec = decide(tabulate(rows));
  assert.equal(spec.charts[0].type, "timeline");
  assert.ok(spec.charts.length <= MAX_CHARTS);
  const many = decide(tabulate(Array.from({ length: 60 }, (_, i) => ({ cat: `c${i % 20}` }))));
  assert.match(many.charts[0].reason, /fold into "other"/);
});

test("constants and free text are reported, never charted", () => {
  const rows = Array.from({ length: 10 }, (_, i) => ({ v: i * i, k: "same", t: `entry ${i} ` + "long ".repeat(20) }));
  const spec = decide(tabulate(rows));
  assert.ok(spec.notes.some((n) => /k is same in every row/.test(n)));
  assert.ok(spec.notes.some((n) => /t is free text/.test(n)));
  assert.ok(spec.charts.every((c) => !JSON.stringify(c.fields).includes('"k"')));
});

test("the chart verb: page in view, named sources, or nothing to chart", () => {
  const page = { n: 3, source: { type: "utterance", text: "find x" }, envelope: {} };
  assert.deepEqual(visInstruction("chart", page), { page, focus: "" });
  assert.deepEqual(visInstruction("plot this records", page), { page, focus: "records" });
  assert.equal(visInstruction("chart memory", null), "memory");
  assert.equal(visInstruction("visualise audit", page), "audit");
  assert.deepEqual(visInstruction("chart", null), { missing: true });
  assert.equal(visInstruction("charter a boat", page), null);
  assert.equal(visInstruction("find charts", page), null);
});

test("a number that is a linear function of another is charted once", () => {
  const rows = Array.from({ length: 30 }, (_, i) => {
    const f = 50 + i * 1.3 + (i % 4);
    return { f, omega: 2 * Math.PI * f, bits: Math.sin(i) * 3 + i * 0.1 };
  });
  const spec = decide(tabulate(rows));
  const sc = spec.charts.find((c) => c.type === "scatter");
  assert.ok(sc && !(["f", "omega"].includes(sc.fields.x) && ["f", "omega"].includes(sc.fields.y)), "not the trivial pair");
  assert.ok(spec.notes.some((n) => /moves exactly with/.test(n)));
  assert.equal(spec.charts.filter((c) => c.type === "histogram" && ["f", "omega"].includes(c.fields.field)).length, 1);
});

test("a 1…n counter is an identifier, not a measure", () => {
  const rows = Array.from({ length: 12 }, (_, i) => ({ act: i + 1, ms: [4, 47, 16, 111][i % 4] + i, at: 1.7e12 + i * 5e3 }));
  const f = profileFields(rows).find((x) => x.name === "act");
  assert.equal(f.type, "identifier");
  const tl = decide(tabulate(rows)).charts.find((c) => c.type === "timeline");
  assert.equal(tl.fields.value, "ms", "the measure, not the counter, is followed over time");
});
