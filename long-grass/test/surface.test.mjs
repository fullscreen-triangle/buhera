// Tests for the blank surface's pure layer: the page book (append-only,
// inert snapshots, fork-to-end, persistence) and the edge filing.
import test from "node:test";
import assert from "node:assert/strict";

import {
  appendPage, emptyBook, loadBook, pageContext, saveBook, snapshot, STORAGE_KEY,
} from "../src/lib/surface/book.js";
import {
  EDGES, edgeOf, glanceOf, isTemplate, modulesAt, searchModules,
} from "../src/lib/surface/edges.js";

const utter = (text) => ({ type: "utterance", text });
const env = (lines) => ({ kind: "text", lines });

function memoryStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); },
    raw: m,
  };
}

test("appendPage numbers pages and never mutates the previous book", () => {
  const b0 = emptyBook();
  const b1 = appendPage(b0, { source: utter("a"), envelope: env(["1"]) });
  const b2 = appendPage(b1, { source: utter("b"), envelope: env(["2"]) });
  assert.equal(b0.pages.length, 0);
  assert.equal(b1.pages.length, 1);
  assert.deepEqual(b2.pages.map((p) => p.n), [1, 2]);
  assert.equal(b2.pages[0], b1.pages[0], "earlier pages are shared, not copied or rewritten");
});

test("a fork from an older page appends to the end and records where it began", () => {
  let b = emptyBook();
  for (const t of ["a", "b", "c"]) b = appendPage(b, { source: utter(t), envelope: env([t]) });
  const forked = appendPage(b, { source: utter("from a"), envelope: env(["x"]), from: 1 });
  assert.equal(forked.pages.length, 4);
  assert.equal(forked.pages[3].from, 1);
  assert.deepEqual(forked.pages.slice(0, 3), b.pages, "history before the fork is untouched");
});

test("pages are inert: frozen, and detached from the live result", () => {
  const live = { kind: "text", lines: ["before"] };
  const b = appendPage(emptyBook(), { source: utter("q"), envelope: live });
  live.lines.push("after");
  const page = b.pages[0];
  assert.deepEqual(page.envelope.lines, ["before"], "later mutation of the live value does not reach the page");
  assert.ok(Object.isFrozen(page) && Object.isFrozen(page.envelope.lines));
  assert.throws(() => { "use strict"; page.envelope.lines.push("x"); });
});

test("snapshot drops what cannot be drawn and survives what cannot be serialised", () => {
  assert.deepEqual(snapshot({ a: 1, f: () => 1 }), { a: 1 });
  const cyclic = {}; cyclic.self = cyclic;
  const s = snapshot(cyclic);
  assert.equal(s.kind, "text");
  assert.match(s.lines[0], /could not be snapshotted/);
});

test("pageContext exposes only the page's visible content", () => {
  assert.equal(pageContext(null), null);
  const b = appendPage(emptyBook(), { source: utter("q"), envelope: env(["r"]), at: 5 });
  assert.deepEqual(pageContext(b.pages[0]), { n: 1, source: utter("q"), envelope: env(["r"]) });
});

test("the book round-trips through storage", () => {
  const store = memoryStorage();
  let b = emptyBook();
  b = appendPage(b, { source: utter("a"), envelope: env(["1"]) });
  b = appendPage(b, { source: utter("b"), envelope: env(["2"]), from: 1 });
  assert.equal(saveBook(b, store), true);
  const back = loadBook(store);
  assert.deepEqual(back.pages.map((p) => p.source.text), ["a", "b"]);
  assert.equal(back.pages[1].from, 1);
});

test("loadBook tolerates garbage and missing storage", () => {
  const store = memoryStorage();
  store.setItem(STORAGE_KEY, "{not json");
  assert.equal(loadBook(store).pages.length, 0);
  assert.equal(loadBook(null).pages.length, 0);
});

test("saveBook trims the oldest pages to fit, renumbering on load", () => {
  const store = memoryStorage();
  let b = emptyBook();
  const big = "x".repeat(900_000);
  for (let i = 0; i < 3; i++) b = appendPage(b, { source: utter(`p${i}`), envelope: env([big]), from: i || null });
  assert.equal(saveBook(b, store), true);
  const back = loadBook(store);
  assert.ok(back.pages.length < 3, "something had to be dropped");
  assert.equal(back.pages.at(-1).source.text, "p2", "the newest page is kept");
  assert.equal(back.pages[0].n, 1);
  assert.ok(back.pages.every((p) => p.from === null), "stale fork links are dropped");
});

test("every edge module is filed exactly once", () => {
  const ids = Object.values(EDGES).flatMap((e) => e.entries.map((x) => x.id));
  assert.equal(new Set(ids).size, ids.length);
  assert.equal(edgeOf("gateway"), "right");
  assert.equal(edgeOf("disk"), "bottom");
  assert.equal(edgeOf("spraypaint"), "top");
  assert.equal(edgeOf("lavoisier"), null, "domain modules live only in the left column");
  assert.deepEqual(glanceOf("catalysts"), "list");
  assert.equal(glanceOf("restart"), undefined, "restart has no glance — opening it must not act");
});

test("modulesAt keeps table order and skips unregistered ids", () => {
  const reg = [{ id: "catalysts" }, { id: "gateway" }, { id: "echo" }];
  assert.deepEqual(modulesAt("right", reg).map((m) => m.id), ["gateway", "catalysts"]);
});

test("searchModules matches id and description, sorted by id", () => {
  const reg = [
    { id: "zeta", description: "mass spectrometry" },
    { id: "alpha", description: "memory" },
    { id: "spray", description: "search" },
  ];
  assert.deepEqual(searchModules(reg, "").map((m) => m.id), ["alpha", "spray", "zeta"]);
  assert.deepEqual(searchModules(reg, "MASS").map((m) => m.id), ["zeta"]);
  assert.deepEqual(searchModules(reg, "spr").map((m) => m.id), ["spray"]);
});

test("isTemplate spots instructions that need filling in", () => {
  assert.equal(isTemplate('memory store "<name>" = "<text>"'), true);
  assert.equal(isTemplate('dispatch("spraypaint", { kind: "web", query: "..." })'), true);
  assert.equal(isTemplate('dispatch("disk", "usage")'), false);
  assert.equal(isTemplate("kernel stats"), false);
});

test("searchModules ranks name matches above description matches", () => {
  const reg = [
    { id: "sbs", description: "flux visibility V" },
    { id: "vis", description: "charts" },
    { id: "revise", description: "" },
  ];
  assert.deepEqual(searchModules(reg, "vis").map((m) => m.id), ["vis", "revise", "sbs"]);
});
