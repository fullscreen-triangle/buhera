// Tests for the surface settings store: sections merge, plans and reports
// accumulate per project, and every change notifies subscribers.
import test from "node:test";
import assert from "node:assert/strict";

import { getSettings, update, subscribe, addPlan, addReport, resetSection, DEFAULTS } from "../src/lib/surface/settings.js";

test("defaults are complete and sections merge on update", () => {
  assert.equal(getSettings().preferences.textScale, 1);
  update("preferences", { textScale: 1.2 });
  assert.equal(getSettings().preferences.textScale, 1.2);
  assert.equal(getSettings().preferences.spacing, DEFAULTS.preferences.spacing, "untouched keys survive");
  resetSection("preferences");
  assert.equal(getSettings().preferences.textScale, 1);
});

test("subscribers hear every change", () => {
  let n = 0;
  const off = subscribe(() => n++);
  update("code", { showTrace: true });
  update("project", "enzymes");
  off();
  update("code", { showTrace: false });
  assert.equal(n, 2);
  assert.equal(getSettings().project, "enzymes");
});

test("plans and reports are kept newest first, stamped with the active project", () => {
  update("project", "p450");
  const a = addPlan({ source: "store a note", script: 'memory store "a" = "b"', by: "person" });
  const b = addPlan({ source: "find it", script: 'memory find nearest "a"', by: "model" });
  assert.deepEqual(getSettings().plans.slice(0, 2).map((p) => p.id), [b.id, a.id]);
  assert.equal(b.project, "p450");
  const r = addReport({ source: "find it", script: b.script, summary: "0 node(s)" });
  assert.equal(getSettings().reports[0].id, r.id);
  assert.equal(r.project, "p450");
});
