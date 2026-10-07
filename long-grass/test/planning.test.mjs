// Tests for the planning store and the work verbs typed on the blank screen.
import test from "node:test";
import assert from "node:assert/strict";

import { EXPERIMENT_STEPS, addItem, addJob, addRef, addStep, getItems, itemsInProject, matchItems, removeItem, toMarkdown, toggleStep, updateItem } from "../src/lib/surface/planning.js";
import { workVerb } from "../src/lib/surface/verbs.js";

test("an experiment starts with the usual steps; a task with none", () => {
  const e = addItem({ title: "PC 34:1 lipid series", kind: "experiment" });
  const t = addItem({ title: "book the 4090" });
  assert.equal(e.steps.length, EXPERIMENT_STEPS.length);
  assert.equal(e.status, "idea");
  assert.equal(t.kind, "task");
  assert.equal(t.steps.length, 0);
  removeItem(e.id); removeItem(t.id);
});

test("steps, refs and jobs", () => {
  const i = addItem({ title: "lipids" });
  addStep(i.id, "plate layout");
  const s = getItems().find((x) => x.id === i.id).steps[0];
  toggleStep(i.id, s.id);
  addRef(i.id, { source: "mail", cite: "mail:uni/INBOX/2", title: "Sample prep protocol", verdict: null });
  addRef(i.id, { source: "mail", cite: "mail:uni/INBOX/2", title: "again" });
  addJob(i.id, { repo: "/r", unit: "score" });
  const now = getItems().find((x) => x.id === i.id);
  assert.equal(now.steps[0].done, true);
  assert.equal(now.refs.length, 1, "the same citation is kept once");
  assert.equal(now.jobs.length, 1);
  assert.equal(now.status, "running", "a job makes an idea running");
  removeItem(i.id);
});

test("closed items sort after open ones, and find matches every word", () => {
  const a = addItem({ title: "alpha lipid run" });
  const b = addItem({ title: "beta lipid run" });
  updateItem(b.id, { status: "done" });
  const order = itemsInProject().map((x) => x.id);
  assert.ok(order.indexOf(a.id) < order.indexOf(b.id));
  assert.deepEqual(matchItems("lipid alpha").map((x) => x.id), [a.id]);
  assert.deepEqual(matchItems(""), []);
  removeItem(a.id); removeItem(b.id);
});

test("an item as Markdown", () => {
  const i = addItem({ title: "lipids", kind: "experiment" });
  addRef(i.id, { source: "files", cite: "notes/plate.md:3-7", title: "plate layout", verdict: "covered", snippet: "A1 blank" });
  const md = toMarkdown(getItems().find((x) => x.id === i.id));
  assert.match(md, /^# lipids\n/);
  assert.match(md, /- \[ \] state the question/);
  assert.match(md, /`notes\/plate.md:3-7` \(search verdict: covered\)/);
  removeItem(i.id);
});

test("the work verbs", () => {
  assert.deepEqual(workVerb("mail from:anna protocol"), { module: "mail", instruction: { kind: "search", query: "from:anna protocol" } });
  assert.deepEqual(workVerb("inbox"), { module: "mail", instruction: { kind: "search", query: "" } });
  assert.equal(workVerb("mail").instruction, "accounts");
  assert.deepEqual(workVerb("find plate layout"), { module: "planning", instruction: { kind: "find", query: "plate layout" } });
  assert.deepEqual(workVerb("plan experiment PC 34:1 series"), { module: "planning", instruction: { kind: "new", type: "experiment", title: "PC 34:1 series" } });
  assert.deepEqual(workVerb("plan an experiment: lipids").instruction, { kind: "new", type: "experiment", title: "lipids" });
  assert.deepEqual(workVerb("plan book the room").instruction, { kind: "new", type: "task", title: "book the room" });
  assert.equal(workVerb("plans").instruction, "board");
  assert.equal(workVerb("apphub").module, "lattice");
  assert.equal(workVerb("describe sample with \"x\"\nspawn lavoisier from sample"), null, "a script is not a verb");
  assert.equal(workVerb("finding nemo"), null);
  assert.equal(workVerb("mailbox store"), null);
});
