// Unit tests for the hfq and ladder Buhera modules — thin wrappers around the
// REAL vendored engines (@hegel/hfq, @levinthal/ladder), not reimplementations.
// These tests exercise the actual query methods: Mark Doerr's real biocatalysis
// questions, the six-verdict system, and the contact-graph ladder construction.
import test from "node:test";
import assert from "node:assert/strict";

import { hfqModule } from "../src/lib/modules/hfq-module.js";
import { ladderModule } from "../src/lib/modules/ladder-module.js";

test("hfq: demo plan (the paper's worked example) answers every step", async () => {
  const res = await hfqModule.execute("demo");
  assert.equal(res.ok, true);
  assert.equal(res.output_delta.result.ok, true);
  const verdicts = res.output_delta.result.steps.map((s) => s.verdict);
  assert.deepEqual(verdicts, ["answer", "answer", "answer"]);
});

test("hfq: list_presets surfaces all 24 real plans, including Mark Doerr's five", async () => {
  const res = await hfqModule.execute({ kind: "list_presets" });
  assert.equal(res.ok, true);
  assert.equal(res.output_delta.plans.length, 24);
  const ids = res.output_delta.plans.map((p) => p.id);
  for (const id of ["mark_q1", "mark_q2", "mark_q3", "mark_q4", "mark_q5"]) {
    assert.ok(ids.includes(id), `missing ${id}`);
  }
  for (let i = 1; i <= 8; i++) {
    assert.ok(ids.includes(`dcat_g${i}`), `missing dcat_g${i}`);
  }
});

test("hfq: preset mark_q1 (bacterial transaminase, no cysteine) runs the real biocat world", async () => {
  const res = await hfqModule.execute({ kind: "preset", id: "mark_q1" });
  assert.equal(res.ok, true);
  assert.equal(res.output_delta.result.world, "biocat");
  assert.equal(res.output_delta.result.ok, true);
});

test("hfq: an ill-capability plan is refused statically, before any request", async () => {
  const res = await hfqModule.execute({
    kind: "run",
    source: `plan bad {
  budget 10 requests
  let x = from enzdb ask reactions_consuming("EC:1.1.1.1")
  emit x
}`,
  });
  // enzdb declares {lookup, link}, not pattern — this must refuse at the
  // check stage, with zero requests issued (cor:refuse-before-contact).
  assert.equal(res.output_delta.result.halted_early, true);
  assert.equal(res.output_delta.result.requests_issued, 0);
  assert.equal(res.output_delta.result.steps[0].verdict, "surface");
});

test("hfq: unknown preset id returns a clean error, not a throw", async () => {
  const res = await hfqModule.execute({ kind: "preset", id: "does_not_exist" });
  assert.equal(res.ok, false);
});

test("ladder: derive accepts a plain-object graph (as sent over dispatch/JSON)", async () => {
  const res = await ladderModule.execute({
    op: "derive",
    graph: {
      vertices: ["m", "v0", "v1", "v2"],
      weights: { "v0|m": 1.0, "v1|m": 1.0, "v2|m": 1.0, "v0|v1": 2.0, "v1|v2": 0.3 },
      medium: "m",
    },
    vertex: "v0",
    radius: 1,
  });
  assert.equal(res.ok, true);
  assert.equal(res.output_delta.kind, "ladder_derived");
  assert.ok(res.output_delta.intensive >= 0 && res.output_delta.intensive <= 1);
});

test("ladder: demo derives real intensive power over a chain graph", async () => {
  const res = await ladderModule.execute("demo");
  assert.equal(res.ok, true);
  assert.equal(res.output_delta.powers.length, 6);
  for (const { power } of res.output_delta.powers) {
    assert.ok(power >= 0 && power <= 1);
  }
});

test("ladder: compose multiplicative matches the levinthal notebook's cell 9 value", async () => {
  const res = await ladderModule.execute({ op: "compose", powers: [0.45, 0.30, 0.55] });
  assert.equal(res.ok, true);
  // 1 - (0.55)(0.70)(0.45) = 0.82675, the exact figure cells.js CELLS.federated documents.
  assert.ok(Math.abs(res.output_delta.multiplicative - 0.82675) < 1e-9);
});

test("ladder: climb refuses (subfloor) before any commitment when target is unreachable", async () => {
  const res = await ladderModule.execute({
    op: "climb",
    powers: [0.45, 0.30, 0.55],
    target: 0.95,
  });
  assert.equal(res.ok, true);
  assert.equal(res.output_delta.verdict, "subfloor");
  assert.equal(res.output_delta.M, 0);
});

test("ladder: climb reaches and commits once per rung when target is reachable", async () => {
  const res = await ladderModule.execute({
    op: "climb",
    powers: [0.45, 0.30, 0.55],
    target: 0.70,
  });
  assert.equal(res.ok, true);
  assert.equal(res.output_delta.verdict, "reached");
  assert.equal(res.output_delta.M, 3);
});

test("modules conform to the registry's Module trait", () => {
  // bootstrap.js pulls in the whole federation (including graffiti, which the
  // Node test resolver can't directory-import) — test the trait shape
  // directly instead of going through bootstrapFederation().
  for (const mod of [hfqModule, ladderModule]) {
    assert.equal(typeof mod.id, "string");
    assert.equal(typeof mod.execute, "function");
    assert.equal(typeof mod.describe, "function");
    const desc = mod.describe();
    assert.equal(desc.id, mod.id);
    assert.ok(Array.isArray(desc.instructions) && desc.instructions.length > 0);
  }
});
