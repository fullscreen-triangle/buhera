// Unit tests for the smith agent-generation module and its compiler.
// Runs against the REAL ported compiler (the web-side twin of the Rust agent
// tool), so this doubles as a conformance check for the DSL and the split-
// attention mathematics (χ, realised floor, deterministic tick loop).
import test from "node:test";
import assert from "node:assert/strict";

import { compile, run, parse } from "../src/lib/smith/compiler.js";
import { smithModule } from "../src/lib/modules/smith-module.js";

const SCOUT = `
agent Scout {
  purpose minimise Threat;
  scene watch serves Threat with radar;
  scene listen serves Threat with sonar;
  self { parts { eye, ear, mind } separations (eye,ear: 3) (ear,mind: 2) (eye,mind: 2) }
  budget 4; floor 2;
  coherence keeps { eye, mind }
}
`;

test("valid multiline agent checks, with χ / floor / partition", () => {
  const c = compile(SCOUT);
  assert.equal(c.check.ok, true);
  assert.deepEqual(c.check.errors, []);
  assert.equal(c.check.agents.length, 1);
  const a = c.check.agents[0];
  assert.equal(a.name, "Scout");
  assert.equal(a.regime, "character"); // purpose minimise → character regime
  assert.equal(Number.isFinite(a.chi), true);
  assert.equal(Number.isFinite(a.floor), true);
  assert.ok(a.chiPartition.length >= 2); // χ is over ≥2-block partitions
});

test("single-line agent parses (the webtool terminal route)", () => {
  const c = compile(
    `agent Ok { purpose minimise T; scene s serves T with h; self { parts { a, b } separations (a,b: 3) } budget 2; floor 2 }`
  );
  assert.equal(c.check.ok, true);
  assert.equal(c.check.agents[0].name, "Ok");
});

test("checker rejects below-floor separations and off-target scenes", () => {
  const c = compile(
    `agent Bad { purpose reach Goal; scene s serves Other with h; self { parts { a, b } separations (a,b: 1) } budget 2; floor 2 }`
  );
  assert.equal(c.check.ok, false);
  const msgs = c.check.errors.map((e) => e.message).join(" | ");
  assert.match(msgs, /below the floor/);
  assert.match(msgs, /serves "Other"/);
});

test("society extracts inner agents despite nested self-braces", () => {
  const c = compile(
    `society Team {
       agent A { purpose minimise T; scene s serves T with h; self { parts { p, q } separations (p,q: 3) } budget 3; floor 2 }
       agent B { purpose minimise T; scene s2 serves T with h2; self { parts { r, w } separations (r,w: 3) } budget 3; floor 2 }
       tie(A,B: 3);
       couple 1
     }`
  );
  assert.equal(c.check.ok, true);
  assert.deepEqual(c.check.agents.map((a) => a.name), ["A", "B"]);
});

test("run is deterministic across compiles (no Math.random)", () => {
  const a = compile(SCOUT);
  const b = compile(SCOUT);
  const ra = run(a.file, 12);
  const rb = run(b.file, 12);
  assert.deepEqual(ra.finalCounts, rb.finalCounts);
  assert.deepEqual(ra.steps, rb.steps);
});

test("run alternates construction/commitment phases", () => {
  const c = compile(SCOUT);
  const r = run(c.file, 9);
  // tick % 3 === 0 is construction (observe); others are commitment.
  const t3 = r.steps.find((s) => s.tick === 3);
  const t1 = r.steps.find((s) => s.tick === 1);
  assert.equal(t3.phase, "construction");
  assert.equal(t3.outcome, "observe");
  assert.equal(t1.phase, "commitment");
});

test("empty source yields no agent/society declaration error", () => {
  const { file, errors } = parse("   ");
  assert.equal(file.items.length, 0);
  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /No agent or society declarations/);
});

test("module: string instruction checks and returns agent_generated", async () => {
  const res = await smithModule.execute(SCOUT);
  assert.equal(res.ok, true);
  assert.equal(res.output_delta.kind, "agent_generated");
  assert.equal(res.output_delta.agents[0].name, "Scout");
  assert.equal(res.completed, true);
  // residue mirrors aggregate realised floor (finite, ≥ 0)
  assert.equal(typeof res.residue, "number");
  assert.ok(res.residue >= 0);
});

test("module: { source, run:true } attaches a run trace", async () => {
  const res = await smithModule.execute({ source: SCOUT, run: true, maxTicks: 9 });
  assert.equal(res.ok, true);
  assert.ok(Array.isArray(res.output_delta.steps));
  assert.equal(res.output_delta.steps.length, 9);
  assert.ok(res.output_delta.finalCounts.Scout >= 1);
});

test("module: empty source is a clean error, not a throw", async () => {
  const res = await smithModule.execute("");
  assert.equal(res.ok, false);
  assert.equal(res.output_delta.kind, "agent_generated");
  assert.equal(res.error, "no-source");
});

test("module: rejected check does not run the tick loop", async () => {
  const res = await smithModule.execute({
    source: `agent Bad { purpose reach Goal; scene s serves Other with h; self { parts { a, b } separations (a,b: 1) } budget 2; floor 2 }`,
    run: true,
  });
  assert.equal(res.ok, false);
  assert.equal(res.output_delta.steps, undefined); // no trace when check failed
});
