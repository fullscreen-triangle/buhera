// The smith module over musande's canonical Agent Smith engine
// (vendor/agent-smith). Replaces the tests of the former regex-stub port:
// every claim is now checked against the language's real front end.
import test from "node:test";
import assert from "node:assert/strict";

import { smithModule } from "../src/lib/modules/smith-module.js";

const SCOUT = `agent Scout {
  purpose minimise backlog
  scenes {
    scene watch serves backlog with look_hook
    scene report serves backlog with tell_hook
  }
  self {
    parts { eyes, legs, voice }
    separations { (eyes, legs: 3), (legs, voice: 2), (voice, eyes: 4) }
  }
  budget 2.0
  floor 2.0
}`;

test("check: a typed agent carries χ, realised floor and a two-sided partition", async () => {
  const r = await smithModule.execute(SCOUT);
  assert.equal(r.ok, true);
  const a = r.output_delta.agents[0];
  assert.equal(a.name, "Scout");
  assert.equal(a.regime, "character", "purpose minimise → character regime");
  assert.equal(a.chi, 5, "cheapest bipartition of the triangle: {legs} | {eyes, voice}, cut 3 + 2");
  assert.equal(a.chiPartition.length, 2);
  assert.equal(r.residue, 0);
  assert.equal(r.completed, true);
});

test("check: the one-line canonical syntax parses (the terminal route)", async () => {
  const r = await smithModule.execute("agent Ok { purpose reach done scenes { scene s serves done with h } self { parts { p, q } separations { (p, q: 3) } } budget 1 floor 2 }");
  assert.equal(r.ok, true);
  assert.equal(r.output_delta.agents[0].regime, "task", "purpose reach → task-agent");
});

test("check: the real typechecker refuses below-floor costs and off-target scenes", async () => {
  const below = await smithModule.execute(SCOUT.replace("(legs, voice: 2)", "(legs, voice: 1)"));
  assert.equal(below.ok, false);
  const off = await smithModule.execute(SCOUT.replace("scene report serves backlog", "scene report serves other"));
  assert.equal(off.ok, false);
  assert.ok(off.residue >= 1, "check-only residue counts the diagnostics");
});

test("check: society members are typed individually", async () => {
  const src = `society pair {
    agent A { purpose minimise forge_residual scenes { scene s serves forge_residual with h } self { parts { a1, a2 } separations { (a1, a2: 3) } } budget 1 floor 2 }
    agent B { purpose minimise forge_residual scenes { scene t serves forge_residual with h } self { parts { b1, b2 } separations { (b1, b2: 2) } } budget 1 floor 2 }
    tie (A, B: 2)
    couple 3
  }`;
  const r = await smithModule.execute(src);
  assert.equal(r.ok, true);
  assert.deepEqual(r.output_delta.agents.map((a) => a.name), ["A", "B"]);
  assert.ok(r.output_delta.society);
});

test("run: deterministic, models off, trace flattened for the renderer", async () => {
  const a = await smithModule.execute({ source: SCOUT, run: true, maxTicks: 8 });
  const b = await smithModule.execute({ source: SCOUT, run: true, maxTicks: 8 });
  assert.deepEqual(a.output_delta.steps, b.output_delta.steps);
  assert.ok(a.output_delta.steps.length >= 1);
  assert.ok(a.output_delta.steps.every((s) => typeof s.tick === "number" && s.agent === "Scout"));
  assert.ok(a.output_delta.finalCounts.Scout >= 1);
  assert.ok(a.output_delta.steps.every((s) => s.model == null));
});

test("a refused program does not run", async () => {
  const r = await smithModule.execute({ source: "agent x {", run: true });
  assert.equal(r.ok, false);
  assert.equal(r.output_delta.steps, undefined);
});
