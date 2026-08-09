// Tests for the CKG runtime and module — the specimen the "complete CKG
// experiment" tutorial drives. Exercises the three properties the tutorial
// exhibits: non-judging dispatch (an error is a recorded value, not a halt),
// emergent trajectory (two seeds → two edge sets over the same nodes), and
// protocol-not-results reproducibility (the fingerprint hashes structure, not
// run values).
import test from "node:test";
import assert from "node:assert/strict";

import {
  CkgRuntime,
  makeNode,
  addChunk,
  provenanceFingerprint,
  tauShape,
} from "../src/lib/ckg/runtime.js";
import { ckgModule } from "../src/lib/modules/ckg-module.js";
import { register } from "../src/lib/modules/registry.js";
import { echoModule } from "../src/lib/modules/echo-module.js";

// Make a real module reachable for the module-chunk / attach path.
register(echoModule);

// --- runtime: dispatch runs everything and judges nothing -------------------

test("dispatch runs every chunk and records an error as a value, not a halt", async () => {
  const rt = new CkgRuntime();
  const node = makeNode("t", ["a"]);
  addChunk(node, "good", () => ({ kind: "v", payload: 1, source_chunk: "good" }));
  addChunk(node, "bad", () => { throw new Error("boom"); });
  addChunk(node, "after", () => ({ kind: "w", payload: 2, source_chunk: "after" }));

  const deltas = await rt.dispatch(node);

  // all three ran — the throw did not stop the run
  assert.equal(deltas.length, 3);
  assert.deepEqual(deltas.map((d) => d.kind), ["v", "error", "w"]);
  // the error is on the node as a value, alongside the others
  assert.equal(node.values.v, 1);
  assert.equal(node.values.w, 2);
  assert.match(String(node.values.error), /boom/);
  // audit records all three acts; none is gated on a verdict
  assert.equal(rt.audit.length, 3);
  assert.equal(rt.audit.filter((a) => a.raised).length, 1);
  // the runtime never emitted a verdict
  assert.equal("ok" in deltas[0], false);
});

test("a null-returning chunk is an audited noop that clobbers no channel", async () => {
  const rt = new CkgRuntime();
  const node = makeNode("t");
  addChunk(node, "real", () => ({ kind: "sig", payload: 5, source_chunk: "real" }));
  addChunk(node, "silent", () => null);
  await rt.dispatch(node);
  assert.equal(node.values.sig, 5);
  assert.equal("noop" in node.values, false); // noop is audited, not stored
  assert.equal(rt.audit.find((a) => a.chunk === "silent").emitted_kind, "noop");
});

// --- fingerprint: protocol, not results -------------------------------------

test("fingerprint excludes run values: same protocol → same hash after a run", async () => {
  const a = makeNode("assay", ["ckg", "assay"]);
  addChunk(a, "publish", () => ({ kind: "signal", payload: 1, source_chunk: "p" }));
  const before = provenanceFingerprint([a]);

  const rt = new CkgRuntime();
  await rt.dispatch(a); // produces values
  const after = provenanceFingerprint([a]);

  assert.equal(before, after, "values must not enter the fingerprint");
  // shape is tau + sorted chunk names
  assert.equal(tauShape(a), "assay{publish}");
});

test("fingerprint separates distinct protocols", () => {
  const a = makeNode("x", ["a"]);
  addChunk(a, "c1", () => null);
  const b = makeNode("x", ["a"]);
  addChunk(b, "c2", () => null);
  assert.notEqual(provenanceFingerprint([a]), provenanceFingerprint([b]));
});

// --- module: the tutorial's verbs -------------------------------------------

test("represent → attach(real module) → dispatch folds the module's delta as a fact", async () => {
  await ckgModule.execute({ op: "reset" });
  await ckgModule.execute({ op: "represent", tau: "assay", seed: 1 });
  const at = await ckgModule.execute({
    op: "attach",
    tau: "assay",
    name: "echoed",
    module: "echo",
    instruction: "hello federation",
  });
  assert.equal(at.ok, true);
  assert.ok(at.output_delta.chunks.includes("echoed"));

  await ckgModule.execute({ op: "dispatch", tau: "assay" });

  const graph = await ckgModule.execute({ op: "graph" });
  assert.equal(graph.output_delta.kind, "ckg_graph");
  const assay = graph.output_delta.nodes.find((n) => n.tau === "assay");
  const echoFact = assay.facts.find((f) => f.predicate === "fact:echo");
  assert.ok(echoFact, "echo module's output_delta became a fact on the node");
  assert.equal(echoFact.object.ok, true);
});

test("the report is assembled from emitted values (the account the original lacked)", async () => {
  await ckgModule.execute({ op: "reset" });
  await ckgModule.execute({ op: "represent", tau: "a", seed: 1 });
  await ckgModule.execute({ op: "attach", tau: "a", module: "echo", instruction: "x" });
  await ckgModule.execute({ op: "dispatch", tau: "a" });

  const rep = (await ckgModule.execute({ op: "report" })).output_delta;
  assert.equal(rep.kind, "ckg_report");
  assert.ok(rep.contributors.includes("echo"));
  assert.ok(rep.acts >= 1);
  assert.ok(Array.isArray(rep.audit));
});

test("emergent trajectory: two seeds yield different edge sets over the same nodes", async () => {
  // Build a fresh line of nodes a..e, seed the FRONT, run the carry.
  async function trajectoryWithSeedAt(seedTau, mag) {
    await ckgModule.execute({ op: "reset" });
    for (const tau of ["a", "b", "c", "d", "e"]) {
      await ckgModule.execute({ op: "represent", tau, seed: 0 });
    }
    await ckgModule.execute({ op: "seed", tau: seedTau, value: mag });
    const carry = await ckgModule.execute({ op: "carry" });
    return carry.output_delta.trajectory.slice().sort();
  }

  const front = await trajectoryWithSeedAt("a", 3);
  const mid = await trajectoryWithSeedAt("c", 3);

  // both non-empty, and not identical — seeding a different node reshapes the
  // reads, so the run-induced edge set differs (Thm. 5).
  assert.ok(front.length > 0);
  assert.ok(mid.length > 0);
  assert.notDeepEqual(front, mid);
});

test("fingerprint verb is stable across two runs of the same protocol", async () => {
  async function buildAndPrint() {
    await ckgModule.execute({ op: "reset" });
    await ckgModule.execute({ op: "represent", tau: "a", seed: 1 });
    await ckgModule.execute({ op: "attach", tau: "a", module: "echo", instruction: "x" });
    await ckgModule.execute({ op: "carry" }); // run it — produces values
    return (await ckgModule.execute({ op: "fingerprint" })).output_delta.fingerprint;
  }
  const f1 = await buildAndPrint();
  const f2 = await buildAndPrint();
  assert.equal(f1, f2, "same protocol reproduces the same fingerprint despite differing run values");
});
