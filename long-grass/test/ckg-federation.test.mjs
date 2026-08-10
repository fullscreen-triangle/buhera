// Confirms the "complete CKG experiment" tutorial's attached modules each fold
// a real fact onto the graph when dispatched via ckg — i.e. they return an
// output_delta for "demo", and the run records it whether or not their linked
// runtime is present. This is the "recorded, not raised" property the tutorial
// leans on.
//
// NOTE: `scope` is exercised in the tutorial too, but it transitively imports a
// vendored .ts compiler that the node test resolver can't load (webpack/next
// handles it in the browser). So the import-level test covers echo /
// shapeshifter / sbs; scope's fold is verified by hand in the browser.
import test from "node:test";
import assert from "node:assert/strict";

import { ckgModule } from "../src/lib/modules/ckg-module.js";
import { register } from "../src/lib/modules/registry.js";
import { echoModule } from "../src/lib/modules/echo-module.js";
import { shapeshifterModule } from "../src/lib/modules/shapeshifter-module.js";
import { sbsModule } from "../src/lib/modules/sbs-module.js";

register(echoModule);
register(shapeshifterModule);
register(sbsModule);

test("the tutorial's five-node build folds a fact from every attached module", async () => {
  await ckgModule.execute({ op: "reset" });
  for (const tau of ["sample", "prep", "measure", "process", "call"]) {
    await ckgModule.execute({ op: "represent", tau, seed: tau === "sample" ? 1 : 0 });
  }
  await ckgModule.execute({ op: "attach", tau: "sample", name: "seen", module: "echo", instruction: "sample logged" });
  await ckgModule.execute({ op: "attach", tau: "measure", name: "spectra", module: "shapeshifter", instruction: "demo" });
  await ckgModule.execute({ op: "attach", tau: "process", name: "pathway", module: "sbs", instruction: "demo" });

  await ckgModule.execute({ op: "carry" });

  const graph = (await ckgModule.execute({ op: "graph" })).output_delta;
  assert.equal(graph.node_count, 5);

  const factPreds = new Set(
    graph.nodes.flatMap((n) => n.facts.map((f) => f.predicate))
  );
  // every attached module left a fact — none was dropped or halted the run.
  // Facts are keyed `fact:<module>#<chunk>` now (so one node can hold several
  // facts from one module), so match on the module prefix, not the whole key.
  const moduleOf = (pred) => pred.slice("fact:".length).split("#")[0];
  const factMods = new Set([...factPreds].filter((p) => p.startsWith("fact:")).map(moduleOf));
  for (const mod of ["echo", "shapeshifter", "sbs"]) {
    assert.ok(
      factMods.has(mod),
      `${mod} should have folded a fact onto the graph`
    );
  }

  const report = (await ckgModule.execute({ op: "report" })).output_delta;
  assert.deepEqual(
    report.contributors,
    ["echo", "sbs", "shapeshifter"],
    "report accounts for every contributing module"
  );

  // The report is a DOSSIER, not a tally: each contribution must carry the
  // module's WHOLE output_delta (the thing <Artifact> re-renders as the real
  // chart) plus an extracted findings headline. This is the property whose
  // absence made the old report "practically a joke".
  const sbsContribs = report.contributions.sbs || [];
  assert.ok(sbsContribs.length >= 1, "sbs contributed at least once");
  const sbsC = sbsContribs[0];
  // full delta survives — same kind the sbs module returns on a direct dispatch
  assert.equal(sbsC.delta.kind, "sbs_result", "sbs's full output_delta is carried, not summarised away");
  assert.ok(sbsC.delta.circuit, "the circuit (drawn by MetricsDashboard) is present on the fact");
  assert.ok(sbsC.delta.metrics, "the S-entropy metrics (R, V) are present on the fact");
  // extracted findings give the report its readable headline + named props
  assert.ok(sbsC.findings, "an extracted findings digest accompanies the delta");
  assert.equal(typeof sbsC.findings.headline, "string");
  const labels = sbsC.findings.props.map((p) => p.label);
  assert.ok(
    labels.includes("coherence R") && labels.includes("flux visibility V"),
    "sbs findings surface the named properties R and V, not just a count"
  );

  const shapeContribs = report.contributions.shapeshifter || [];
  assert.ok(shapeContribs.length >= 1, "shapeshifter contributed");
  assert.equal(
    shapeContribs[0].delta.kind,
    "shapeshifter_run",
    "shapeshifter's full output_delta (workspace + term) is carried"
  );
  assert.ok(
    Array.isArray(shapeContribs[0].delta.workspace),
    "the produced spectra workspace survives on the fact for the renderer to draw"
  );

  // The same rich delta is visible when walking the graph, so a fact on a node
  // can expand into the module's own chart there too.
  const graph2 = (await ckgModule.execute({ op: "graph" })).output_delta;
  const processNode = graph2.nodes.find((n) => n.tau === "process");
  const sbsFact = processNode.facts.find((f) => f.predicate.startsWith("fact:sbs"));
  assert.equal(sbsFact.object.delta.kind, "sbs_result", "the graph view carries the full delta on the node fact");
});
