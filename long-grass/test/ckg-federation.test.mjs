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
  // every attached module left a fact — none was dropped or halted the run
  for (const mod of ["echo", "shapeshifter", "sbs"]) {
    assert.ok(
      factPreds.has(`fact:${mod}`),
      `${mod} should have folded a fact:${mod} onto the graph`
    );
  }

  const report = (await ckgModule.execute({ op: "report" })).output_delta;
  assert.deepEqual(
    report.contributors,
    ["echo", "sbs", "shapeshifter"],
    "report accounts for every contributing module"
  );
});
