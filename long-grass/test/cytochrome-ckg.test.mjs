// The cytochrome P450 monograph, driven as a CKG runtime trajectory — in full.
//
// The levinthal transaminase model was a *medium-vertex CKG solution* that drew
// ONE distinction (participant vs carrier) from TWO kinds of fact. The point of
// this extension is that a richer corpus makes a richer graph: MORE fact kinds
// meeting on MORE, DENSER vertices. So this suite does not sample the corpus —
// it exercises EVERY fact family the monograph carries, with the monograph's own
// numbers, and then builds a trajectory in which three federation modules
// (cytochrome + sbs + shapeshifter, each with its own DSL) fold their real
// output onto a shared P450 catalytic cycle, several facts to a stage.
//
// Nothing here is authored into a schema. The stages are represented as nodes,
// contributors are attached, the carry runs, and the trajectory is read back.
import test from "node:test";
import assert from "node:assert/strict";

import { ckgModule } from "../src/lib/modules/ckg-module.js";
import { register } from "../src/lib/modules/registry.js";
import {
  cytochromeModule,
  P450_ET_SBS,
  P450_MS_SS,
} from "../src/lib/modules/cytochrome-module.js";
import { sbsModule } from "../src/lib/modules/sbs-module.js";
import { shapeshifterModule } from "../src/lib/modules/shapeshifter-module.js";

register(cytochromeModule);
register(sbsModule);
register(shapeshifterModule);

// Every fact family the monograph carries, each dispatched directly, checked
// against the monograph's own value. A model uses facts — here are all of them.
test("the cytochrome module folds EVERY monograph fact family, with the monograph's numbers", async () => {
  const cyp = (instr) => cytochromeModule.execute(instr).then((r) => r.output_delta);

  // 1 — electron-transfer chain: 3 hops, d_C=4, λ=0.85 eV, FMN→heme limits
  const et = await cyp({ op: "electron-transfer" });
  assert.equal(et.kind, "cyp_electron_transfer");
  assert.equal(et.chain.length, 3, "NADPH→FAD→FMN→heme is three hops");
  assert.equal(et.d_C, 4);
  assert.equal(et.marcus_lambda_eV, 0.85);
  assert.equal(et.rate_limiting, "FMN→heme");

  // 2 — Compound I: d_C=1 aperture of depth ΔM=ln2, PCET, KIE≈1.7
  const ci = await cyp({ op: "compound-i" });
  assert.equal(ci.kind, "cyp_compound_i");
  assert.ok(Math.abs(ci.dM - Math.log(2)) < 1e-3, "ΔM = ln2");
  assert.match(ci.mechanism, /PCET/);
  assert.equal(ci.kie, 1.7);

  // 3 — the seven-state closed orbit, ΣΔM ≈ 4.963
  const orbit = await cyp({ op: "states" });
  assert.equal(orbit.kind, "cyp_states");
  assert.equal(orbit.states.length, 7, "seven catalytic states");
  assert.equal(orbit.closed, true, "the orbit closes (state 7 → state 1)");
  assert.equal(orbit.orbit_sum_dM_reference, 4.963);

  // 4 — reaction pathways: ALL SEVEN families, each its own fact, and the
  //     monograph's ΔM ordering S-ox < N-ox < N-dealk < O-dealk < aliphatic.
  const families = [
    "s-oxidation", "n-oxidation", "n-dealkylation", "o-dealkylation",
    "aliphatic-hydroxylation", "aromatic-hydroxylation", "desaturation",
  ];
  const dMs = [];
  for (const rxn of families) {
    const p = await cyp({ op: "pathway", reaction: rxn });
    assert.equal(p.kind, "cyp_pathway", `${rxn} folds a pathway fact`);
    assert.equal(typeof p.dM, "number", `${rxn} carries an aperture ΔM`);
    dMs.push([rxn, p.dM, p.kie]);
  }
  const order = ["s-oxidation", "n-oxidation", "n-dealkylation", "o-dealkylation", "aliphatic-hydroxylation"];
  const byName = Object.fromEntries(dMs.map(([n, d]) => [n, d]));
  for (let i = 1; i < order.length; i++) {
    assert.ok(
      byName[order[i]] > byName[order[i - 1]],
      `aperture ordering: ${order[i]} (ΔM ${byName[order[i]]}) > ${order[i - 1]} (ΔM ${byName[order[i - 1]]})`
    );
  }
  // KIE diagnostic: HAT reactions show a large KIE; direct O-transfer does not.
  const sox = dMs.find(([n]) => n === "s-oxidation");
  const chx = dMs.find(([n]) => n === "aliphatic-hydroxylation");
  assert.ok(sox[2] < 2, "S-oxidation is direct O-transfer (KIE≈1)");
  assert.ok(chx[2] > 2, "aliphatic hydroxylation is HAT (large KIE)");

  // 5 — spectroscopy, now THREE independent observables, each its own fact.
  const soret = await cyp({ op: "soret" });
  assert.equal(soret.kind, "cyp_soret");
  assert.equal(soret.resting_nm, 417);
  assert.equal(soret.compound_i_nm, 392);
  const epr = await cyp({ op: "epr" });
  assert.equal(epr.kind, "cyp_epr");
  assert.equal(epr.g_low, 2.42);
  const raman = await cyp({ op: "raman" });
  assert.equal(raman.kind, "cyp_raman");
  assert.equal(raman.fe_o_cm, 795);
  assert.equal(raman.fe_o_18O_cm, 758, "¹⁸O shifts the Fe=O stretch to 758 cm⁻¹");

  // 6 — isoform / pharmacogenomic ΔM: every CYP2D6 phenotype class, and the
  //     monograph ordering UM < EM < IM < PM; plus the CYP2C9*3 LOF allele.
  const phenos = ["UM", "EM", "IM", "PM"];
  const pdM = [];
  for (const ph of phenos) {
    const iso = await cyp({ op: "isoform", cyp: "CYP2D6", phenotype: ph });
    assert.equal(iso.kind, "cyp_isoform");
    assert.equal(typeof iso.dM, "number", `CYP2D6 ${ph} has a ΔM`);
    pdM.push(iso.dM);
  }
  for (let i = 1; i < pdM.length; i++) {
    assert.ok(pdM[i] > pdM[i - 1], `pharmacogenomic ordering ${phenos[i]} > ${phenos[i - 1]}`);
  }
  const c9 = await cyp({ op: "isoform", cyp: "CYP2C9", phenotype: "*3" });
  assert.equal(c9.dM, 3.6, "CYP2C9*3 loss-of-function ΔM = 3.60");

  // 7 — participant / carrier cut (the transaminase twin): heme Fe is a carrier.
  const parts = await cyp({ op: "participants", reaction: "aliphatic-hydroxylation" });
  assert.equal(parts.kind, "cyp_participants");
  assert.ok(parts.carriers.some((c) => c.includes("heme Fe")), "heme Fe is a carrier");
  assert.ok(!parts.participants.some((p) => p.includes("heme")), "heme Fe is in no equation");
  assert.ok(parts.events.every((e) => e.residue >= parts.floor - 1e-15), "no cut below the floor");

  // 8 — the conditioned floor β: at depth 9 the conversion term dominates.
  const floor = await cyp({ op: "floor" });
  assert.equal(floor.kind, "cyp_floor");
  assert.equal(floor.dominant_term, "conv", "β is conversion-dominated at depth 9");
  assert.ok(floor.beta > 0);
});

// The two OTHER federation DSLs, given P450-specific scripts, produce real
// artifacts — proving the corpus is expressible across the whole federation,
// not just the native cytochrome module.
test("the P450 corpus runs through the sbs and shapeshifter DSLs, producing real artifacts", async () => {
  const sbs = (await sbsModule.execute(P450_ET_SBS)).output_delta;
  assert.equal(sbs.kind, "sbs_result");
  assert.equal(sbs.circuit.numNodes, 4, "NADPH, FAD, FMN, heme");
  assert.equal(sbs.circuit.numEdges, 3, "the three redox hops");
  assert.equal(typeof sbs.metrics.R, "number", "coherence R computed by the solver");
  assert.equal(typeof sbs.metrics.V, "number", "flux visibility V computed by the solver");
  // backward navigation walks the redox ladder from heme back to NADPH
  assert.equal(sbs.navigation[0].name, "heme");
  assert.equal(sbs.navigation[sbs.navigation.length - 1].name, "NADPH");

  const ss = (await shapeshifterModule.execute(P450_MS_SS)).output_delta;
  assert.equal(ss.kind, "shapeshifter_run");
  assert.equal(ss.ok, true, "the .ss script compiles and runs");
  assert.ok(ss.workspace.length >= 2, "the acquisition produces a records + field workspace");
});

// The dense trajectory: three modules fold onto ONE P450 catalytic cycle, and
// several facts meet on shared stages. This is the medium-vertex graph made
// dense — the extension's whole claim, made checkable.
test("the P450 cycle is a DENSE ckg trajectory: three modules, many fact kinds, meeting on shared stages", async () => {
  await ckgModule.execute({ op: "reset" });

  const stages = ["resting", "substrate-bound", "reduction", "oxygen-binding", "compound-i", "oxidation", "product-release"];
  for (const tau of stages) {
    await ckgModule.execute({ op: "represent", tau, seed: tau === "resting" ? 2 : 0 });
  }

  // Attach many facts, DELIBERATELY several to some stages — the channel fix
  // (fact:<module>#<chunk>) lets them coexist, so these no longer clobber.
  const attach = (tau, name, module, instruction) =>
    ckgModule.execute({ op: "attach", tau, name, module, instruction });

  // resting: the closed orbit + the resting-state EPR fingerprint (TWO facts)
  await attach("resting", "orbit", "cytochrome", { op: "states" });
  await attach("resting", "epr", "cytochrome", { op: "epr" });

  // reduction: the native ET fact AND the sbs redox circuit for the SAME chain
  // (TWO facts, TWO modules, one stage)
  await attach("reduction", "et", "cytochrome", { op: "electron-transfer" });
  await attach("reduction", "redox-circuit", "sbs", P450_ET_SBS);

  // oxygen-binding: the Soret shift
  await attach("oxygen-binding", "soret", "cytochrome", { op: "soret" });

  // compound-i: the Compound-I chemistry, the Fe=O Raman confirmation, AND the
  // shapeshifter mass-spec acquisition (THREE facts, TWO modules, one stage)
  await attach("compound-i", "cpdI", "cytochrome", { op: "compound-i" });
  await attach("compound-i", "raman", "cytochrome", { op: "raman" });
  await attach("compound-i", "ms", "shapeshifter", P450_MS_SS);

  // oxidation: the reaction pathway aperture
  await attach("oxidation", "pathway", "cytochrome", { op: "pathway", reaction: "aliphatic-hydroxylation" });

  // product-release: the participant/carrier cut + the pharmacogenomic ΔM (TWO facts)
  await attach("product-release", "parts", "cytochrome", { op: "participants", reaction: "aliphatic-hydroxylation" });
  await attach("product-release", "pgx", "cytochrome", { op: "isoform", cyp: "CYP2D6", phenotype: "PM" });

  await ckgModule.execute({ op: "carry" });

  const graph = (await ckgModule.execute({ op: "graph" })).output_delta;
  assert.equal(graph.node_count, 7, "seven catalytic stages");

  // DENSITY 1 — the whole graph holds far more than the transaminase's two facts.
  assert.ok(graph.fact_count >= 11, `dense graph: ≥11 facts folded (got ${graph.fact_count})`);

  // DENSITY 2 — the fact kinds are DIVERSE, not one repeated. Count distinct
  // delta kinds across every fact on the graph.
  const kinds = new Set();
  for (const n of graph.nodes) {
    for (const f of n.facts) {
      const d = f.object && f.object.delta;
      if (d && d.kind) kinds.add(d.kind);
    }
  }
  assert.ok(kinds.size >= 9, `≥9 distinct fact kinds meet on the graph (got ${kinds.size}: ${[...kinds].join(", ")})`);

  // DENSITY 3 — shared stages: at least two stages each carry MULTIPLE facts.
  const factsAt = (tau) => graph.nodes.find((n) => n.tau === tau).facts.length;
  assert.ok(factsAt("compound-i") >= 3, `compound-i is a meeting ground (≥3 facts, got ${factsAt("compound-i")})`);
  assert.ok(factsAt("reduction") >= 2, `reduction holds ET + sbs circuit (≥2 facts, got ${factsAt("reduction")})`);
  assert.ok(factsAt("resting") >= 2, `resting holds orbit + EPR (≥2 facts, got ${factsAt("resting")})`);

  // DENSITY 4 — THREE modules contributed, not one.
  const report = (await ckgModule.execute({ op: "report" })).output_delta;
  for (const mod of ["cytochrome", "sbs", "shapeshifter"]) {
    assert.ok(report.contributors.includes(mod), `${mod} contributed to the graph`);
  }

  // the sbs contribution carries its WHOLE real delta (R, V, circuit) onto the
  // reduction node — the same object its MetricsDashboard renders.
  const sbsC = (report.contributions.sbs || [])[0];
  assert.equal(sbsC.delta.kind, "sbs_result");
  assert.ok(sbsC.delta.metrics && typeof sbsC.delta.metrics.R === "number", "sbs R survives onto the P450 graph");

  // the shapeshifter contribution carries its produced spectra workspace.
  const ssC = (report.contributions.shapeshifter || [])[0];
  assert.equal(ssC.delta.kind, "shapeshifter_run");
  assert.ok(Array.isArray(ssC.delta.workspace) && ssC.delta.workspace.length >= 2, "shapeshifter workspace survives");

  // the cytochrome contributions span the corpus: many distinct cyp_* kinds.
  const cypKinds = new Set((report.contributions.cytochrome || []).map((c) => c.delta && c.delta.kind));
  assert.ok(cypKinds.size >= 7, `cytochrome contributed ≥7 distinct fact kinds (got ${cypKinds.size})`);

  // the transaminase invariant survived the port: heme Fe reported as a carrier.
  const partContrib = (report.contributions.cytochrome || []).find(
    (c) => c.delta && c.delta.kind === "cyp_participants"
  );
  assert.match(partContrib.findings.headline, /carrier/);

  // protocol reproducibility: the fingerprint hashes the seven stages + bags.
  const fp = (await ckgModule.execute({ op: "fingerprint" })).output_delta;
  assert.equal(fp.nodes, 7);
  assert.equal(typeof fp.fingerprint, "string");
});
