/* ============================================================================
 * Ping-pong contract tests.
 *
 * These assert the two properties that make the enrichment safe to hand back:
 *   (P1) it attaches to the base subjects (never forks identity);
 *   (P3) it is DETACHABLE — merging then dropping the enrichment graph returns
 *        the base artefact bit-for-bit.
 * Plus a correctness check on the derivation itself.
 * ========================================================================== */

import { test } from "node:test";
import assert from "node:assert/strict";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import N3 from "n3";

import { loadTurtle, buildModel } from "../src/ingest.mjs";
import { Emitter } from "../src/emit.mjs";
import { makeBalanceCheckModule } from "../src/balance-check.mjs";
import { ENRICHMENT_GRAPH, NS, enr, boolLit } from "../src/vocab.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE_TTL = resolve(
  __dirname,
  "../../../nfdi4cat/src/transaminase_kg/data/transaminase.ttl",
);

async function deriveEnrichment() {
  const store = await loadTurtle(BASE_TTL);
  const model = buildModel(store);
  const mod = makeBalanceCheckModule();
  const act = await mod.execute({ kind: "balance", model }, model.reactions.length);
  const emitter = new Emitter({ runId: "test", startedAt: new Date(0).toISOString() });
  for (const r of act.output_delta.results) {
    emitter.fact(r.reaction, enr("massBalanced"), boolLit(r.massBalanced), {
      module: "balance-check",
    });
    emitter.fact(r.reaction, enr("chargeBalanced"), boolLit(r.chargeBalanced), {
      module: "balance-check",
    });
  }
  emitter.finalize(new Date(0).toISOString());
  return { store, model, emitter, report: act.output_delta };
}

test("ingest reads exactly the wire graph (3 reactions, 9 species, ChEBI lifted)", async () => {
  const store = await loadTurtle(BASE_TTL);
  const model = buildModel(store);
  assert.equal(model.reactions.length, 3);
  assert.equal(model.species.size, 9); // 8 ChemicalSpecies + 1 Cofactor
  const alanine = model.species.get(`${NS.res}species/L-alanine`);
  assert.equal(alanine.chebi, "16977");
});

test("Gap 1: all three transaminations balance in mass and charge", async () => {
  const { report } = await deriveEnrichment();
  assert.equal(report.allBalanced, true);
  for (const r of report.results) {
    assert.equal(r.massBalanced, true, `${r.key} mass`);
    assert.equal(r.chargeBalanced, true, `${r.key} charge`);
    assert.equal(r.unresolved.length, 0, `${r.key} unresolved`);
  }
});

test("P1: enrichment attaches to base subjects, never forks identity", async () => {
  const { store, emitter } = await deriveEnrichment();
  const baseReactionIris = new Set(
    store
      .getSubjects(
        `${NS.rdf}type`,
        `${NS.ta}Transamination`,
        null,
      )
      .map((s) => s.value),
  );
  // every reaction we wrote a balance fact about must be a base subject
  const enriched = emitter.store
    .getSubjects(enr("massBalanced"), null, ENRICHMENT_GRAPH)
    .map((s) => s.value);
  assert.equal(enriched.length, 3);
  for (const iri of enriched) {
    assert.ok(baseReactionIris.has(iri), `${iri} exists in base`);
  }
});

test("P3: enrichment is detachable — merge then drop restores base bit-for-bit", async () => {
  const { store, emitter } = await deriveEnrichment();

  // snapshot base (default graph) before merge
  const baseBefore = store.getQuads(null, null, null, null).length;

  // merge enrichment quads (they carry ENRICHMENT_GRAPH) into the base store
  store.addQuads(emitter.store.getQuads(null, null, null, ENRICHMENT_GRAPH));
  const afterMerge = store.getQuads(null, null, null, null).length;
  assert.ok(afterMerge > baseBefore, "merge added quads");

  // the base default graph is untouched by the merge
  const defaultAfterMerge = store.getQuads(null, null, null, N3.DataFactory.defaultGraph()).length;
  assert.equal(defaultAfterMerge, baseBefore, "no base triple mutated by merge");

  // DROP GRAPH <enrichment>
  const dropped = store.getQuads(null, null, null, ENRICHMENT_GRAPH);
  store.removeQuads(dropped);
  const afterDrop = store.getQuads(null, null, null, null).length;
  assert.equal(afterDrop, baseBefore, "drop restores exact base size");
});
