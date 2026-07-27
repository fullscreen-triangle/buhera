#!/usr/bin/env node
/* ============================================================================
 * buhera-enrich — the ping-pong orchestrator (seam 1).
 *
 *   1. RECEIVE : parse the partner pipeline's Turtle into a queryable model.
 *   2. DERIVE  : dispatch the ingest model through the Buhera module
 *                federation (Gap 1 = balance-check today).
 *   3. SEND    : serialise the derived facts as a detachable enrichment graph
 *                and write it back over the wire.
 *
 * Neither pipeline is rewritten to accommodate the other. RDF is the only
 * contract. The partner optionally merges the enrichment named graph; DROP
 * GRAPH returns its artefact untouched.
 *
 * Usage:
 *   node bin/enrich.mjs [--in <base.ttl>] [--out <enrichment.ttl>] [--trig]
 * ========================================================================== */

import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

import { register, dispatch, getAuditLog } from "../../src/lib/modules/registry.js";
import { loadTurtle, buildModel } from "../src/ingest.mjs";
import { Emitter } from "../src/emit.mjs";
import { makeBalanceCheckModule } from "../src/balance-check.mjs";
import { enr, decLit, intLit, boolLit, literal, iri } from "../src/vocab.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ENRICH_ROOT = resolve(__dirname, "..");
const REPO_ROOT = resolve(ENRICH_ROOT, "..", "..");

const DEFAULT_IN = resolve(
  REPO_ROOT,
  "nfdi4cat/src/transaminase_kg/data/transaminase.ttl",
);
const DEFAULT_OUT = resolve(ENRICH_ROOT, "out/enrichment.ttl");

function parseArgs(argv) {
  const args = { in: DEFAULT_IN, out: DEFAULT_OUT, trig: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--in") args.in = resolve(argv[++i]);
    else if (a === "--out") args.out = resolve(argv[++i]);
    else if (a === "--trig") args.trig = true;
  }
  return args;
}

/**
 * Deterministic run id without Date.now()/random (both unavailable in some
 * sandboxes and undesirable for reproducibility): derive from input digest.
 */
function runIdFor(model) {
  const n = model.reactions.length + model.species.size;
  return `r${n}-${model.reactions.map((r) => r.key).join("-")}`.slice(0, 48);
}

/** Turn a balance_report ActResult into PROV-stamped enrichment facts. */
function emitBalance(emitter, report) {
  for (const r of report.results) {
    const subj = r.reaction;
    const mod = "balance-check";

    emitter.fact(subj, enr("massBalanced"), boolLit(r.massBalanced), {
      module: mod,
    });
    emitter.fact(subj, enr("chargeBalanced"), boolLit(r.chargeBalanced), {
      module: mod,
    });
    emitter.fact(subj, enr("chargeResidual"), intLit(r.chargeResidual), {
      module: mod,
    });
    // mass residual reified so it carries its provenance explicitly
    emitter.reifiedFact(
      subj,
      enr("massResidualDa"),
      decLit(r.massResidual.toFixed(6)),
      { module: mod },
    );
    emitter.fact(
      subj,
      enr("substrateFormula"),
      literal(r.substrateFormula),
      { module: mod },
    );
    emitter.fact(subj, enr("productFormula"), literal(r.productFormula), {
      module: mod,
    });
    if (r.unresolved.length) {
      for (const u of r.unresolved) {
        emitter.fact(subj, enr("unresolvedSpecies"), iri(u), { module: mod });
      }
    }
  }
}

async function main() {
  const args = parseArgs(process.argv);

  // 1. RECEIVE
  const store = await loadTurtle(args.in);
  const model = buildModel(store);
  console.error(
    `[receive] ${model.reactions.length} reactions, ${model.species.size} species from ${args.in}`,
  );

  // 2. DERIVE — register the module and dispatch through the real registry
  register(makeBalanceCheckModule());
  const startedAt = new Date(0).toISOString(); // deterministic stamp for reproducible output
  const runId = runIdFor(model);
  const emitter = new Emitter({ runId, startedAt });

  const act = await dispatch(
    "balance-check",
    { kind: "balance", model },
    model.reactions.length,
  );
  if (!act.ok) {
    console.error(`[derive] balance-check failed: ${act.error}`);
    process.exit(1);
  }
  const report = act.output_delta;
  console.error(
    `[derive] balance-check: allBalanced=${report.allBalanced}, residue=${act.residue}`,
  );

  // 3. SEND
  emitBalance(emitter, report);
  emitter.finalize(new Date(0).toISOString());

  const serialised = args.trig
    ? await emitter.toTriG()
    : await emitter.toTurtle();
  await writeFile(args.out, serialised, "utf8");
  console.error(
    `[send] wrote ${emitter.store.size} enrichment quads -> ${args.out}`,
  );

  // audit trail (the registry's own log of every dispatch)
  const log = getAuditLog();
  console.error(`[audit] ${log.length} dispatch(es) recorded`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
