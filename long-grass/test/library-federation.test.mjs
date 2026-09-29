// The library federation as long-grass hosts it: the DSL registry facade
// carries every catalogue language with its REAL front end (the Rust ones via
// wasm), and library modules dispatch through the historical registry API.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { listDsls, validate } from "../src/lib/purpose/dsl/validators.js";
import { register, dispatch, getAuditLog } from "../src/lib/modules/registry.js";
import { pylonModule } from "../src/lib/modules/pylon-module.js";
import { tempusModule } from "../src/lib/modules/tempus-module.js";

const catalogue = JSON.parse(readFileSync(new URL("../../specifications/registry/catalogue.json", import.meta.url), "utf8"));

test("every catalogue language with a TS validator is registered in long-grass", () => {
  const want = catalogue.dsls.filter((d) => d.validators.includes("ts")).map((d) => d.id);
  for (const id of want) assert.ok(listDsls().includes(id), `missing ${id}`);
});

test("each language's own front end accepts a valid script and rejects a broken one", () => {
  const cases = {
    turbulance: ["item x = 1\n// ---\nprint(x)", "item = = 1"],
    wt: ['scope S:\n    include "src/**"\n', "assert:\n    r_est >> 3\n"],
    sbs: ["circuit c { node A { mu: 1 } }", "circuit c { node A { mu: 1 }"],
    hfq: ["plan p {\n  budget 5 requests\n  let x = from chebi ask descendants_of(\"CHEBI:1\")\n  emit x\n}", "plan p {\n  let x = from chebi ask\n}"],
    srn: ["|t : (2,1,0,+)| not { k } do { emit self.n } to { * }", "|t : (2,1,0,+)| do { emit self.n } to { * }"],
    smith: ["agent a { purpose reach done scenes { scene s serves done with h } self { parts { p, q } separations { (p, q: 3) } } budget 1 floor 2 }", "agent a { purpose reach done }"],
    synopsis: [
      "open m = \"m.fa\"\nopen t = \"t.fa\"\nunder nucleotide {\n    let q = project m by channels(dna)\n    let s = project t by channels(dna)\n    bind r, res_r = compare q against s by xcorr(normalised)\n    record r, res_r\n}\nreport to \"x.report\"\n",
      "open m = \"m.fa\"\nopen t = \"t.fa\"\nunder nucleotide {\n    let q = project m by channels(dna)\n    let s = project t by channels(dna)\n    bind r, res_r = compare q against s by xcorr(normalised)\n    record r\n}\nreport to \"x.report\"\n",
    ],
    cfc: ["floor 1e-9\nemit \"ok\"\n", "floor 1e-9\nadmit h yield v\n"],
    sthurbert: ["navigate * ; show chi", "show nope"],
    honjo: ["floor 1.0\nO := cut 8\nobserve O\n", "floor 0\nO := cut 8\n"],
    mishima: ["floor 0.02\nseek g\n  not    { thin }\n  toward { region(r) }\n  via    { rung a at 0.45 >> rung b at 0.30 >> rung c at 0.55 }\n  until  closure\n  otherwise decline\n  yield  found\n", "floor 0.02\nseek g\n  toward { region(r) }\n  via    { rung a at 0.45 >> rung b at 0.30 >> rung c at 0.55 }\n  until  closure\n  otherwise decline\n  yield  found\n"],
    sangoma: ["floor 0.02\nconstruct c {\n  stage s\n  target {\n    crest >= 6.0#0.5\n  }\n  via { rung s at 0.40 >> rung t at 0.35 >> rung u at 0.55 }\n}\n", "floor 0.0\n"],
    mekaneck: ["# Individuate a coherence regime against the rest of the record.\n#\n# The exclusion clause is mandatory: a target is not determined by a positive\n# description alone, so `excluding` says what the regime is being told apart\n# from. Termination is by closure, not by a confidence threshold.\n\nsubstrate Osc {\n  receivers  : recordings(\"cohort-A\");\n  observable : coherence_index();\n  events     : label_change();\n  floor      : asymptotic_separation();   # falsifiable; not sample_minimum()\n}\n\n# Three mutually independent catalysts: fewer cannot survive the loss of a\n# member, so the type checker rejects them.\ncatalyst spectral  : band_decomposition() independent surrogate, phase;\ncatalyst surrogate : phase_randomised()   independent spectral, phase;\ncatalyst phase     : locking_value()      independent spectral, surrogate;\n\nlet regime =\n  seek        target_state(\"high-coherence\")\n  excluding   all_other_states()\n  via         (spectral, surrogate, phase)\n  until       closure;\n\nreport regime;\n", "# Individuate a coherence regime against the rest of the record.\n#\n# The exclusion clause is mandatory: a target is not determined by a positive\n# description alone, so `excluding` says what the regime is being told apart\n# from. Termination is by closure, not by a confidence threshold.\n\nsubstrate Osc {\n  receivers  : recordings(\"cohort-A\");\n  observable : coherence_index();\n  events     : label_change();\n  floor      : asymptotic_separation();   # falsifiable; not sample_minimum()\n}\n\n# Three mutually independent catalysts: fewer cannot survive the loss of a\n# member, so the type checker rejects them.\ncatalyst spectral  : band_decomposition() independent surrogate, phase;\ncatalyst surrogate : phase_randomised()   independent spectral, phase;\ncatalyst phase     : locking_value()      independent spectral, surrogate;\n\nlet regime =\n  seek        target_state(\"high-coherence\")\n  via         (spectral, surrogate, phase)\n  until       closure;\n\nreport regime;\n"],
    scope: ["scope nuclear_separation_dynamics {\n  channels {\n    sync dapi at 0.1 µm/pixel\n    cell PROPHASE  bounds (-2.0e-6, -0.8e-6) action nucleus_pair_measurement\n    cell METAPHASE bounds (-0.8e-6,  0.8e-6) action membrane_boundary\n    cell ANAPHASE  bounds ( 0.8e-6,  2.0e-6) action nucleus_pair_measurement\n  }\n\n  coordinate_space {\n    field 100 x 100 µm\n    depth 10\n    lambda_s 0.10\n    lambda_t 0.05\n  }\n\n  goal {\n    distance_uncertainty < 0.5 µm\n    s_entropy_conservation < 1e-12\n    snr > 8.0\n  }\n\n  rule conservation(dna_mass) {\n    invariant: \"total DAPI-stained area is conserved ±5%\"\n    epsilon: 0.008\n  }\n\n  nucleus_pair_measurement = observe(load(db=\"BBBC\", dataset=\"BBBC007\", image=\"A9 p10d.tif\"), n = 10)\n    |> visualise(scale_field)\n    |> catalyze(conservation(dna_mass))\n    |> catalyze(phase_lock(chromatin), confidence = 0.9)\n    |> access(nucleus_a)\n    |> access(nucleus_b)\n    |> visualise(segmentation)\n    |> measure_distance(nucleus_a, nucleus_b)\n    |> visualise(spectral_power)\n    |> visualise(entropy_trajectory)\n    |> visualise(geodesic)\n\n  membrane_boundary = observe(load(db=\"BBBC\", dataset=\"BBBC007\", image=\"A9 p10f.tif\"), n = 10)\n    |> catalyze(phase_lock(plasma_membrane))\n    |> access(cell_boundary)\n    |> visualise(scale_field)\n    |> visualise(segmentation)\n\n  dispatch {\n    when PROPHASE  do execute(nucleus_pair_measurement)\n    when METAPHASE do execute(membrane_boundary)\n    when ANAPHASE  do execute(nucleus_pair_measurement)\n  }\n}", "scope x { }"],
    tempus: ["sync c at 1e6 freq\ncell A bounds (0, 1) action 0\ncompose d=1 channels c into t\nwhen A do emit ok", "cell A bounds (1, 0) action 0"],
  };
  for (const [id, [good, bad]] of Object.entries(cases)) {
    assert.equal(validate(id, good).ok, true, `${id} rejected a valid script: ${JSON.stringify(validate(id, good).errors)}`);
    const v = validate(id, bad);
    assert.equal(v.ok, false, `${id} accepted a broken script`);
    assert.ok(v.errors.length >= 1, `${id}: an invalid verdict carries a diagnostic (L4)`);
  }
});

test("library modules dispatch and audit through the long-grass facade", async () => {
  register(pylonModule);
  register(tempusModule);
  const before = getAuditLog().length;
  const p = await dispatch("pylon", "demo");
  const t = await dispatch("tempus", "demo");
  assert.equal(p.ok, true);
  assert.equal(t.ok, true);
  const log = getAuditLog();
  assert.equal(log.length, before + 2);
  assert.equal(log.at(-1).act_id, log.at(-2).act_id + 1);
});
