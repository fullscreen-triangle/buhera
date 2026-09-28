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
