// Tests for the spraypaint bridge's pure half: which tree a request may
// search, the argv of an ask, and reading 0.2.0's JSON (with the text
// fallbacks kept for older builds).
import test from "node:test";
import assert from "node:assert/strict";

import { askArgs, parseJsonLoose, readCount, readIdentity, readScenes, readVerify, resolveRoot } from "../src/lib/server/spraypaint.js";

const base = { envRoot: null, fallback: "/repo" };

test("a local request may choose any root and falls back to the repo", () => {
  assert.equal(resolveRoot({ ...base, local: true }).root, "/repo");
  assert.equal(resolveRoot({ ...base, local: true, requested: "/elsewhere" }).root, "/elsewhere");
  assert.equal(resolveRoot({ ...base, local: true, envRoot: "/corpus" }).root, "/corpus");
  assert.equal(resolveRoot({ ...base, local: true, action: "index" }).root, "/repo");
});

test("a remote request never falls back to the server's parent directory", () => {
  const r = resolveRoot({ ...base, local: false });
  assert.equal(r.status, 403);
  assert.equal(r.root, undefined);
});

test("a remote request searches SPRAYPAINT_ROOT only, and may not index", () => {
  const env = { ...base, envRoot: "/corpus", local: false };
  assert.equal(resolveRoot(env).root, "/corpus");
  assert.equal(resolveRoot({ ...env, requested: "/corpus" }).root, "/corpus");
  assert.equal(resolveRoot({ ...env, requested: "/etc" }).status, 403);
  assert.equal(resolveRoot({ ...env, action: "index" }).status, 403);
});

test("an ask's argv carries the budget, scenes and dry run", () => {
  assert.deepEqual(askArgs({ query: "water filling", root: "/r" }), ["ask", "water filling", "--root", "/r", "--json", "-k", "8"]);
  const a = askArgs({ query: "q", root: "/r", budget: 0, scenes: ["docs", "src"], dryRun: true });
  assert.deepEqual(a.slice(5), ["-k", "1", "--scenes", "docs,src", "--dry-run"]);
});

test("JSON is found after a progress line", () => {
  assert.deepEqual(parseJsonLoose("Indexing /r ...\n{\"documents\": 3}"), { documents: 3 });
  assert.equal(parseJsonLoose("no json here"), null);
});

test("identity, count and scenes read 0.2.0 JSON", () => {
  assert.deepEqual(readIdentity('{"fingerprint":"b3:ab","char_invariant":1.000002,"floor":1e-6,"n_vertices":3,"n_edges":3}'),
    { fingerprint: "b3:ab", chi: 1.000002, floor: 1e-6, vertices: 3, edges: 3 });
  assert.equal(readCount('{"committed_count":4}'), 4);
  assert.equal(readCount('{"committed_count":0}'), 0);
  assert.deepEqual(readScenes('[{"documents":1,"name":"docs","passages":2}]'), [{ scene: "docs", documents: 1, passages: 2 }]);
});

test("identity, count and scenes still read an older build's text", () => {
  assert.equal(readIdentity("fingerprint: b3:ff\nchar invariant (chi): 2.5\nfloor: 0.001\nvertices: 7\nedges: 21").chi, 2.5);
  assert.equal(readCount("committed acts: 12"), 12);
  assert.deepEqual(readScenes("docs   3 doc(s), 9 passage(s)"), [{ scene: "docs", documents: 3, passages: 9 }]);
});

test("verify keeps each check, N/A and the degeneracies", () => {
  const out = readVerify(JSON.stringify({
    overall: "N/A", pass: false, degeneracies: ["fewer than two scenes"],
    inv1_identity: { status: "PASS", pass: true, checks: [{ name: "fingerprint", status: "PASS", detail: "matches" }] },
    inv2_count: { status: "N/A", pass: false, checks: [{ name: "count_readable", status: "N/A", detail: "nothing committed" }] },
  }), 2);
  assert.equal(out.overall, "N/A");
  assert.deepEqual(out.degeneracies, ["fewer than two scenes"]);
  assert.deepEqual(out.invariants.map((i) => i.status), ["PASS", "N/A"]);
  assert.equal(out.invariants[1].checks[0].name, "count_readable");
});

test("verify falls back to text and to the exit code", () => {
  assert.equal(readVerify("Inv 1 conserved identity  [PASS] ok\noverall: PASS", 0).invariants.length, 1);
  assert.equal(readVerify("garbled", 2).overall, "N/A");
  assert.equal(readVerify("garbled", 1).overall, "FAIL");
});
