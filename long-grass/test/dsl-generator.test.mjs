// Unit tests for the generate->validate->repair loop (A4). A stateful fake
// draftFn emits invalid vaHera on the first round and valid on the second,
// proving the loop feeds the REAL compiler's errors back and converges.
import test from "node:test";
import assert from "node:assert/strict";

import { generateDsl } from "../src/lib/purpose/dsl-generator.js";
import { validateVahera } from "../src/lib/purpose/dsl/validators.js";

// Build a draftFn that returns the given code strings in sequence across rounds
// (one provider -> single survivor -> no integration, so round N uses seq[N]).
function sequencedDraft(seq) {
  let i = 0;
  return async ({ provider }) => {
    const code = seq[Math.min(i, seq.length - 1)];
    i++;
    return { ok: true, provider, model: "fake-model", content: JSON.stringify({ code }) };
  };
}

test("valid on first try: 0 repairs", async () => {
  const out = await generateDsl({
    dslId: "vahera",
    instructions: "list memories",
    federationOpts: { providers: ["ollama"], draftFn: sequencedDraft(["memory list"]) },
  });
  assert.equal(out.ok, true);
  assert.equal(out.repairs, 0);
  assert.equal(out.code, "memory list");
  assert.equal(validateVahera(out.code).ok, true);
});

test("invalid then valid: repairs once and converges", async () => {
  const out = await generateDsl({
    dslId: "vahera",
    instructions: "store x=1 then list",
    maxRepairs: 3,
    federationOpts: {
      providers: ["ollama"],
      draftFn: sequencedDraft(["totally bogus line", 'memory store "x" = "1"\nmemory list']),
    },
  });
  assert.equal(out.ok, true, JSON.stringify(out.errors));
  assert.equal(out.repairs, 1);
  assert.equal(out.attempts, 2);
  assert.equal(validateVahera(out.code).ok, true);
});

test("never valid: exhausts repairs and returns ok:false with last code", async () => {
  const out = await generateDsl({
    dslId: "vahera",
    instructions: "do something impossible",
    maxRepairs: 2,
    federationOpts: { providers: ["ollama"], draftFn: sequencedDraft(["bad one", "bad two", "bad three"]) },
  });
  assert.equal(out.ok, false);
  assert.equal(out.repairs, 2);
  assert.equal(out.attempts, 3);
  assert.ok(out.errors.length >= 1);
  assert.ok(out.code); // last attempt surfaced for diagnostics
});

test("no provider: fails fast without repairing", async () => {
  const out = await generateDsl({
    dslId: "vahera",
    instructions: "list memories",
    federationOpts: { providers: [] },
  });
  assert.equal(out.ok, false);
  assert.equal(out.repairs, 0);
  assert.equal(out.errors[0].stage, "provider");
});

test("unknown dsl and empty instructions are rejected", async () => {
  assert.equal((await generateDsl({ dslId: "nope", instructions: "x" })).ok, false);
  assert.equal((await generateDsl({ dslId: "vahera", instructions: "  " })).ok, false);
});
