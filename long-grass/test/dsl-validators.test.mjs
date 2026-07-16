// Unit tests for the vaHera validator adapter (A2).
// Runs against the REAL parseVahera — this is the empty-dictionary ground
// truth, so the test doubles as a spec conformance check for the pack examples.
import test from "node:test";
import assert from "node:assert/strict";

import { validate, validateVahera, listDsls, getDsl } from "../src/lib/purpose/dsl/validators.js";

test("registry exposes vahera", () => {
  assert.ok(listDsls().includes("vahera"));
  const dsl = getDsl("vahera");
  assert.equal(dsl.moduleId, "vahera");
  assert.equal(dsl.packId, "vahera");
});

test("valid vaHera script passes", () => {
  const src = [
    'memory store "x" = "1"',
    'memory store "y" = "2"',
    'memory find nearest "x" k=2',
    "memory list",
  ].join("\n");
  const res = validateVahera(src);
  assert.equal(res.ok, true);
  assert.deepEqual(res.errors, []);
});

test("all 15 statement forms parse (grammar conformance)", () => {
  const src = [
    'describe SOD1 with "superoxide dismutase 1"',
    "resolve SOD1",
    "spawn analysis from SOD1",
    "navigate to penultimate",
    "complete trajectory",
    "memory create at S(0.2, 1.0, -0.5)",
    'memory store "greeting" = "hello world"',
    'memory find nearest "greeting" k=5',
    "memory list",
    "memory dump greeting",
    "demon sort",
    "controller verify",
    "kernel stats",
    "kernel trace",
    "process list",
  ].join("\n");
  assert.equal(validateVahera(src).ok, true);
});

test("invalid line is rejected with a line number", () => {
  const src = ['memory store "x" = "1"', "this is not vahera"].join("\n");
  const res = validateVahera(src);
  assert.equal(res.ok, false);
  assert.equal(res.errors.length, 1);
  assert.equal(res.errors[0].line, 2);
  assert.match(res.errors[0].message, /unknown vaHera/);
});

test("malformed coordinate is rejected", () => {
  const res = validateVahera("memory create at S(bogus)");
  assert.equal(res.ok, false);
  assert.equal(res.errors[0].line, 1);
});

test("validate() dispatches by dslId and throws on unknown DSL", () => {
  assert.equal(validate("vahera", "memory list").ok, true);
  assert.throws(() => validate("nope", "x"), /unknown DSL/);
});
