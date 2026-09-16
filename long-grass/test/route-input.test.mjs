// Unit tests for route-input.js's scientific-statement shorthands — the
// sentence-shaped forms (observed/hypothesize/run/compare/record/check/rank)
// that route to the same canonical vaHera the primitive forms already
// execute. These are pure classification tests: no kernel, no React.
import test from "node:test";
import assert from "node:assert/strict";

import { routeInput } from "../src/lib/runtime/route-input.js";

test("observed <name> as \"<text>\" -> describe", () => {
  const r = routeInput('observed ethanol_bp as "boiling point of ethanol"');
  assert.deepEqual(r, {
    type: "vahera",
    vahera: 'describe ethanol_bp with "boiling point of ethanol"',
  });
});

test("hypothesize <name>: \"<text>\" -> describe + resolve", () => {
  const r = routeInput('hypothesize ethanol_bp: "boiling point of ethanol"');
  assert.deepEqual(r, {
    type: "vahera",
    vahera:
      'describe ethanol_bp with "boiling point of ethanol"\nresolve ethanol_bp',
  });
});

test("run <program> on <name> -> spawn", () => {
  const r = routeInput("run query on ethanol_bp");
  assert.deepEqual(r, { type: "vahera", vahera: "spawn query from ethanol_bp" });
});

test("to completion -> navigate + complete", () => {
  const r = routeInput("to completion");
  assert.deepEqual(r, {
    type: "vahera",
    vahera: "navigate to penultimate\ncomplete trajectory",
  });
});

test('compare <name> to "<text>" defaults k=3', () => {
  const r = routeInput('compare aspirin_like to "aspirin"');
  assert.deepEqual(r, {
    type: "vahera",
    vahera: 'memory find nearest "aspirin" k=3',
  });
});

test('compare <name> to "<text>" k=<n> honors explicit k', () => {
  const r = routeInput('compare aspirin_like to "aspirin" k=5');
  assert.deepEqual(r, {
    type: "vahera",
    vahera: 'memory find nearest "aspirin" k=5',
  });
});

test('record "<name>" = "<text>" -> memory store', () => {
  const r = routeInput('record "meeting" = "team retro Thursday 2pm"');
  assert.deepEqual(r, {
    type: "vahera",
    vahera: 'memory store "meeting" = "team retro Thursday 2pm"',
  });
});

test("check consistency -> controller verify", () => {
  const r = routeInput("check consistency");
  assert.deepEqual(r, { type: "vahera", vahera: "controller verify" });
});

test("rank by category -> demon sort", () => {
  const r = routeInput("rank by category");
  assert.deepEqual(r, { type: "vahera", vahera: "demon sort" });
});

test("case-insensitive matching", () => {
  const r = routeInput('Observed X as "test"');
  assert.deepEqual(r, { type: "vahera", vahera: 'describe X with "test"' });
});

test("primitive forms still pass through unchanged", () => {
  assert.deepEqual(routeInput('describe x with "y"'), {
    type: "vahera",
    vahera: 'describe x with "y"',
  });
  assert.deepEqual(routeInput("controller verify"), {
    type: "vahera",
    vahera: "controller verify",
  });
});

test("existing shorthands still work alongside new ones", () => {
  assert.deepEqual(routeInput('store note = "buy milk"'), {
    type: "vahera",
    vahera: 'memory store "note" = "buy milk"',
  });
  assert.deepEqual(routeInput("verify"), {
    type: "vahera",
    vahera: "controller verify",
  });
});

test("malformed scientific-statement input falls through to nl route", () => {
  // No quoted text — doesn't match the "observed" regex, so it isn't
  // silently mis-parsed; it falls through to the general nl/search route.
  const r = routeInput("observed ethanol_bp as boiling point");
  assert.equal(r.type, "nl");
});
