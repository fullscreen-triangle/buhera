// Tests for the lattice bridge's parsers, against output captured from the
// real CLI (lattice 0.2.0) in test/fixtures/lattice/.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

import { parseResults, parseSummary, parseTasks, parseUnits, readNotes, resultsState, specArgs } from "../src/lib/server/lattice.js";

const fixture = (n) => fs.readFileSync(new URL(`./fixtures/lattice/${n}`, import.meta.url), "utf8");

test("a spec becomes lattice's flags", () => {
  assert.deepEqual(specArgs({ task: "train", matrix: ["seed=1..5", "model=a,b"], outputs: ["results/**"], gpu: "required", parallel: 2.7 }),
    ["train", "--matrix", "seed=1..5", "--matrix", "model=a,b", "--output", "results/**", "--gpu", "required", "--parallel", "2"]);
  assert.deepEqual(specArgs({ name: "sweep", command: ["python", "sweep.py", "--lr", "{lr}"] }), ["--name", "sweep", "--", "python", "sweep.py", "--lr", "{lr}"]);
  assert.deepEqual(specArgs({ gpu: "lots" }), []);
});

test("tasks and units", () => {
  assert.deepEqual(parseTasks("vscode  score\npixi    train model\n"), [{ source: "vscode", name: "score" }, { source: "pixi", name: "train model" }]);
  assert.deepEqual(parseUnits("score                        3 shard(s)  GPU none        score\nold   unreadable: bad toml\n"),
    [{ name: "score", shards: 3, gpu: "none", from: "score" }, { name: "old", error: "bad toml" }]);
});

test("a real plan summary", () => {
  const s = parseSummary(fixture("plan.txt"));
  assert.equal(s.name, "score");
  assert.equal(s.shards, 3);
  assert.equal(s.split, "seed×3");
  assert.equal(s.compute, "CPU is enough: no GPU library found");
  assert.match(s.runs, /python3 score\.py \{seed\}/);
  assert.equal(s.steps.length, 2);
  assert.equal(s.steps[1].what, "run");
  assert.equal(s.steps[1].command, "cd <repo> && bash .lattice/score/run.sh --detach");
  assert.match(s.steps[1].more[0], /--part 1\/2/);
  assert.match(s.back, /^lattice results score/);
});

test("notes and warnings in a summary", () => {
  const s = parseSummary("unit         x — 1 shard(s)\nnotes        - first\n             - second\nWARNING      - uncommitted changes\n");
  assert.deepEqual(s.notes, ["first", "second"]);
  assert.deepEqual(s.warnings, ["uncommitted changes"]);
});

test("lattice's stderr notes, and its error", () => {
  const n = readNotes(fixture("plan.stderr.txt"));
  assert.equal(n.notes.length, 3);
  assert.match(n.notes[1], /no remote of this repository is on git\.uni-greifswald\.de/);
  assert.equal(n.error, null);
  assert.equal(readNotes("lattice: error: not a git repository").error, "not a git repository");
});

test("real results: the run, every shard, the totals", () => {
  const r = parseResults(fixture("results.txt"));
  assert.equal(r.unit, "score");
  assert.equal(r.runs.length, 1);
  assert.equal(r.runs[0].state, "done");
  assert.equal(r.runs[0].cpus, "8");
  assert.deepEqual(r.shards.map((s) => [s.id, s.state, s.exit]), [["seed-1", "done", 0], ["seed-2", "done", 0], ["seed-3", "done", 0]]);
  assert.deepEqual(r.totals, { done: 3, failed: 0, running: 0, pending: 0, of: 3 });
  assert.equal(r.complete, true);
});

test("pending and running shards have fewer columns", () => {
  const r = parseResults("u — results at abc (2m ago)\n\n  shard  state  exit  took  host\n  seed-1  running  3m…  hostA\n  seed-2  pending\n  seed-3  failed  1  12s  hostB\n\n  0 done, 1 failed, 1 running, 1 not reported, of 3\n");
  assert.deepEqual(r.shards, [
    { id: "seed-1", state: "running", exit: null, took: "3m…", host: "hostA" },
    { id: "seed-2", state: "pending", exit: null, took: null, host: null },
    { id: "seed-3", state: "failed", exit: 1, took: "12s", host: "hostB" },
  ]);
  assert.equal(r.complete, false);
  assert.deepEqual([0, 2, 3, 1].map(resultsState), ["complete", "nothing-pushed", "incomplete", "failed"]);
});
