// Per-module behaviour against the REAL engines (specifications/specs/*.md,
// "Conformance" sections). The wasm-hosted cases assert the same facts as
// buhera-modules/tests/conformance.rs: one implementation, two hosts.
import test from "node:test";
import assert from "node:assert/strict";

import { createFederation } from "../src/modules/index.ts";
import { realEngines } from "./engines.ts";

const fed = createFederation(await realEngines());
const run = (id: string, instruction: Parameters<typeof fed.registry.dispatch>[1], budget = 1) =>
  fed.registry.dispatch(id, instruction, budget);

// ── wasm-hosted Rust modules (parity with the Rust host) ───────────────────

test("ndombolo (wasm): demo scores the manuscript proposition at 0.9", async () => {
  const r = await run("ndombolo", "demo");
  assert.equal(r.ok, true);
  const m = (r.output_delta as any).propositions[0].motions[0];
  assert.equal(m.name, "Strong");
  assert.ok(Math.abs(m.score - 0.9) < 1e-12);
  assert.equal((r.output_delta as any).stopped_at, null);
});

test("ndombolo (wasm): turbulance validator reports script-absolute lines", () => {
  const v = fed.dsls.validate("turbulance", "item a = 1\n// ---\nitem = = 2");
  assert.equal(v.ok, false);
  assert.equal(v.errors[0]?.line, 3);
});

test("windtunnel (wasm): unmeasured assertions are skipped, never passed", async () => {
  const script = 'scope S:\n    include "src/**"\n\nassert:\n    regime >= Coherent\n    r_est >= 0.75\n    no holonomy_violations\n';
  const r = await run("windtunnel", { kind: "evaluate", script, metric: { regime: "Coherent", r_est: 0.81 } });
  const d = r.output_delta as any;
  assert.equal(d.verdict, "incomplete");
  assert.equal(d.passed, 2);
  assert.equal(d.skipped, 1);
  assert.ok(Math.abs(r.residue - 1 / 3) < 1e-12);
});

test("windtunnel (wasm): measure is unavailable without a filesystem-bearing host", async () => {
  const r = await run("windtunnel", { kind: "measure", script: "scope S:\n", project: "/x" });
  assert.equal(r.ok, false);
  assert.equal(r.error, "bridge unavailable");
});

test("tracker (wasm): χ of a small index; filesystem verbs refuse on this host", async () => {
  const sym = (name: string, file: string, snippet: string) => ({ name, kind: "fn", file, line: 1, snippet });
  const index = { root: "/r", symbols: [sym("alpha_parse", "src/a.rs", "calls beta_emit"), sym("beta_emit", "src/b.rs", "calls alpha_parse"), sym("gamma_note", "docs/c.md", "prose")] };
  const r = await run("tracker", { kind: "character", index });
  const d = r.output_delta as any;
  assert.equal(d.blocks, 3);
  assert.equal(d.fragments, 2);
  assert.ok(d.chi > 0);
  const fs = await run("tracker", { kind: "list", root: "/r" });
  assert.equal(fs.error, "unavailable on this host");
});

// ── TS-native modules ───────────────────────────────────────────────────────

test("sbs: glycolysis demo on the CPU path, residue = 1 − V", async () => {
  const empty = await run("sbs", { kind: "run", source: "", preferCPU: true });
  assert.equal(empty.ok, true, "a script with no circuit compiles and observes nothing");
  assert.equal(empty.residue, 0);
  const demo = await run("sbs", "demo");
  const d = demo.output_delta as any;
  assert.equal(d.kind, "sbs_result");
  assert.equal(d.circuit.numNodes, 10);
  assert.equal(d.backend, "cpu");
  assert.ok(Math.abs(demo.residue - (1 - d.metrics.V)) < 1e-12);
  assert.ok(d.metrics.V < 0.2, "perturbation reduces visibility");
});

test("sbs: adapter warns about the engine's silent edge-prop behaviour", async () => {
  const src = 'circuit c { node A { mu: -10, concentration: 1 } node B { mu: -20, concentration: 1 } edge A -> B { rate: 1 } }\nperturb c { edge: "A->B", factor: 0.2 }';
  const r = await run("sbs", src);
  assert.ok((r.output_delta as any).adapter_warnings.some((w: string) => w.includes("edge")));
});

test("sbs: validator is the engine's own checker", () => {
  assert.equal(fed.dsls.validate("sbs", "circuit c { node A { mu: 1 } }").ok, true);
  assert.equal(fed.dsls.validate("sbs", "circuit c { node A { mu: 1 }").ok, false);
});

test("hfq: Mark Doerr's mark_q1 runs; a static refusal is ok:true with refused_statically", async () => {
  const q1 = await run("hfq", { kind: "preset", id: "mark_q1" });
  assert.equal(q1.ok, true);
  assert.equal((q1.output_delta as any).result.world, "biocat");
  const ill = await run("hfq", { kind: "preset", id: "ill_capability" });
  assert.equal(ill.ok, true);
  assert.equal((ill.output_delta as any).refused_statically, true);
  assert.ok(ill.residue >= 1, "blocked steps are residue");
});

test("hfq: empty is an answer, not residue; budget_trap is refused/budget", async () => {
  const empty = await run("hfq", { kind: "preset", id: "empty_answer" });
  assert.equal((empty.output_delta as any).verdicts.empty >= 1, true);
  const trap = await run("hfq", { kind: "preset", id: "budget_trap" });
  assert.equal((trap.output_delta as any).blockers.budget >= 1, true);
});

test("hfq: validator catches unknown sources before any request", () => {
  const v = fed.dsls.validate("hfq", "plan p {\n  budget 5 requests\n  let x = from nowhere ask record(\"a\")\n  emit x\n}");
  assert.equal(v.ok, false);
  assert.match(v.errors[0]?.message ?? "", /nowhere/);
});

test("pylon: submit allocates an agent; stepping drives residual down to retirement", async () => {
  const s = await run("pylon", { kind: "submit", source: "|t : (2,1,0,+)| not { k } do { emit self.n } to { * }", goal: [0.003, 0.004] });
  assert.equal(s.ok, true);
  const id = (s.output_delta as any).yield.agent as string;
  const before = s.residue;
  const step = await run("pylon", { kind: "step", agent: id }, 100);
  assert.ok(step.residue < before);
  assert.equal(step.completed, true, "goal 0.005 away, 100 ticks of 1e-3 reach it");
});

test("pylon: glyph without a negation boundary is rejected by the real parser", () => {
  const v = fed.dsls.validate("srn", "|x : (2,1,0,+)| do { emit self.n } to { n = 2 }");
  assert.equal(v.ok, false);
  assert.match(v.errors[0]?.message ?? "", /negation/);
});

test("pylon: clear refuses an empty slot list instead of letting the engine throw", async () => {
  const r = await run("pylon", { kind: "clear", agents: ["x"], slots: [] });
  assert.equal(r.error, "no slots");
});

test("tempus: the coolant lesson compiles; simulate is seeded and paged by the act budget", async () => {
  const c = await run("tempus", "demo");
  assert.equal(c.ok, true);
  const missing = await run("tempus", { kind: "simulate", totalEvents: 100 });
  assert.equal(missing.error, "invalid instruction", "simulate needs source text");
  const sim1 = await run("tempus", { kind: "simulate", source: DEMO_TEMPUS, totalEvents: 100, batchSize: 25, seed: 7 }, 2);
  const sim2 = await run("tempus", { kind: "simulate", source: DEMO_TEMPUS, totalEvents: 100, batchSize: 25, seed: 7 }, 2);
  assert.deepEqual(sim1.output_delta, sim2.output_delta, "deterministic under a seed");
  assert.equal((sim1.output_delta as any).synthetic, true);
  assert.equal((sim1.output_delta as any).generated, 50, "2 budget units × 25-event batches");
  assert.equal(sim1.completed, false);
  assert.ok(Math.abs(sim1.residue - 0.5) < 1e-12);
});

test("tempus: diagnostics lesson surfaces the engine's did-you-mean", () => {
  const v = fed.dsls.validate("tempus", "cell WARN bounds (0, 1) action 0\nwhen WARM do emit x");
  assert.equal(v.ok, false);
  assert.match(v.errors.map((e) => e.message).join(" "), /WARN/);
});

test("zangalewa-dsl: an unreachable broker is a result, never a throw", async () => {
  const r = await run("zangalewa-dsl", { kind: "generate", dslId: "vahera", instructions: "list memory" });
  assert.equal(r.ok, false);
  assert.equal(r.error, "remote unreachable");
});

const DEMO_TEMPUS = `sync coolant at 10.0e6 freq
cell NOMINAL  bounds (-1.0e-7, 1.0e-7) action 0
cell WARM     bounds ( 1.0e-7, 5.0e-7) action 1
compose d=1 channels coolant into coolant_traj
when NOMINAL do emit status_ok
when WARM do emit status_warn`;

test("lazy wasm modules: describe before load, run after; a failed load is a result", async () => {
  const { makeLazyWasmModules } = await import("../src/modules/rust-wasm.ts");
  const { loadWasmEngine } = await import("../src/wasm.ts");
  const { readFile } = await import("node:fs/promises");
  const bytes = await readFile(new URL("../wasm/buhera_modules.wasm", import.meta.url));
  const [nd] = makeLazyWasmModules(() => loadWasmEngine(bytes));
  assert.equal(nd?.describe().dsl, "turbulance");
  const r = await nd!.execute("demo", 1);
  assert.equal(r.ok, true);
  assert.equal(nd!.describe().binding, "native");
  const [broken] = makeLazyWasmModules(() => Promise.reject(new Error("404")));
  const b = await broken!.execute("demo", 1);
  assert.equal(b.error, "engine unavailable");
});

test("remote (gateway): verbatim result with executed_on; transport failures are results", async () => {
  const { makeRemoteModule } = await import("../src/modules/remote.ts");
  const fakeFetch = (async (url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body));
    assert.equal(url, "https://gw.example/api/dispatch");
    assert.equal((init.headers as Record<string, string>)["Authorization"], "Bearer t0k");
    return new Response(JSON.stringify({ executed_on: "gateway", act_id: 7, result: { ok: true, output_delta: { kind: "sbs_core_result", module: body.module }, residue: 0.88, completed: true } }), { status: 200 });
  }) as typeof fetch;
  const m = makeRemoteModule({ id: "sbs-core", description: "", instructions: [] }, { baseUrl: () => "https://gw.example/", token: () => "t0k", fetch: fakeFetch });
  const r = await m.execute("demo", 1);
  assert.equal(r.residue, 0.88);
  assert.equal((r.output_delta as any).executed_on, "gateway");
  assert.equal((r.output_delta as any).remote_act_id, 7);
  const signedOut = makeRemoteModule({ id: "sbs-core", description: "", instructions: [] }, { baseUrl: () => "x", token: () => null });
  assert.equal((await signedOut.execute("demo", 1)).error, "remote unauthorized");
  const down = makeRemoteModule({ id: "sbs-core", description: "", instructions: [] }, { baseUrl: () => "http://127.0.0.1:9", token: () => "t" });
  assert.equal((await down.execute("demo", 1)).error, "remote unreachable");
});
