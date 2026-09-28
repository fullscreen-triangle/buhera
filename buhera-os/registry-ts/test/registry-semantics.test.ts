// Registry semantics R1–R6 and DSL routing D1 (specification 03-registry.md
// §4). The Rust crate runs the same cases in
// buhera-registry/tests/registry_semantics.rs.
import test from "node:test";
import assert from "node:assert/strict";

import {
  DslRegistry,
  Registry,
  UnknownModuleError,
  done,
  invalidSource,
  lineFromMessage,
  valid,
  type Module,
} from "../src/index.ts";

function echo(start = 0): Module {
  let calls = start;
  return {
    id: "echo",
    describe: () => ({ id: "echo", description: "echo", instructions: [], binding: "native" }),
    execute(instruction) {
      calls += 1;
      return done({ kind: "echo", value: instruction, calls }, 0);
    },
  };
}

const thrower: Module = {
  id: "panicky",
  describe: () => ({ id: "panicky", description: "always throws", instructions: [], binding: "native" }),
  async execute() {
    throw new Error("boom");
  },
};

test("R1 unknown module throws and is not audited", async () => {
  const r = new Registry();
  await assert.rejects(r.dispatch("nope", "x"), UnknownModuleError);
  assert.equal(r.auditLog().length, 0);
});

test("R2 a throwing module is contained with a null delta", async () => {
  const r = new Registry();
  r.register(thrower);
  const res = await r.dispatch("panicky", {});
  assert.deepEqual(res, { ok: false, output_delta: null, residue: 0, completed: true, error: "boom" });
  assert.equal(r.auditLog().length, 1);
});

test("R3 act ids are monotone and survive clearing", async () => {
  const r = new Registry();
  r.register(echo());
  r.register(thrower);
  await r.dispatch("echo", 1);
  await r.dispatch("panicky", 2);
  r.clearAuditLog();
  await r.dispatch("echo", 3);
  assert.equal(r.auditLog()[0]?.act_id, 3);
});

test("R4 hooks run in order; a failing hook is isolated", async () => {
  const r = new Registry();
  r.register(echo());
  const seen: string[] = [];
  const warn = console.warn;
  console.warn = () => {};
  try {
    r.onDispatch((e) => seen.push(`a${e.act_id}`));
    r.onDispatch(() => {
      throw new Error("hook failure");
    });
    const offB = r.onDispatch((e) => seen.push(`b${e.act_id}`));
    assert.equal((await r.dispatch("echo", "x")).ok, true);
    offB();
    await r.dispatch("echo", "y");
  } finally {
    console.warn = warn;
  }
  assert.deepEqual(seen, ["a1", "b1", "a2"]);
});

test("R5 register replaces and returns the previous binding", async () => {
  const r = new Registry();
  assert.equal(r.register(echo()), null);
  await r.dispatch("echo", 1);
  assert.notEqual(r.register(echo(100)), null);
  const res = await r.dispatch("echo", 1);
  assert.equal(res.output_delta?.["calls"], 101);
});

test("R6 module state persists between acts", async () => {
  const r = new Registry();
  r.register(echo());
  for (let i = 0; i < 3; i++) await r.dispatch("echo", null);
  const e = r.auditLog()[2];
  assert.equal(e?.result.output_delta?.["calls"], 3);
  assert.equal(e?.act_budget, 1);
  assert.ok(e?.timestamp.endsWith("Z"));
});

test("D1 DSL registry routes and validates", () => {
  const d = new DslRegistry();
  d.register({
    id: "braces",
    label: "Braces",
    extension: ".br",
    moduleId: "echo",
    packId: "braces",
    validate(src) {
      let depth = 0;
      const lines = src.split("\n");
      for (let i = 0; i < lines.length; i++) {
        for (const c of lines[i] as string) {
          depth += c === "{" ? 1 : c === "}" ? -1 : 0;
          if (depth < 0) return invalidSource([{ message: "unbalanced }", line: i + 1 }]);
        }
      }
      return depth === 0 ? valid() : invalidSource([{ message: "unclosed {" }]);
    },
  });
  assert.equal(d.validate("braces", "a { b }").ok, true);
  assert.equal(d.validate("braces", "a\n}").errors[0]?.line, 2);
  assert.throws(() => d.validate("nope", ""));
  assert.equal(d.byExtension(".BR")?.moduleId, "echo");
  assert.equal(lineFromMessage("line 12: unexpected token"), 12);
});
