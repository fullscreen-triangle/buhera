/* ============================================================================
 * Smith Module — agent generation
 *
 * Buhera module that compiles the agent DSL (Split-Attention Synchronised
 * Agents) into checked, characterised agents and, optionally, a deterministic
 * run trace. Unlike most modules here, smith does NO network I/O: the compiler
 * and its mathematics (χ, realised floor, water-fill) are pure JS that runs
 * in-browser. This is the no-install webtool path — the same tool is offered
 * server-side as a downloadable Rust CLI (the twin), but neither is in the
 * loop of the other.
 *
 * Output kind produced:
 *   agent_generated — per-agent character (χ), realised floor, regime,
 *                     the χ-partition, and diagnostics; plus an optional run
 *                     trace when the caller asks for it.
 *
 * Instruction shapes accepted by execute():
 *   string                              — DSL source; check only
 *   { source, run?, maxTicks? }         — source; run the tick loop when run:true
 * ========================================================================== */

import { compile, run } from "../smith/compiler";

function normalise(instruction) {
  if (instruction == null) return { source: "", run: false, maxTicks: 30 };
  if (typeof instruction === "string") {
    return { source: instruction, run: false, maxTicks: 30 };
  }
  if (typeof instruction === "object") {
    return {
      source: String(instruction.source || ""),
      run: instruction.run === true,
      maxTicks: Number.isInteger(instruction.maxTicks) ? instruction.maxTicks : 30,
    };
  }
  return { source: String(instruction), run: false, maxTicks: 30 };
}

// residue mirrors the aggregate realised floor (lower = more coherent, i.e. a
// cheaper minimal separation), matching purpose/graffiti's residue = floor
// convention. Infinite floors (single-part agents) contribute nothing.
function aggregateFloor(agents) {
  const finite = agents.map((a) => a.floor).filter((f) => Number.isFinite(f));
  if (finite.length === 0) return 0;
  return finite.reduce((s, f) => s + f, 0);
}

export const smithModule = {
  id: "smith",

  describe() {
    return {
      id: "smith",
      description:
        "Smith: generate Split-Attention Synchronised Agents from the agent DSL. " +
        "Parses and checks the script, computes each agent's character invariant χ " +
        "and realised floor, and optionally runs the deterministic tick loop.",
      instructions: [
        'dispatch("smith", "agent A { purpose minimise T; scene s serves T with h; self { parts { p, q } separations (p,q: 3) } budget 4 floor 2 }")',
        'dispatch("smith", { source: "<dsl>", run: true })',
        'dispatch("smith", { source: "<dsl>", run: true, maxTicks: 12 })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const { source, run: doRun, maxTicks } = normalise(instruction);

    if (!source.trim()) {
      return {
        ok: false,
        output_delta: {
          kind: "agent_generated",
          ok: false,
          agents: [],
          diagnostics: [{ message: "smith: no DSL source provided.", severity: "error" }],
        },
        residue: 0,
        completed: true,
        error: "no-source",
      };
    }

    let compiled;
    try {
      compiled = compile(source);
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "agent_generated",
          ok: false,
          agents: [],
          diagnostics: [{ message: `smith: compile error — ${err.message || String(err)}`, severity: "error" }],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }

    const chk = compiled.check;
    const agents = chk.agents.map((a) => ({
      name: a.name,
      regime: a.regime,
      chi: Number.isFinite(a.chi) ? a.chi : null,
      floor: Number.isFinite(a.floor) ? a.floor : null,
      nonLocal: a.nonLocal,
      chiPartition: a.chiPartition,
    }));

    let steps;
    let finalCounts;
    if (doRun && chk.ok) {
      try {
        const r = run(compiled.file, maxTicks);
        steps = r.steps;
        finalCounts = r.finalCounts;
      } catch (err) {
        chk.errors.push({
          message: `smith: run error — ${err.message || String(err)}`,
          severity: "error",
        });
      }
    }

    return {
      ok: chk.ok,
      output_delta: {
        kind: "agent_generated",
        ok: chk.ok,
        agents,
        diagnostics: chk.errors,
        steps,
        finalCounts,
      },
      residue: aggregateFloor(agents),
      completed: true,
      error: chk.ok ? undefined : (chk.errors[0] && chk.errors[0].message) || "check failed",
    };
  },

  outputCell() {
    return { kind: "agent_generated_cell" };
  },
};
