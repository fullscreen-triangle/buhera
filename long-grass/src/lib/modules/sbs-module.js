/* ============================================================================
 * SBS Module Adapter
 *
 * Wraps @sachikonye/sbs — the Systems Biology Shaders DSL runtime — as a Buhera
 * module. The package (source of truth in the hegel repo, linked via npm link)
 * ships its own tokenizer, parser, compiler, WebGL2 solver, and metrics
 * extractor. This adapter's only job is to run a .sbs script and translate the
 * result into an ActResult.
 *
 * Each dispatch is one complete SBS script: source in, result out, stateless.
 * The user progresses by reading a result and writing the next script — the
 * notebook thread lives with the user, not in this module.
 *
 * Instruction shapes:
 *   • a plain string             — treated as .sbs source
 *   • { kind: "run", source }    — same, explicit shape
 *   • "demo" / ""                — the canonical glycolysis demo
 * ========================================================================== */

import { runSBS } from "@sachikonye/sbs";

const DEMO_SOURCE = `// Glycolysis — the canonical SBS demo circuit
circuit glycolysis {
  node Glucose  { mu: -917.0, concentration: 5.0, compartment: "cytoplasm" }
  node G6P      { mu: -1760.0, concentration: 0.083 }
  node F6P      { mu: -1755.0, concentration: 0.014 }
  node FBP      { mu: -2600.0, concentration: 0.031 }
  node G3P      { mu: -1290.0, concentration: 0.14 }
  node BPG13    { mu: -2356.0, concentration: 0.001 }
  node PG3      { mu: -1515.0, concentration: 0.1 }
  node PG2      { mu: -1510.0, concentration: 0.03 }
  node PEP      { mu: -1263.0, concentration: 0.023 }
  node Pyruvate { mu: -472.0, concentration: 0.051 }

  edge Glucose  -> G6P      { rate: 230.0, conductance: 464.1 }
  edge G6P      -> F6P      { rate: 100.0, conductance: 3.35 }
  edge F6P      -> FBP      { rate: 150.0, conductance: 0.85 }
  edge FBP      -> G3P      { rate: 80.0,  conductance: 1.0 }
  edge G3P      -> BPG13    { rate: 200.0, conductance: 11.3 }
  edge BPG13    -> PG3      { rate: 300.0, conductance: 0.12 }
  edge PG3      -> PG2      { rate: 180.0, conductance: 7.27 }
  edge PG2      -> PEP      { rate: 100.0, conductance: 1.21 }
  edge PEP      -> Pyruvate { rate: 500.0, conductance: 4.64 }
}

observe glycolysis
perturb glycolysis { factor: 0.1 }
navigate from Pyruvate
`;

/**
 * Coerce whatever the caller dispatched into SBS source text.
 * Returns null when the instruction cannot be interpreted as a script.
 */
function resolveSource(instruction) {
  if (instruction == null || instruction === "" || instruction === "demo") {
    return DEMO_SOURCE;
  }
  if (typeof instruction === "string") {
    return instruction;
  }
  if (typeof instruction === "object" && instruction.kind === "run") {
    return typeof instruction.source === "string" ? instruction.source : null;
  }
  return null;
}

/** Human-readable one-line summary of a run, for the audit log and text fallback. */
function summarize(result) {
  if (!result.circuit) {
    return "SBS: compiled (no circuit declared)";
  }
  const { numNodes, numEdges } = result.circuit;
  const m = result.metrics;
  const parts = [`${numNodes} nodes, ${numEdges} edges`];
  if (m) {
    parts.push(`R=${m.R.toFixed(3)}`, `V=${m.V.toFixed(3)}`, `[${m.backend}]`);
  }
  return `SBS: ${parts.join("  ")}`;
}

export const sbsModule = {
  id: "sbs",

  describe() {
    return {
      id: "sbs",
      description:
        "Systems Biology Shaders — compile and run .sbs scripts: build a " +
        "cellular circuit, render the S-entropy observation (WebGL2), and " +
        "return coherence (R), flux visibility (V), and backward navigation.",
      instructions: [
        'dispatch("sbs", "demo")',
        'dispatch("sbs", "circuit c { node A { mu: -900, concentration: 5 } ... } observe c")',
        'dispatch("sbs", { kind: "run", source: "<.sbs source>" })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const source = resolveSource(instruction);
    if (source == null) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [
            'sbs: instruction must be .sbs source text, "demo", or ' +
              '{ kind: "run", source }',
          ],
        },
        residue: 0,
        completed: true,
        error: "invalid instruction",
      };
    }

    let result;
    try {
      // runSBS never throws — it returns { ok:false, errors } on compile
      // failure and falls back to the CPU solver when WebGL2 is unavailable.
      // The try/catch is defence in depth so no unexpected throw escapes into
      // the registry's generic error path (which nulls output_delta).
      result = runSBS(source);
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`sbs: unexpected error — ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }

    if (!result.ok) {
      const msgs = (result.errors || []).map(
        (e) => `  line ${e.line ?? 0}: ${e.message}`
      );
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: ["sbs: compile failed", ...msgs],
        },
        residue: 0,
        completed: true,
        error: result.errors?.[0]?.message || "compile failed",
      };
    }

    // Success. Carry the full result in output_delta so the renderer can draw
    // the charts (@sachikonye/sbs/react MetricsDashboard reads metrics +
    // circuit + navigation). residue is a content-count placeholder until the
    // scheduler lands — a real number, no false distance-from-solution claim.
    const nodeCount = result.circuit?.numNodes ?? 0;
    const edgeCount = result.circuit?.numEdges ?? 0;

    return {
      ok: true,
      output_delta: {
        kind: "sbs_result",
        summary: summarize(result),
        circuit: result.circuit,
        metrics: result.metrics,
        navigation: result.navigation,
        observations: result.observations,
        perturbations: result.perturbations,
        warnings: result.warnings,
      },
      residue: nodeCount + edgeCount,
      completed: true,
    };
  },

  outputCell(_instruction) {
    return { kind: "sbs_cell" };
  },
};
