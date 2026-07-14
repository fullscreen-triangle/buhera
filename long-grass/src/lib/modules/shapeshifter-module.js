/* ============================================================================
 * Shape Shifter Module Adapter
 *
 * Wraps the Shape Shifter DSL interpreter (src/lib/lavoisier/shapeshifter) as a
 * Buhera module conforming to the Module trait in registry.js.
 *
 * Shape Shifter is a declarative language (.ss) for describing virtual
 * mass-spec experiments: `objective`, `instrument`, `validate`, `phase`, and
 * `target_list` blocks whose `phase` statements dispatch lavoisier.* capability
 * calls (instrument runs, partition addresses, SEBD-MS, S-entropy, purpose
 * domains). The module compiles the source, runs its phases, and returns the
 * terminal stream plus the produced workspace.
 *
 * Instruction shapes accepted:
 *   • a bare string                       — treated as .ss source and run
 *   • { kind: "run", source: "..." }      — explicit source form
 *   • "demo" | ""                         — a canned lipidomics .ss script
 *
 * Online DB-search calls (lavoisier.db.search*) are returned by the interpreter
 * as unresolved "pending" sentinels; this v1 surfaces them as pending rather
 * than resolving the network fetch. All pure-compute capabilities run fully.
 * ========================================================================== */

import { compileStage, executeStage } from "@lavoisier/shapeshifter";

// --------------------------------------------------------------------------
// Canned demo: one lipid class over a small chain range, then map the records
// to a partition wave-field. Small enough to run instantly, exercises both an
// instrument run and a follow-on partition capability.
// --------------------------------------------------------------------------

const DEMO_SOURCE = `objective demo:
  target: "PC lipids, small chain range, positive mode"

instrument orbi:
  kappa: 1e12
  ref_frequency: 10e6

phase acquire:
  records = lavoisier.instrument.run_experiment(classes: ["PC"], polarity: "+", analyser: "orbitrap", mz_window: [400, 1000])
  field = lavoisier.observe.partition_field(records: records)
`;

/**
 * Resolve an instruction to Shape Shifter source text, or null if the shape is
 * not one the module accepts.
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
  if (typeof instruction === "object" && typeof instruction.source === "string") {
    return instruction.source;
  }
  return null;
}

// --------------------------------------------------------------------------
// The Module trait implementation.
// --------------------------------------------------------------------------

export const shapeshifterModule = {
  id: "shapeshifter",

  describe() {
    return {
      id: "shapeshifter",
      description:
        "Shape Shifter DSL: compile and run a .ss script describing a virtual " +
        "mass-spec experiment (objective/instrument/phase blocks).",
      instructions: [
        'dispatch("shapeshifter", "demo")',
        'dispatch("shapeshifter", "phase p:\\n  r = lavoisier.instrument.run_experiment(classes: [\\"PC\\"])")',
        'dispatch("shapeshifter", { kind: "run", source: "objective o:\\n  target: \\"...\\"\\nphase p:\\n  ..." })',
      ],
    };
  },

  /**
   * Compile and execute a Shape Shifter script.
   *
   * actBudget is a hint about thoroughness. The interpreter is deterministic
   * and ignores it; the signature is kept future-proof.
   */
  async execute(instruction, _actBudget = 1) {
    const source = resolveSource(instruction);
    if (source == null) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [
            'shapeshifter: instruction must be a .ss source string, "demo", ' +
              'or { kind: "run", source }',
          ],
        },
        residue: 0,
        completed: true,
        error: "invalid instruction",
      };
    }

    try {
      const compiled = compileStage(source);
      if (!compiled.ok) {
        const diag = compiled.diagnostics.find((d) => d.severity === "error");
        return {
          ok: false,
          output_delta: {
            kind: "shapeshifter_run",
            ok: false,
            term: compiled.term,
            workspace: [],
            diagnostics: compiled.diagnostics,
          },
          residue: 0,
          completed: true,
          error: diag ? diag.message : "compile failed",
        };
      }

      const ran = executeStage(compiled.ast);
      const term = [...compiled.term, ...ran.term];
      const workspace = ran.workspace || [];

      return {
        ok: true,
        output_delta: {
          kind: "shapeshifter_run",
          ok: true,
          result: ran.result,
          workspace,
          term,
          diagnostics: compiled.diagnostics,
        },
        // Residue = amount of content produced (workspace entries). Higher =
        // more produced, consistent with the content-count convention used by
        // lavoisier and graffiti.
        residue: workspace.length,
        completed: true,
      };
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`shapeshifter error: ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }
  },

  /**
   * For sufficiency checks. The output cell is the produced workspace. v1
   * returns a stub; a real cell arrives with the orchestrator's gate.
   */
  outputCell(_instruction) {
    return { kind: "shapeshifter_cell" };
  },
};
