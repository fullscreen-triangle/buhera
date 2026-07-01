/* ============================================================================
 * lavoisier Module Adapter
 *
 * Wraps lavoisier's virtual mass-spec instrument (src/lib/lavoisier) as a
 * Buhera module conforming to the Module trait in registry.js.
 *
 * v1 exposes one instruction kind: "virtual_run". The instruction describes an
 * experiment (analyte set + ionisation + acquisition), the module calls
 * lavoisier's runExperiment() forward simulator, and returns the predicted
 * records plus a summary.
 *
 * Instruction shapes accepted:
 *   • { kind: "virtual_run", ...config }  — full explicit config
 *   • "demo"                              — canned lipidomics PC demo
 *   • ""                                  — same as "demo"
 * ========================================================================== */

import { runExperiment, summariseRecords } from "../lavoisier/experiment/virtualinstrument.js";

// --------------------------------------------------------------------------
// Default demo config: one common lipid class over a small chain range on an
// orbitrap in positive mode. Small enough to run instantly, large enough to
// return a non-trivial spectrum set.
// --------------------------------------------------------------------------

const DEMO_CONFIG = {
  experimentType: "lipidomics",
  classSpecs: [
    { classKey: "PC", Xmin: 30, Xmax: 38, Ymin: 0, Ymax: 4 },
  ],
  adductsAllowed: null,
  polarity: "+",
  analyser: "orbitrap",
  analyserCfg: {},
  collisionEnergy_eV: 25,
  mzWindow: [400, 1000],
};

function resolveConfig(instruction) {
  if (instruction == null || instruction === "" || instruction === "demo") {
    return DEMO_CONFIG;
  }
  if (typeof instruction === "string") {
    // Any other bare string: treat as an unrecognised alias; fall back to demo.
    return DEMO_CONFIG;
  }
  if (typeof instruction === "object" && instruction.kind === "virtual_run") {
    // Merge over defaults so callers can supply a partial config.
    const { kind: _kind, ...cfg } = instruction;
    return { ...DEMO_CONFIG, ...cfg };
  }
  if (typeof instruction === "object") {
    return { ...DEMO_CONFIG, ...instruction };
  }
  return DEMO_CONFIG;
}

// --------------------------------------------------------------------------
// The Module trait implementation.
// --------------------------------------------------------------------------

export const lavoisierModule = {
  id: "lavoisier",

  describe() {
    return {
      id: "lavoisier",
      description:
        "lavoisier virtual mass-spec: forward-simulate an experiment " +
        "from an analyte + ionisation + acquisition spec.",
      instructions: [
        'dispatch("lavoisier", "demo")',
        'dispatch("lavoisier", { kind: "virtual_run", experimentType: "lipidomics", classSpecs: [{ classKey: "PC" }], polarity: "+", analyser: "orbitrap" })',
      ],
    };
  },

  /**
   * Execute a virtual mass-spec experiment.
   *
   * actBudget is a hint about thoroughness. The current forward simulator is
   * deterministic and ignores it; keeping the signature future-proof.
   */
  async execute(instruction, _actBudget = 1) {
    const cfg = resolveConfig(instruction);

    try {
      const records = runExperiment(cfg);
      const summary = summariseRecords(records);

      return {
        ok: true,
        output_delta: {
          kind: "lavoisier_run",
          summary,
          records,
          config: cfg,
        },
        residue: records.length,
        completed: true,
      };
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`lavoisier error: ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }
  },

  /**
   * For sufficiency checks. The output cell is the current predicted record
   * set. v1 returns a stub; a real cell arrives with the orchestrator's gate.
   */
  outputCell(_instruction) {
    return { kind: "spectra_state" };
  },
};
