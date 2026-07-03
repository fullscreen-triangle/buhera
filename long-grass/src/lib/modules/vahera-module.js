/* ============================================================================
 * vaHera Module Adapter
 *
 * Wraps the existing vaHera interpreter (src/lib/vahera.js) as a Buhera
 * module, conforming to the Module trait from src/lib/modules/registry.js.
 *
 * An "instruction" for this module is a vaHera source string. One execute()
 * call runs the entire script through the kernel and returns the artifacts.
 * ========================================================================== */

import { Kernel } from "@/lib/kernel";
import { executeVahera } from "@/lib/vahera";

// --------------------------------------------------------------------------
// A single shared kernel for the long-grass instance. In a future version
// each agent / session can have its own kernel; for v1 the page-lifetime
// scope is correct.
// --------------------------------------------------------------------------

let _kernel = null;

export function getKernel() {
  if (!_kernel) {
    _kernel = new Kernel(12); // depth-12 ternary refinement
  }
  return _kernel;
}

export function resetKernel() {
  _kernel = null;
}

// --------------------------------------------------------------------------
// The Module trait implementation.
// --------------------------------------------------------------------------

export const vaheraModule = {
  id: "vahera",

  describe() {
    return {
      id: "vahera",
      description:
        "vaHera memory/recall: store, find nearest, list, dump, sort, " +
        "kernel stats, kernel trace, controller verify, demon sort.",
      instructions: [
        'memory store "<name>" = "<text>"',
        'memory find nearest "<query>" k=<n>',
        "memory list",
        "memory dump <name>",
        "demon sort",
        "kernel stats",
        "kernel trace",
        "controller verify",
      ],
    };
  },

  /**
   * Execute a vaHera script (instruction) against the shared kernel.
   *
   * The actBudget is a hint about thoroughness. vaHera is currently
   * deterministic and ignores it; future-proofing the signature now.
   */
  async execute(instruction, _actBudget = 1) {
    const kernel = getKernel();
    const source = typeof instruction === "string"
      ? instruction
      : instruction?.source ?? "";

    if (!source.trim()) {
      return {
        ok: true,
        output_delta: { kind: "text", lines: ["(empty vahera script)"] },
        residue: 0,
        completed: true,
      };
    }

    try {
      const out = executeVahera(source, kernel, {
        useProteinDb: false,
        rerank: true,
      });

      // The result we surface upstream is whichever artifact list the
      // script produced. The orchestrator / terminal can choose to render
      // results one-by-one or the last.
      const artifact = out.lastResult ?? null;
      const residue = out.results ? out.results.length : 0;

      return {
        ok: true,
        output_delta: {
          kind: "vahera_results",
          artifact,
          all_results: out.results || [],
          trace: out.trace || [],
        },
        residue,
        completed: true,
      };
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`vahera error: ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }
  },

  /**
   * For sufficiency checks. The output cell is the kernel's current
   * state summary. v1 returns a stub; full S-distance comparison
   * arrives with the orchestrator's gate.
   */
  outputCell(_instruction) {
    return { kind: "kernel_state", kernel: getKernel() };
  },
};
