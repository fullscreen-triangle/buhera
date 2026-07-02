/* ============================================================================
 * Graffiti Module Adapter
 *
 * Wraps the graffiti library (src/graffiti/) — the .grf DSL for
 * individuation-theoretic search — as a Buhera module.
 *
 * The graffiti library ships its own parser, typechecker, interpreter, and
 * orchestrator. This adapter builds a shared catalyst registry with the
 * built-in fixture catalysts (which cost no runtime dependencies) and
 * exposes runSource() as the dispatch surface.
 *
 * v1 instruction shapes:
 *   • a plain string           — treated as .grf source
 *   • { kind: "seek", source } — same, explicit shape
 *   • "demo"                   — canned example demonstrating one catalyst
 *
 * Later passes: allow the orchestrator to inject additional catalysts
 * (real web search, real inference) via a { kind: "seek", source, catalysts }
 * shape.
 * ========================================================================== */

import {
  CatalystRegistry,
  createFixtureSearchCatalyst,
  createMockInferenceCatalyst,
  runSource,
} from "@/graffiti";

// --------------------------------------------------------------------------
// Default catalyst registry: two harmless mock catalysts so .grf scripts can
// run without any external dependency. Real catalysts (web search, remote
// retrieval, ML inference) plug in via the same CatalystDefinition contract.
// --------------------------------------------------------------------------

function buildDefaultRegistry() {
  const registry = new CatalystRegistry();
  registry.register(
    createFixtureSearchCatalyst(
      "local_search",
      {
        // A few fixture answers demonstrating the interface. Real corpora
        // land when we wire vahera's kernel into a catalyst.
        "founding_year_of(TUM)": "1868",
        "founding_year_of(Technical University of Munich)": "1868",
      },
      0.7,
    ),
  );
  registry.register(
    createMockInferenceCatalyst(
      "restate",
      (claim) => `restated(${claim})`,
      0.5,
    ),
  );
  return registry;
}

// --------------------------------------------------------------------------
// Demo source: a single-seek project that resolves against the local_search
// fixture. Useful smoke test for the terminal.
// --------------------------------------------------------------------------

const DEMO_SOURCE = `
floor 0.02

catalyst local_search {
  namespace: local
  input: Region output: Claim
}

project greeting {
  seek year
    not{ "disputed dates" }
    toward{ founding_year_of(TUM) }
    via{ local_search(year) }
    until converge
    yield year
}
`.trim();

function resolveSource(instruction) {
  if (instruction == null || instruction === "" || instruction === "demo") {
    return DEMO_SOURCE;
  }
  if (typeof instruction === "string") {
    return instruction;
  }
  if (typeof instruction === "object") {
    if (instruction.kind === "seek" && typeof instruction.source === "string") {
      return instruction.source;
    }
    if (typeof instruction.source === "string") {
      return instruction.source;
    }
  }
  return String(instruction);
}

// Serialise a projectResults map to a plain object the renderer can display.
function serialiseResults(projectResults) {
  const out = {};
  for (const [projectName, yieldMap] of projectResults) {
    const yields = {};
    for (const [name, value] of yieldMap) {
      yields[name] = value;
    }
    out[projectName] = yields;
  }
  return out;
}

export const graffitiModule = {
  id: "graffiti",

  describe() {
    return {
      id: "graffiti",
      description:
        "Graffiti: individuation-theoretic search calculus. Runs .grf DSL " +
        "scripts against a catalyst registry.",
      instructions: [
        'dispatch("graffiti", "demo")',
        'dispatch("graffiti", "floor 0.02\\n\\ncatalyst local_search { ... }\\n\\nproject P { seek ... yield x }")',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const source = resolveSource(instruction);
    if (!source.trim()) {
      return {
        ok: true,
        output_delta: { kind: "text", lines: ["(empty graffiti script)"] },
        residue: 0,
        completed: true,
      };
    }

    const registry = buildDefaultRegistry();
    try {
      const result = await runSource(source, registry);
      const results = serialiseResults(result.projectResults);
      const diagnostics = result.compile.diagnostics || [];
      // Residue = number of yielded claims across projects. Higher = more
      // things individuated; a real residue mapping arrives when the paper's
      // scheduler lands here (Buhera's scheduler already exists elsewhere).
      let claimCount = 0;
      for (const projectName of Object.keys(results)) {
        claimCount += Object.keys(results[projectName]).length;
      }
      return {
        ok: true,
        output_delta: {
          kind: "graffiti_result",
          projects: results,
          diagnostics,
          ambient_floor: result.compile.ambientFloor,
        },
        residue: claimCount,
        completed: true,
      };
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`graffiti error: ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }
  },

  outputCell(_instruction) {
    return { kind: "claim_cell" };
  },
};
