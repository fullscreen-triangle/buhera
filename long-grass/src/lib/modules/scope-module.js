/* ============================================================================
 * scope Module Adapter
 *
 * Wraps the SCOPE DSL (compiler + runtime, from the linked `scope-lang`
 * package) as a Buhera module conforming to the Module trait in registry.js.
 *
 * SCOPE runs here as a REPL, not as a whole-script sandbox: the user writes
 * SCOPE declarations in pieces, cell by cell, and they accumulate into one
 * growing program (the "script so far"). A cell that only defines things
 * (coordinate_space, channels, a morphism without visualise) is acknowledged;
 * a cell whose code reaches a `visualise` runs the phases against the linked
 * image and returns a chart — the same semantics as the sandbox, fed
 * incrementally. The image is needed only when a cell actually executes.
 *
 * Instruction shapes accepted:
 *   • "<scope source>"                     — evaluate a REPL cell (bare decls)
 *   • { kind: "cell", source }             — same, explicit
 *   • { kind: "load", image: {data,width,height} }  — link an image payload
 *   • { kind: "state" }                    — report what the session knows
 *   • { kind: "reset" }                    — clear definitions + image
 * ========================================================================== */

import { createSession } from "scope-lang";

// --------------------------------------------------------------------------
// Module-private session singleton. Per the trait guidance, state lives in a
// module-private variable, never on the trait object. Reset via { kind:"reset" }.
// --------------------------------------------------------------------------

let _session = createSession();

/** Replace the live session (used by { kind: "reset" }). */
export function resetScopeSession() {
  _session = createSession();
}

/** Link an image into the current session (used by the terminal's `:scope
 *  load` command, which decodes a URL to an ImagePayload first). */
export function linkScopeImage(imagePayload) {
  _session.setImage(imagePayload);
}

function isImagePayload(x) {
  return (
    x &&
    typeof x === "object" &&
    x.data &&
    typeof x.width === "number" &&
    typeof x.height === "number"
  );
}

export const scopeModule = {
  id: "scope",

  describe() {
    return {
      id: "scope",
      description:
        "SCOPE DSL REPL: write microscopy-analysis declarations cell by cell; " +
        "a cell that visualises produces a chart against the linked image.",
      instructions: [
        'dispatch("scope", "coordinate_space { field 100 x 100 µm  depth 4  lambda_s 0.10  lambda_t 0.05 }")',
        'dispatch("scope", "seg = observe(load(db=\\"cells\\", dataset=\\"cells\\", image=\\"x.jpg\\"), n=4) |> visualise(scale_field)")',
        'dispatch("scope", { kind: "state" })',
        'dispatch("scope", { kind: "reset" })',
      ],
    };
  },

  /**
   * Evaluate one REPL cell (or a control instruction).
   *
   * actBudget is a thoroughness hint; SCOPE's phase pipeline is deterministic
   * and ignores it.
   */
  async execute(instruction, _actBudget = 1) {
    // ── Control instructions ────────────────────────────────────────────
    if (instruction && typeof instruction === "object") {
      if (instruction.kind === "reset") {
        resetScopeSession();
        return {
          ok: true,
          output_delta: { kind: "text", lines: ["scope: session reset"] },
          residue: 0,
          completed: true,
        };
      }

      if (instruction.kind === "state") {
        const s = _session.state();
        const lines = [
          `coordinate_space: ${s.hasCoordinateSpace ? "set" : "—"}`,
          `channels: ${s.hasChannels ? "set" : "—"}`,
          `morphisms: ${s.morphisms.length ? s.morphisms.join(", ") : "—"}`,
          `goal: ${s.hasGoal ? "set" : "—"}  dispatch: ${s.hasDispatch ? "set" : "—"}`,
          `image: ${s.hasImage ? "linked" : "not linked"}`,
        ];
        return {
          ok: true,
          output_delta: { kind: "text", lines },
          residue: s.morphisms.length,
          completed: true,
        };
      }

      if (instruction.kind === "load") {
        if (!isImagePayload(instruction.image)) {
          return badInstruction(
            "scope: load requires an image payload { data, width, height }",
          );
        }
        _session.setImage(instruction.image);
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: [
              `scope: image linked (${instruction.image.width}×${instruction.image.height})`,
            ],
          },
          residue: 1,
          completed: true,
        };
      }
    }

    // ── A REPL cell ─────────────────────────────────────────────────────
    const source =
      typeof instruction === "string"
        ? instruction
        : instruction && typeof instruction === "object" && instruction.kind === "cell"
          ? String(instruction.source ?? "")
          : null;

    if (source == null) {
      return badInstruction(
        'scope: instruction must be SCOPE source, or { kind: "cell"|"load"|"state"|"reset" }',
      );
    }

    let cell;
    try {
      cell = await _session.run(source);
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`scope error: ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }

    // Map the CellResult onto an ActResult.
    if (cell.kind === "noop") {
      return {
        ok: true,
        output_delta: { kind: "text", lines: ["(empty cell)"] },
        residue: 0,
        completed: true,
      };
    }

    if (cell.kind === "error") {
      return {
        ok: false,
        output_delta: { kind: "text", lines: cell.log.length ? cell.log : [cell.error] },
        residue: 0,
        completed: true,
        error: cell.error,
      };
    }

    if (cell.kind === "define") {
      // A defining cell: acknowledge what was added. residue counts the
      // session's accumulated morphisms (content produced grows the count).
      return {
        ok: true,
        output_delta: { kind: "text", lines: cell.log },
        residue: _session.state().morphisms.length,
        completed: true,
      };
    }

    // cell.kind === "chart": an executing cell produced a full ScopeResult.
    return {
      ok: true,
      output_delta: {
        kind: "scope_run",
        result: cell.result,
        log: cell.log,
      },
      // Confidence proxy: closer to the goal is lower residue. Use the mean
      // s-entropy sum as a stand-in signal (always finite, non-negative).
      residue: residueFromResult(cell.result),
      completed: true,
    };
  },

  /**
   * For sufficiency checks. The output cell is the current session's rendered
   * result. v1 returns a stub; a real cell arrives with the orchestrator's gate.
   */
  outputCell(_instruction) {
    return { kind: "scope_cell" };
  },
};

// --------------------------------------------------------------------------
// Helpers.
// --------------------------------------------------------------------------

function badInstruction(msg) {
  return {
    ok: false,
    output_delta: { kind: "text", lines: [msg] },
    residue: 0,
    completed: true,
    error: msg,
  };
}

function residueFromResult(result) {
  const sum = result?.sEntropy?.sum;
  return typeof sum === "number" && isFinite(sum) && sum >= 0 ? sum : 1;
}
