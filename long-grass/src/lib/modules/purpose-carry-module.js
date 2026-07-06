/* ============================================================================
 * Purpose-Carry Module Adapter
 *
 * Wraps @buhera/purpose (the tandem carry library — carry the uncertainty, not
 * the knowledge) as a Buhera module.
 *
 * A per-page Session singleton accumulates a Step for every act dispatched
 * through the registry. Callers (turbulance, other modules) then ask this
 * module for the minimum-sufficient carry toward a goal under a token budget.
 *
 * Instruction shapes:
 *   • { kind: "carry", goal: string[] | string, budget?: number }
 *     - goal: an array of terms or a single content string τ-extracted
 *     - budget: token budget (default 2048)
 *
 *   • { kind: "add", id: string, content: string, terms?: string[] }
 *     - explicit step registration for callers that bypass the audit-log feeder
 *
 *   • { kind: "stats" } — returns { stepCount, ambientFloor }
 *   • { kind: "reset" } — wipes the session
 *
 * Not to be confused with `purpose-module.js`, which wraps the older
 * federation-synthesis version. Different tool, different concern.
 * ========================================================================== */

import { Session } from "@buhera/purpose";
import { extractTerms } from "@/lib/purpose-terms";

// --------------------------------------------------------------------------
// Session singleton. Page-lifetime scope; a fresh page reload starts empty.
// resetSession() is exported for tests and for the "reset" instruction.
// --------------------------------------------------------------------------

let _session = null;

export function getSession() {
  if (!_session) {
    _session = new Session({ cascadeArity: 1 });
  }
  return _session;
}

export function resetSession() {
  _session = null;
}

// --------------------------------------------------------------------------
// Instruction normalisation.
// --------------------------------------------------------------------------

function resolveGoal(raw) {
  if (Array.isArray(raw)) {
    return new Set(raw.map((t) => String(t).toLowerCase()).filter((t) => t.length >= 3));
  }
  if (typeof raw === "string") {
    return extractTerms(raw);
  }
  return new Set();
}

function serializeCarry(carry) {
  // CarryResult contains a Map, which doesn't survive React state; flatten to
  // a plain object the renderer can walk.
  const residueEntries = [];
  if (carry.residueMap && typeof carry.residueMap.entries === "function") {
    for (const [id, r] of carry.residueMap.entries()) {
      residueEntries.push([id, r]);
    }
  }
  return {
    ok: carry.ok,
    keep: carry.keep,
    regenerable: carry.regenerable,
    dropped: carry.dropped,
    ambientFloor: carry.ambientFloor,
    residue_entries: residueEntries,
    diagnostics: carry.diagnostics,
  };
}

// --------------------------------------------------------------------------
// The Module trait.
// --------------------------------------------------------------------------

export const purposeCarryModule = {
  id: "purpose-carry",

  describe() {
    return {
      id: "purpose-carry",
      description:
        "Purpose (tandem carry): given the accumulated audit-log history, " +
        "return the minimum-sufficient carry toward a goal under a token " +
        "budget. Implements @buhera/purpose.",
      instructions: [
        'dispatch("purpose-carry", { kind: "carry", goal: ["term1","term2"], budget: 500 })',
        'dispatch("purpose-carry", { kind: "carry", goal: "free-text description" })',
        'dispatch("purpose-carry", { kind: "add", id: "note1", content: "..." })',
        'dispatch("purpose-carry", { kind: "stats" })',
        'dispatch("purpose-carry", { kind: "reset" })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const session = getSession();

    // Normalise instruction shape (accept a plain string as "carry").
    const inst =
      typeof instruction === "string"
        ? { kind: "carry", goal: instruction }
        : instruction || {};

    const kind = inst.kind || "carry";

    try {
      if (kind === "stats") {
        return {
          ok: true,
          output_delta: {
            kind: "purpose_carry_stats",
            stepCount: session.stepCount(),
            ambientFloor: session.floor(),
          },
          residue: session.stepCount(),
          completed: true,
        };
      }

      if (kind === "reset") {
        resetSession();
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: ["purpose-carry: session reset."],
          },
          residue: 0,
          completed: true,
        };
      }

      if (kind === "add") {
        const id = String(inst.id ?? "").trim();
        const content = String(inst.content ?? "");
        if (!id) throw new Error("kind=add requires a non-empty 'id'");
        const terms = Array.isArray(inst.terms)
          ? new Set(inst.terms.map((t) => String(t).toLowerCase()))
          : extractTerms(content);
        session.addStep({
          id,
          terms,
          cost: content.length > 0 ? Math.max(1, Math.ceil(content.length / 4)) : 1,
          timestamp: Date.now(),
          payload: { content },
        });
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: [
              `purpose-carry: added step "${id}" with ${terms.size} terms.`,
              `session now holds ${session.stepCount()} steps.`,
            ],
          },
          residue: session.stepCount(),
          completed: true,
        };
      }

      if (kind === "carry") {
        const goalTerms = resolveGoal(inst.goal);
        if (goalTerms.size === 0) {
          return {
            ok: false,
            output_delta: {
              kind: "text",
              lines: [
                "purpose-carry: goal is empty after τ-extraction.",
                "supply { goal: [\"term1\",\"term2\",...] } or a longer free-text goal.",
              ],
            },
            residue: 0,
            completed: true,
            error: "empty-goal",
          };
        }
        const budget = Number.isFinite(inst.budget) ? Number(inst.budget) : 2048;
        const carryResult = session.carry({
          goal: { terms: goalTerms },
          budget,
        });

        if (!carryResult.ok) {
          return {
            ok: false,
            output_delta: {
              kind: "text",
              lines: [
                `purpose-carry: ${carryResult.error.kind}` +
                  (carryResult.error.message
                    ? ` — ${carryResult.error.message}`
                    : ""),
              ],
            },
            residue: 0,
            completed: true,
            error: carryResult.error.kind,
          };
        }

        return {
          ok: true,
          output_delta: {
            kind: "purpose_carry",
            ...serializeCarry(carryResult),
            goal_terms: Array.from(goalTerms),
            budget,
            session_step_count: session.stepCount(),
          },
          // Residue = how much of the session's history the goal drew from.
          // Higher = more relevant history existed.
          residue: carryResult.keep.length,
          completed: true,
        };
      }

      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`purpose-carry: unknown kind "${kind}"`],
        },
        residue: 0,
        completed: true,
        error: `unknown kind "${kind}"`,
      };
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`purpose-carry error: ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }
  },

  outputCell(_instruction) {
    return { kind: "carry_cell" };
  },
};
