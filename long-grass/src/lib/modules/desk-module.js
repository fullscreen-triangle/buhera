/* ============================================================================
 * Desk Module Adapter — the blank surface that holds the tagged intent.
 *
 * Buhera is a raw device: seven modules, each with its own DSL, dispatched
 * through one registry, every act appended to an audit log. `desk` is the one
 * thing a human actually talks to — the blank surface of the orchestra paper
 * ("an operating system for research that shows nothing until asked"). It holds
 * no domain competence of its own; it holds exactly ONE thing: the user's
 * standing *intent* — the reason the work is being done.
 *
 * The formal object it carries is the global goal g of
 * `semantic-categorical-orchestra.tex` (Def. "intent; the orchestrator") =
 * the conserved residual invariant of `instantiation-of-finite-weighted-
 * graphs.tex` (T3). The user *tags* it once; it outlives every individual
 * script. Every subsequent act dispatched through the federation is then scored
 * for its CONTRIBUTION toward g — the orchestra paper's necessity measure
 * (Thm. "contribution decides necessity"): a task is necessary iff its removal
 * worsens what the ensemble can reach toward g; contribution 0 ⇔ purposeless.
 *
 * What desk deliberately does NOT do (Principle "separation of competence"):
 *   • It does not judge correctness — that is each module's, via its DSL.
 *   • It does not re-run necessity as a hard gate / pruner — kwasa-kwasa owns
 *     the necessity gate; graffiti owns closure. desk only HOLDS the intent and
 *     REPORTS contribution, so the surface can show "what was necessary" and
 *     "what was merely correct" without ever redoing a module's work.
 *   • It does not store content (empty-dictionary principle). It holds the
 *     intent's TERMS (a coordinate) and act HANDLES (ids), never payloads.
 *
 * The scoring is fed by the registry's onDispatch hook, wired in the terminal's
 * mount effect exactly as the purpose-carry feeder is — see BuheraTerminal.js.
 *
 * Instruction shapes:
 *   • { kind: "tag", reason: string }   — set/replace the standing intent g.
 *       A plain string instruction is treated as { kind: "tag", reason }.
 *   • { kind: "surface" }               — the blank surface: the tagged reason,
 *       plus acts that contributed toward it (necessary) vs. did not
 *       (purposeless), read off the recorded acts since the tag.
 *   • { kind: "stats" }                 — { tagged, termCount, actsSeen, ... }
 *   • { kind: "clear" }                 — drop the intent, keep nothing.
 *   • { kind: "reset" }                 — alias for clear (parity with peers).
 * ========================================================================== */

import { extractTerms, extractTermsFromInstruction } from "@/lib/purpose-terms";

// --------------------------------------------------------------------------
// Module-private session. Per the trait guidance, state lives in a
// module-private variable, never on the trait object. Reset via
// { kind: "clear" | "reset" } or clearDesk().
//
// The session holds:
//   • intent:  { reason: string, terms: Set<string>, tagged_at: number } | null
//   • acts:    Array<{ act_id, module_id, terms: string[], contribution }>
//              — the acts observed since the current intent was tagged. Handles
//                only; no payloads. Reset whenever the intent is (re)tagged.
// --------------------------------------------------------------------------

function freshSession() {
  return { intent: null, acts: [] };
}

let _session = freshSession();

/** Drop the intent and all observed acts (used by clear/reset and tests). */
export function clearDesk() {
  _session = freshSession();
}

/** Read access for the terminal's surface renderer / tests. */
export function getDeskState() {
  return {
    intent: _session.intent
      ? {
          reason: _session.intent.reason,
          terms: Array.from(_session.intent.terms),
          tagged_at: _session.intent.tagged_at,
        }
      : null,
    acts: _session.acts.slice(),
  };
}

// --------------------------------------------------------------------------
// The contribution measure.
//
// orchestra `def:contribution`: δS(t) = S♭(E∖{t}) − S♭(E) ≥ 0, the increase in
// the ensemble floor caused by removing task t; > 0 ⇔ necessary, = 0 ⇔
// purposeless. We surface a computable proxy on the term-graph the rest of the
// federation already uses (see purpose-terms.js and the purpose feeder): a
// task's marginal reach toward g is the fraction of g's terms it covers that
// nothing else has to. The simple, monotone, ablation-free proxy is the
// overlap of the act's terms with the intent's terms, normalised by |g|.
//
//   contribution(t) = |terms(t) ∩ terms(g)| / |terms(g)|   ∈ [0, 1]
//
// contribution = 0 is exactly purposelessness (no overlap ⇒ removing t leaves
// the reach toward g unchanged); contribution > 0 marks a necessary act. This
// is a bounded, ordinal signal, faithful to the sign of δS, computed without
// re-executing any module (orchestra `thm:gate-orthogonal`).
// --------------------------------------------------------------------------

/**
 * Score an act's contribution toward the standing intent.
 * @param {Set<string>} actTerms
 * @param {Set<string>} goalTerms
 * @returns {number} contribution in [0, 1]
 */
export function contribution(actTerms, goalTerms) {
  if (!goalTerms || goalTerms.size === 0) return 0;
  let hit = 0;
  for (const g of goalTerms) if (actTerms.has(g)) hit++;
  return hit / goalTerms.size;
}

// --------------------------------------------------------------------------
// The dispatch observer.
//
// Called once per dispatched act (wired via registry.onDispatch in the terminal
// mount effect). It scores the act against the standing intent and records a
// handle. Best-effort and side-effect-free beyond the module's own session:
// a throw here is swallowed by the registry's hook isolation.
//
// We skip acts targeting `desk` itself (tagging/surfacing is not work toward g)
// and acts targeting `purpose-carry` (bookkeeping, mirrors that feeder's skip).
// --------------------------------------------------------------------------

const _selfSkip = new Set(["desk", "purpose-carry"]);

/**
 * Observe one audit-log entry. No-op until an intent is tagged.
 * @param {object} entry  the registry audit-log entry
 */
export function observeAct(entry) {
  if (!_session.intent) return; // nothing to gate toward yet
  if (!entry || _selfSkip.has(entry.module_id)) return;

  const actTerms = extractTermsFromInstruction(entry.instruction);
  const score = contribution(actTerms, _session.intent.terms);

  _session.acts.push({
    act_id: entry.act_id,
    module_id: entry.module_id,
    terms: Array.from(actTerms),
    contribution: score,
  });
}

// --------------------------------------------------------------------------
// Instruction normalisation.
// --------------------------------------------------------------------------

function normaliseInstruction(instruction) {
  if (typeof instruction === "string") {
    return { kind: "tag", reason: instruction };
  }
  return instruction || {};
}

// --------------------------------------------------------------------------
// The Module trait.
// --------------------------------------------------------------------------

export const deskModule = {
  id: "desk",

  describe() {
    return {
      id: "desk",
      description:
        "Desk: the blank surface. Holds the user's standing intent — the " +
        "reason the work is being done (the global goal g / conserved " +
        "residual) — and scores every subsequent act's contribution toward " +
        "it. Tag once; the reason outlives every script. Surfacing shows " +
        "which acts were necessary vs. merely correct.",
      instructions: [
        'dispatch("desk", "why you are doing this work")',
        'dispatch("desk", { kind: "tag", reason: "map the PC lipidome shift under heat stress" })',
        'dispatch("desk", { kind: "surface" })',
        'dispatch("desk", { kind: "stats" })',
        'dispatch("desk", { kind: "clear" })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    const inst = normaliseInstruction(instruction);
    const kind = inst.kind || "tag";

    try {
      // ── tag: set / replace the standing intent ─────────────────────────
      if (kind === "tag") {
        const reason = String(inst.reason ?? "").trim();
        if (!reason) {
          return badInstruction(
            "desk: tag requires a non-empty 'reason' — the standing intent g."
          );
        }
        const terms = extractTerms(reason);
        if (terms.size === 0) {
          return badInstruction(
            "desk: reason is empty after τ-extraction. Give a longer, more " +
              "specific reason (words < 3 chars and stopwords are dropped)."
          );
        }
        // Re-tagging is a new intent — a new collective, new residual
        // (orchestra T7 fission). Prior observations no longer apply.
        _session = {
          intent: { reason, terms, tagged_at: Date.now() },
          acts: [],
        };
        return {
          ok: true,
          output_delta: {
            kind: "desk_tag",
            reason,
            terms: Array.from(terms),
          },
          // Residue = size of the intent's coordinate. A richer reason
          // individuates a tighter goal region.
          residue: terms.size,
          completed: true,
        };
      }

      // ── surface: the blank surface, filled on demand ───────────────────
      if (kind === "surface") {
        if (!_session.intent) {
          return {
            ok: true,
            output_delta: {
              kind: "text",
              lines: [
                "desk: the surface is blank — no intent tagged.",
                'tag one with  dispatch("desk", "why you are doing this work")',
              ],
            },
            residue: 0,
            completed: true,
          };
        }

        const necessary = _session.acts.filter((a) => a.contribution > 0);
        const purposeless = _session.acts.filter((a) => a.contribution === 0);
        // Coverage: the fraction of the intent's terms touched by some act.
        const covered = new Set();
        for (const a of necessary) {
          for (const t of a.terms) {
            if (_session.intent.terms.has(t)) covered.add(t);
          }
        }
        const coverage =
          _session.intent.terms.size > 0
            ? covered.size / _session.intent.terms.size
            : 0;

        return {
          ok: true,
          output_delta: {
            kind: "desk_surface",
            reason: _session.intent.reason,
            goal_terms: Array.from(_session.intent.terms),
            covered_terms: Array.from(covered),
            coverage,
            // Ranked most-necessary first — the "important parts" of the
            // session, contribution being the necessity signal.
            necessary: necessary
              .slice()
              .sort((a, b) => b.contribution - a.contribution),
            purposeless,
            acts_seen: _session.acts.length,
          },
          residue: necessary.length,
          completed: true,
        };
      }

      // ── stats ──────────────────────────────────────────────────────────
      if (kind === "stats") {
        const necessaryCount = _session.acts.filter(
          (a) => a.contribution > 0
        ).length;
        return {
          ok: true,
          output_delta: {
            kind: "desk_stats",
            tagged: !!_session.intent,
            reason: _session.intent?.reason ?? null,
            term_count: _session.intent ? _session.intent.terms.size : 0,
            acts_seen: _session.acts.length,
            necessary: necessaryCount,
            purposeless: _session.acts.length - necessaryCount,
          },
          residue: _session.acts.length,
          completed: true,
        };
      }

      // ── clear / reset ──────────────────────────────────────────────────
      if (kind === "clear" || kind === "reset") {
        clearDesk();
        return {
          ok: true,
          output_delta: {
            kind: "text",
            lines: ["desk: surface cleared — intent dropped."],
          },
          residue: 0,
          completed: true,
        };
      }

      return badInstruction(`desk: unknown kind "${kind}"`);
    } catch (err) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [`desk error: ${err.message || String(err)}`],
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }
  },

  outputCell(_instruction) {
    return { kind: "desk_cell" };
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
