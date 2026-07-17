/* ============================================================================
 * Shared input runner.
 *
 * Given a string the user typed (a full terminal-style command), route it,
 * dispatch it, and return a normalized result the caller renders however they
 * like. Used by both BuheraTerminal and the RunnableCell in tutorial pages.
 *
 * Return shape (a normalized envelope, one of):
 *   { kind: "artifact", result }        — one Artifact-renderable result
 *   { kind: "multi", results }          — many Artifact-renderable results
 *   { kind: "text", lines }             — plain text
 *   { kind: "error", message }          — an error string
 *   { kind: "noop" }                    — empty input, do nothing
 *   { kind: "external", meta, message } — meta-command that navigates elsewhere
 * ========================================================================== */

import { routeInput } from "@/components/BuheraTerminal";
import { Kernel } from "@/lib/kernel";
import { executeVahera } from "@/lib/vahera";
import { translate } from "@/lib/translator";
import { run as runTurbulance } from "@/lib/turbulance";
import { listModules, dispatch as dispatchModule, getAuditLog } from "@/lib/modules/registry";

// The tour vahera script — kept in sync with the terminal's copy.
const TOUR_VAHERA = `
memory store "weekend"   = "I need to do laundry and clean the kitchen this weekend"
memory store "groceries" = "buy milk eggs bread and coffee from the supermarket"
memory store "exercise"  = "go for a run on Saturday morning before it gets hot"
memory store "travel"    = "book a flight to Munich for the conference next month"
memory store "code"      = "refactor the database connection pool to use async"
memory find nearest "shopping list" k=3
memory find nearest "morning workout" k=3
memory find nearest "flight to Germany" k=3
kernel stats
`.trim();

/**
 * Create a runtime context. Both the terminal and tutorial pages hold one
 * of these; they own the kernel and the proteins-mode flag.
 */
export function createRuntimeContext() {
  return {
    kernel: new Kernel(12),
    proteinsMode: false,
  };
}

/**
 * Dispatch a line of input against the given context. Returns a normalized
 * envelope; never throws (errors are returned as { kind: "error", ... }).
 *
 * @param {string} text
 * @param {object} ctx  the runtime context (kernel + flags)
 */
export async function runInput(text, ctx) {
  const route = routeInput(text);
  if (route.type === "noop") return { kind: "noop" };

  try {
    if (route.type === "meta") {
      if (route.meta === "help") {
        return { kind: "text", lines: ["(help — see the terminal for full HELP text)"] };
      }
      if (route.meta === "clear") {
        ctx.kernel = new Kernel(12);
        ctx.proteinsMode = false;
        return { kind: "text", lines: ["(kernel reset)"] };
      }
      if (route.meta === "tour") {
        const out = executeVahera(TOUR_VAHERA, ctx.kernel, { useProteinDb: false, rerank: true });
        return { kind: "multi", results: out.results };
      }
      if (route.meta === "modules") {
        const mods = listModules();
        const lines = mods.length === 0
          ? ["(no modules registered)"]
          : mods.flatMap((m) => [
              `[${m.id}]${m.description ? "  " + m.description : ""}`,
              ...(m.instructions || []).map((i) => "    " + i),
            ]);
        return { kind: "text", lines };
      }
      if (route.meta === "audit") {
        const log = getAuditLog().slice(-15);
        const lines = log.length === 0
          ? ["(audit log is empty)"]
          : log.map((e) => `#${e.act_id} ${e.module_id} (${e.wall_clock_ms}ms) — ${
              typeof e.instruction === "string"
                ? e.instruction.slice(0, 60)
                : "[non-string instruction]"
            }`);
        return { kind: "text", lines };
      }
      if (route.meta === "tutorials") {
        return { kind: "external", meta: "tutorials", message: "(already reading the tutorials)" };
      }
      if (route.meta === "proteins") {
        return {
          kind: "text",
          lines: [
            "(proteins demo — supported in the full terminal; skipped in this runner)",
          ],
        };
      }
      if (route.meta === "quit") {
        return { kind: "text", lines: ["(can't quit a browser tab from here)"] };
      }
      return { kind: "text", lines: [`(meta command "${route.meta}" not handled here)`] };
    }

    if (route.type === "vahera") {
      const out = executeVahera(route.vahera, ctx.kernel, {
        useProteinDb: ctx.proteinsMode,
        rerank: true,
      });
      if (out.results.length === 1) return { kind: "artifact", result: out.results[0] };
      if (out.results.length > 1)  return { kind: "multi", results: out.results };
      if (out.lastResult)          return { kind: "artifact", result: out.lastResult };
      return { kind: "text", lines: ["ok"] };
    }

    if (route.type === "turbulance") {
      const tb = await runTurbulance(route.source);
      return { kind: "artifact", result: { kind: "turbulance_result", tb } };
    }

    if (route.type === "scope_ctl") {
      const instr = route.ctl === "reset" ? { kind: "reset" } : { kind: "state" };
      const res = await dispatchModule("scope", instr);
      return { kind: "artifact", result: res.output_delta };
    }

    if (route.type === "scope") {
      const res = await dispatchModule("scope", route.source);
      return { kind: "artifact", result: res.output_delta };
    }

    if (route.type === "srn") {
      const res = await dispatchModule("srn", route.instruction);
      return { kind: "artifact", result: res.output_delta };
    }

    // NL input.
    if (ctx.proteinsMode) {
      const vh = translate(route.text);
      const out = executeVahera(vh, ctx.kernel, { useProteinDb: true, rerank: true });
      if (out.lastResult) return { kind: "artifact", result: out.lastResult };
      if (out.results.length) return { kind: "multi", results: out.results };
      return { kind: "text", lines: ["no categorical match."] };
    }

    // Bare line, no proteins mode → search.
    const safe = route.text.replace(/"/g, "'");
    const vh = `memory find nearest "${safe}" k=3`;
    const out = executeVahera(vh, ctx.kernel, { useProteinDb: false, rerank: true });
    return { kind: "artifact", result: out.lastResult };
  } catch (err) {
    return { kind: "error", message: err.message || String(err) };
  }
}
