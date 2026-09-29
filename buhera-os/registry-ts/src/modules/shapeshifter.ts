/* ============================================================================
 * shapeshifter — Shapeshifter (.ss), lavoisier's language for virtual
 * mass-spectrometry experiments (specification specs/shapeshifter.md). Wraps
 * the web interpreter (web/src/lib/shapeshifter: compileStage, executeStage),
 * vendored, over lavoisier's experiment/ and partition/ science code.
 *
 * Two honesty rules the engine does not enforce by itself:
 *  - executeStage catches a global runtime failure and returns an "empty"
 *    result; that is reported here as ok:false, never as a successful run.
 *  - residue counts what the run did NOT accomplish: warnings (unknown
 *    operations, skipped entities), unresolved pending db.* sentinels. It is
 *    not the number of workspace entries (that is output, not remaining work).
 *
 * No language entry is registered: the only admissible validator (L2: no
 * execution) is compileStage, and it accepts any text — garbage compiles with
 * warnings (paper Rem. 7.8; finding U-ss-1). A validator that cannot reject
 * would tell the generation loop every draft is valid. The module still
 * offers { kind: "compile" } for the structural warnings it does give.
 * ========================================================================== */

import type { ActResult, Instruction, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";

interface TermLine {
  stream: string;
  text: string;
}
interface Diagnostic {
  severity: string;
  message: string;
}
interface WorkspaceEntry {
  name: string;
  kind: string;
  value?: unknown;
  [key: string]: unknown;
}

/** The subset of the vendored Shapeshifter interpreter this adapter uses. */
export interface ShapeshifterEngine {
  compileStage(source: string): { ok: boolean; ast: unknown; term: TermLine[]; diagnostics: Diagnostic[] };
  executeStage(ast: unknown): { result: unknown; logs: Array<{ level: string; message: string }>; workspace: WorkspaceEntry[]; term: TermLine[] };
}

export const SHAPESHIFTER_ID = "shapeshifter";

/** Paper §9.1-style lipidomics run, then a partition field over its records. */
const DEMO_SOURCE = `objective demo:
  target: "PC lipids, small chain range, positive mode"

instrument orbi:
  kappa: 1e12
  ref_frequency: 10e6

phase acquire:
  records = lavoisier.instrument.run_experiment(classes: ["PC"], polarity: "+", analyser: "orbitrap", mz_window: [400, 1000])
  field = lavoisier.observe.partition_field(records: records)
`;

/** Timing lines ("compiled in 1.2 ms", "finished in …") make output nondeterministic; move them out. */
const TIMING = /\b(compiled|finished) in [\d.]+ ms\b/;

function splitTiming(term: TermLine[]): { term: TermLine[]; timing: string[] } {
  const timing: string[] = [];
  const kept = term.filter((l) => {
    if (TIMING.test(l.text)) {
      timing.push(l.text);
      return false;
    }
    return true;
  });
  return { term: kept, timing };
}

export function makeShapeshifterModule(engine: ShapeshifterEngine): Module {
  const expected = '.ss source, "demo", or { kind: "run" | "compile", source }';
  return {
    id: SHAPESHIFTER_ID,
    describe: () => ({
      id: SHAPESHIFTER_ID,
      description:
        "Shapeshifter — compile and run a .ss virtual mass-spectrometry experiment (objective / instrument / " +
        "target_list / phase / validate blocks) over lavoisier's instrument, partition, MS/MS and S-entropy " +
        "operations. Returns the terminal stream, the primary result and the workspace in declaration order.",
      instructions: [
        'dispatch("shapeshifter", "demo")',
        'dispatch("shapeshifter", { kind: "compile", source })',
        'dispatch("shapeshifter", "phase p:\\n  records = lavoisier.instrument.run_experiment(classes: [\\"PC\\"])")',
      ],
      binding: "native",
    }),

    async execute(instruction: Instruction): Promise<ActResult> {
      let kind = "run";
      let source: string | null = null;
      if (instruction == null || instruction === "" || instruction === "demo") source = DEMO_SOURCE;
      else if (typeof instruction === "string") source = instruction;
      else if (typeof instruction === "object" && !Array.isArray(instruction)) {
        if (typeof instruction["kind"] === "string") kind = instruction["kind"];
        if (typeof instruction["source"] === "string") source = instruction["source"];
      }
      if (source == null || !["run", "compile"].includes(kind)) return invalid(SHAPESHIFTER_ID, expected);

      let compiled;
      try {
        compiled = engine.compileStage(source);
      } catch (err) {
        return fail([`shapeshifter: ${errorText(err)}`], errorText(err));
      }
      const compileErrors = compiled.diagnostics.filter((d) => d.severity === "error");
      if (!compiled.ok || compileErrors.length) {
        const { term, timing } = splitTiming(compiled.term);
        return {
          ok: false,
          output_delta: { kind: "shapeshifter_run", ok: false, term, timing, workspace: [], diagnostics: compiled.diagnostics },
          residue: Math.max(1, compileErrors.length),
          completed: true,
          error: compileErrors[0]?.message ?? "compile failed",
        };
      }
      if (kind === "compile") {
        const { term, timing } = splitTiming(compiled.term);
        return done({ kind: "shapeshifter_run", ok: true, summary: "shapeshifter: compiles", term, timing, workspace: [], diagnostics: compiled.diagnostics }, 0);
      }

      let ran;
      try {
        ran = engine.executeStage(compiled.ast);
      } catch (err) {
        return fail([`shapeshifter: ${errorText(err)}`], errorText(err));
      }
      const { term, timing } = splitTiming([...compiled.term, ...ran.term]);
      const workspace = ran.workspace ?? [];
      const log = ran.logs ?? [];
      const runtimeError = ran.term.find((l) => l.stream === "stderr" && l.text.startsWith("error: runtime"));
      const warnings = log.filter((l) => l.level === "warn");
      const pending = workspace.filter((w) => w.kind === "pending");
      const delta = {
        kind: "shapeshifter_run",
        ok: !runtimeError,
        summary: runtimeError
          ? `shapeshifter: ${runtimeError.text}`
          : `shapeshifter: ${workspace.length} binding(s), ${warnings.length} warning(s), ${pending.length} pending`,
        result: ran.result,
        workspace,
        term,
        timing,
        log,
        pending: pending.map((p) => p.name),
        diagnostics: compiled.diagnostics,
      };
      if (runtimeError) {
        // A global failure (e.g. an empty resolved class set): nothing is licensed.
        return { ok: false, output_delta: delta, residue: 1, completed: true, error: runtimeError.text.replace(/^error: runtime\s*[—-]\s*/, "") };
      }
      // Residue: what the run did not accomplish — local failures it skipped
      // past (warnings) and network lookups it could not resolve (pending).
      return done(delta, warnings.length + pending.length);
    },

    outputCell: () => ({ kind: "shapeshifter_run_cell" }),
  };
}
