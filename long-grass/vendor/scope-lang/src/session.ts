// scope-lang — REPL session.
//
// A session is the sandbox's own semantics, fed incrementally. In the sandbox
// a whole `scope name { ... }` script runs the phases and produces charts. Here
// the same script is written in pieces: each cell contributes SCOPE declarations
// that accumulate into one growing program, and a cell produces a chart exactly
// when its code drives the program to a `visualise` step — the same thing that
// makes the sandbox emit a chart. Cells that only define (coordinate_space,
// channels, a morphism without visualise, goal, dispatch) merge into the session
// and acknowledge what they defined; no image is needed until a cell executes.

import { Parser, ParseError } from './compiler/parser';
import { typeCheck, formatErrors, formatWarnings } from './compiler/type-checker';
import type {
  ScopeProgram, MorphismDecl, ChannelsDecl, CoordinateSpaceDecl,
  GoalDecl, RuleDecl, DispatchDecl,
} from './compiler/ast';
import { runScope } from './runtime/runtime';
import type { ImagePayload, ScopeResult } from './runtime/runtime';

export type { ImagePayload, ScopeResult } from './runtime/runtime';

/** What a single evaluated cell produced. */
export interface CellResult {
  ok: boolean;
  /** 'chart' — an executing cell that ran the phases; 'define' — accumulated
   *  declarations; 'error' — parse or type error; 'noop' — empty input. */
  kind: 'chart' | 'define' | 'error' | 'noop';
  /** Human-readable console lines, formatted like the spec examples. */
  log: string[];
  /** Present iff kind === 'chart'. The full runtime result to render. */
  result?: ScopeResult;
  /** Names this cell added/replaced in the session (define cells). */
  defined?: string[];
  /** Present iff kind === 'error'. */
  error?: string;
}

/** Snapshot of what the session currently knows. */
export interface SessionState {
  hasCoordinateSpace: boolean;
  hasChannels: boolean;
  morphisms: string[];
  hasGoal: boolean;
  hasDispatch: boolean;
  hasImage: boolean;
}

const emptyProgram = (): ScopeProgram => ({
  kind: 'ScopeProgram',
  name: 'repl',
  rules: [],
  morphisms: [],
});

/** Does this accumulated program have a morphism that visualises? i.e. is there
 *  something to render? A cell is "executing" when, after its merge, the program
 *  reaches a VisualiseStep. */
function hasVisualise(program: ScopeProgram): boolean {
  return program.morphisms.some((m) =>
    m.expr.steps.some((s) => s.kind === 'VisualiseStep'),
  );
}

export class ScopeSession {
  /** The accumulated "script so far". */
  private program: ScopeProgram = emptyProgram();
  /** The linked image, if any. Only needed when a cell executes. */
  private image: ImagePayload | null = null;

  /** Link (or replace) the image future executing cells run against. */
  setImage(image: ImagePayload): void {
    this.image = image;
  }

  clearImage(): void {
    this.image = null;
  }

  /** Reset the whole session (definitions and image). */
  reset(): void {
    this.program = emptyProgram();
    this.image = null;
  }

  state(): SessionState {
    return {
      hasCoordinateSpace: !!this.program.coordinateSpace,
      hasChannels: !!this.program.channels,
      morphisms: this.program.morphisms.map((m) => m.name),
      hasGoal: !!this.program.goal,
      hasDispatch: !!this.program.dispatch,
      hasImage: !!this.image,
    };
  }

  /**
   * Evaluate one cell. Parses the fragment, type-checks the merge, folds its
   * declarations into the running program, and — if the merged program reaches
   * a visualise — runs the phases against the linked image and returns a chart.
   * Otherwise returns an acknowledgement of what was defined.
   */
  async run(cellSource: string): Promise<CellResult> {
    const src = cellSource.trim();
    if (!src) return { ok: true, kind: 'noop', log: [] };

    // ── Parse the cell as un-wrapped declarations ──────────────────────────
    let fragment: ScopeProgram;
    try {
      fragment = Parser.fromSource(src).parseFragment();
    } catch (err) {
      if (err instanceof ParseError) {
        return {
          ok: false,
          kind: 'error',
          error: err.message,
          log: [`[PARSE ERROR] line ${err.line} col ${err.col}\n  ${err.message}`],
        };
      }
      const msg = err instanceof Error ? err.message : String(err);
      return { ok: false, kind: 'error', error: msg, log: [`[PARSE ERROR] ${msg}`] };
    }

    // ── Merge onto a trial program and type-check the whole ────────────────
    // Merging first, then checking, means a cell is validated in the context of
    // everything defined before it — exactly as if the pieces were one script.
    const trial = this.merge(this.program, fragment);
    const { errors, warnings, program: resolved } = typeCheck(trial);

    const log: string[] = [...formatWarnings(warnings), ...formatErrors(errors)];

    if (errors.length > 0) {
      // Reject the cell; session state is unchanged. formatErrors() renders any
      // CompileError variant to a readable line; use the first as the summary.
      return {
        ok: false,
        kind: 'error',
        error: formatErrors(errors.slice(0, 1))[0] ?? 'type error',
        log,
      };
    }

    // Accept: the resolved (type-checked) program becomes the session.
    const defined = this.definedNames(fragment);
    this.program = resolved;

    // ── Execute iff the merged program reaches a visualise ─────────────────
    if (hasVisualise(this.program)) {
      if (!this.image) {
        return {
          ok: false,
          kind: 'error',
          error: 'no image linked',
          log: [
            ...log,
            'this cell produces a visual, but no image is linked.',
            'link one first (e.g. `:scope load <png-or-jpeg-url>`).',
          ],
        };
      }
      try {
        const result = await runScope(this.program, this.image);
        return { ok: true, kind: 'chart', log: [...log, ...(result.log ?? [])], result };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return { ok: false, kind: 'error', error: msg, log: [...log, `[RUN ERROR] ${msg}`] };
      }
    }

    // ── Define-only cell ───────────────────────────────────────────────────
    const ackLog =
      defined.length > 0
        ? [...log, `defined: ${defined.join(', ')}`]
        : [...log, 'ok'];
    return { ok: true, kind: 'define', log: ackLog, defined };
  }

  /** Fold a cell's declarations onto a base program. Last write wins:
   *  singleton blocks replace; morphisms replace by name. */
  private merge(base: ScopeProgram, frag: ScopeProgram): ScopeProgram {
    const morphisms: MorphismDecl[] = [...base.morphisms];
    for (const m of frag.morphisms) {
      const i = morphisms.findIndex((x) => x.name === m.name);
      if (i >= 0) morphisms[i] = m;
      else morphisms.push(m);
    }

    return {
      kind: 'ScopeProgram',
      name: base.name,
      channels: (frag.channels ?? base.channels) as ChannelsDecl | undefined,
      coordinateSpace:
        (frag.coordinateSpace ?? base.coordinateSpace) as CoordinateSpaceDecl | undefined,
      goal: (frag.goal ?? base.goal) as GoalDecl | undefined,
      rules: this.mergeRules(base.rules, frag.rules),
      morphisms,
      dispatch: (frag.dispatch ?? base.dispatch) as DispatchDecl | undefined,
    };
  }

  private mergeRules(base: RuleDecl[], frag: RuleDecl[]): RuleDecl[] {
    const out = [...base];
    for (const r of frag) {
      const i = out.findIndex((x) => x.name === r.name);
      if (i >= 0) out[i] = r;
      else out.push(r);
    }
    return out;
  }

  private definedNames(frag: ScopeProgram): string[] {
    const names: string[] = [];
    if (frag.coordinateSpace) names.push('coordinate_space');
    if (frag.channels) names.push('channels');
    if (frag.goal) names.push('goal');
    if (frag.dispatch) names.push('dispatch');
    for (const r of frag.rules) names.push(`rule ${r.name}`);
    for (const m of frag.morphisms) names.push(m.name);
    return names;
  }
}

/** Convenience factory. */
export function createSession(): ScopeSession {
  return new ScopeSession();
}
