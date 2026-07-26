// ─────────────────────────────────────────────────────────────────────────────
// spraypaint client — invokes the REAL binary and parses its JSON.
//
// The client never simulates results. Every method shells out to the installed
// `spraypaint` executable and returns the parsed JSON, typed by types.ts. This
// is the whole point of the module living in buhera OS instead of a mock: the
// charts are drawn from genuine water-filling / χ / committed-count output.
//
// Runner abstraction: the binary may be local (Node child_process) or behind a
// service (HTTP). SpraypaintClient depends only on a Runner, so the same client
// code works in a Next.js route handler, a CLI, or a test with a fake runner.
// ─────────────────────────────────────────────────────────────────────────────

import {
  type AskQuery,
  type AskResult,
  type Identity,
  type CountResult,
  type SceneInfo,
  type VerifyResult,
  queryToArgs,
} from "./types.js";

/** Result of running the binary once. */
export interface RunOutput {
  stdout: string;
  stderr: string;
  exitCode: number;
}

/**
 * Executes `spraypaint <args>` somewhere and returns raw output. Implementations
 * decide *where* the binary runs; the client decides *what* to run and how to
 * interpret it.
 */
export interface SpraypaintRunner {
  run(args: string[]): Promise<RunOutput>;
}

/** Thrown when the binary exits nonzero or emits unparseable JSON. */
export class SpraypaintError extends Error {
  constructor(
    message: string,
    readonly args: string[],
    readonly output?: RunOutput,
  ) {
    super(message);
    this.name = "SpraypaintError";
  }
}

export interface SpraypaintClientOptions {
  /** Repo root passed as `--root`. Omit to let the binary walk up to .git/.spraypaint. */
  root?: string;
}

export class SpraypaintClient {
  constructor(
    private readonly runner: SpraypaintRunner,
    private readonly opts: SpraypaintClientOptions = {},
  ) {}

  // ── construction phase (Inv 4) ────────────────────────────────────────────

  /**
   * Build/rebuild the index. This is the exclusive construction phase — it emits
   * no ranked answer and does not touch the committed count.
   */
  async index(extra: string[] = []): Promise<void> {
    await this.exec(["index", ...extra], { parse: false });
  }

  // ── commitment phase (Inv 4) ──────────────────────────────────────────────

  /** Run one ask. Increments the committed count (Inv 2) unless `--dry-run`. */
  async ask(query: AskQuery): Promise<AskResult> {
    return this.exec<AskResult>(["ask", ...queryToArgs(query)]);
  }

  /**
   * Diagnostic ask: prints compiled query + score heads + p* + allocation WITHOUT
   * incrementing the count (a zero-act read-out emits no committed answer, Inv 3).
   * Returns the same shape as `ask` but `committed_count` is unchanged.
   */
  async dryRun(query: AskQuery): Promise<AskResult> {
    return this.exec<AskResult>(["ask", ...queryToArgs(query), "--dry-run"]);
  }

  // ── introspection ─────────────────────────────────────────────────────────

  /** Inv 1: self-graph fingerprint + χ. */
  async identity(): Promise<Identity> {
    return this.exec<Identity>(["identity", "--json"]);
  }

  /** Inv 2: never-resetting committed count. */
  async count(): Promise<CountResult> {
    return this.exec<CountResult>(["count", "--json"]);
  }

  /** Detected / overridden scenes with document + passage counts. */
  async scenes(): Promise<SceneInfo[]> {
    return this.exec<SceneInfo[]>(["scenes", "--json"]);
  }

  /** Re-check all four invariants. `pass:false` ⇒ a breach; the binary exits nonzero. */
  async verify(): Promise<VerifyResult> {
    // verify exits nonzero on breach BY DESIGN, so we tolerate a nonzero code
    // here and read the JSON verdict instead of throwing on it.
    const out = await this.runner.run(this.withRoot(["verify", "--json"]));
    return this.parseJson<VerifyResult>(out, ["verify", "--json"]);
  }

  // ── internals ─────────────────────────────────────────────────────────────

  private async exec<T>(
    args: string[],
    o: { parse?: boolean } = {},
  ): Promise<T> {
    const full = this.withRoot(args);
    const out = await this.runner.run(full);
    if (out.exitCode !== 0) {
      throw new SpraypaintError(
        `spraypaint ${args[0]} exited ${out.exitCode}: ${out.stderr.trim()}`,
        full,
        out,
      );
    }
    if (o.parse === false) return undefined as T;
    return this.parseJson<T>(out, full);
  }

  private parseJson<T>(out: RunOutput, args: string[]): T {
    try {
      return JSON.parse(out.stdout) as T;
    } catch {
      throw new SpraypaintError(
        `spraypaint ${args[0]} produced non-JSON output`,
        args,
        out,
      );
    }
  }

  private withRoot(args: string[]): string[] {
    if (this.opts.root) return [...args, "--root", this.opts.root];
    return args;
  }
}
