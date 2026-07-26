// ─────────────────────────────────────────────────────────────────────────────
// Node runner — executes the local `spraypaint` binary via child_process.
//
// Kept in a separate module so the rest of the package imports no Node built-ins
// and stays usable in a browser bundle (which would talk to a NodeRunner behind
// an HTTP route instead of importing this directly).
// ─────────────────────────────────────────────────────────────────────────────

import { execFile } from "node:child_process";
import type { SpraypaintRunner, RunOutput } from "./client.js";

export interface NodeRunnerOptions {
  /** Path to the binary. Defaults to "spraypaint" (resolved on PATH). */
  bin?: string;
  /** Working directory to run in. Defaults to process.cwd(). */
  cwd?: string;
  /** Max stdout/stderr buffer in bytes. Defaults to 32 MiB (large indexes). */
  maxBuffer?: number;
}

/** Runs the installed `spraypaint` executable locally. */
export class NodeRunner implements SpraypaintRunner {
  private readonly bin: string;
  private readonly cwd: string | undefined;
  private readonly maxBuffer: number;

  constructor(opts: NodeRunnerOptions = {}) {
    this.bin = opts.bin ?? "spraypaint";
    this.cwd = opts.cwd;
    this.maxBuffer = opts.maxBuffer ?? 32 * 1024 * 1024;
  }

  run(args: string[]): Promise<RunOutput> {
    return new Promise<RunOutput>((resolve) => {
      execFile(
        this.bin,
        args,
        { cwd: this.cwd, maxBuffer: this.maxBuffer },
        (err: Error | null, stdout: string, stderr: string) => {
          // execFile's err carries a numeric `code` on nonzero exit. We resolve
          // (never reject) so the client can inspect exitCode uniformly — some
          // subcommands (verify) exit nonzero as a legitimate verdict.
          const code = (err as unknown as { code?: unknown } | null)?.code;
          const exitCode =
            typeof code === "number" ? code : err ? 1 : 0;
          resolve({ stdout, stderr, exitCode });
        },
      );
    });
  }
}
