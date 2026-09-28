/* ============================================================================
 * zangalewa-dsl — the OS's only AI module, reached from the browser through
 * the Zangalewa interceptor broker (specification specs/zangalewa-dsl.md).
 *
 * Binding: remote. The generate → validate → repair loop runs in the user's
 * local `zangalewa connect` agent (the Rust zangalewa-dsl crate, with the
 * user's own model keys). This module holds the pairing and forwards jobs as
 * opaque payloads through the broker; it never sees a model key and never
 * generates anything itself. Results come back as upstream's GenerateResult,
 * verbatim — every accepted chunk, never a picked winner.
 * ========================================================================== */

import type { ActResult, Instruction, Json, Module } from "../contract.ts";
import { done, errorText, fail, invalid } from "../contract.ts";

/** Structural view of the vendored interceptor client. */
export interface InterceptorClient {
  claim(pairCode: string): Promise<{ sessionId: string; agent?: unknown }>;
  status(): Promise<{ paired: boolean; online: boolean; agent: unknown }>;
  unpair(): void;
  run(payload: unknown, opts?: { pollMs?: number; maxWaitMs?: number }): Promise<unknown>;
}

export interface InterceptorClientCtor {
  new (opts: { baseUrl: string; storageKey?: string }): InterceptorClient;
}

export const ZANGALEWA_ID = "zangalewa-dsl";

/** Default broker address (upstream server.ts). */
export const DEFAULT_BROKER = "http://127.0.0.1:4319";

interface GenerateResult {
  ok: boolean;
  chunks?: unknown[];
  rejected?: unknown[];
  error?: string;
  stage?: "provider" | "compiler";
  [k: string]: unknown;
}

export function makeZangalewaModule(Client: InterceptorClientCtor, baseUrl = DEFAULT_BROKER): Module {
  let client = new Client({ baseUrl });
  return {
    id: ZANGALEWA_ID,
    describe: () => ({
      id: ZANGALEWA_ID,
      description:
        "Zangalewa — natural language → DSL chunks the owning compiler accepts. Runs in your local `zangalewa " +
        "connect` agent (your model keys), reached through the interceptor broker; pair once with a code.",
      instructions: [
        'dispatch("zangalewa-dsl", { kind: "pair", code: "ABCD-EFGH" })',
        'dispatch("zangalewa-dsl", { kind: "generate", dslId: "vahera", instructions: "store a note about Friday" })',
        'dispatch("zangalewa-dsl", "status")',
      ],
      binding: "remote",
    }),

    async execute(instruction: Instruction, actBudget = 1): Promise<ActResult> {
      const obj =
        typeof instruction === "string"
          ? { kind: instruction }
          : typeof instruction === "object" && instruction && !Array.isArray(instruction)
            ? instruction
            : null;
      const kind = obj?.["kind"];
      try {
        if (kind === "broker") {
          if (typeof obj?.["baseUrl"] !== "string") return invalid(ZANGALEWA_ID, '{ kind: "broker", baseUrl }');
          client = new Client({ baseUrl: obj["baseUrl"] });
          return done({ kind: "text", lines: [`zangalewa-dsl: broker set to ${obj["baseUrl"]}`] }, 0);
        }
        if (kind === "pair") {
          if (typeof obj?.["code"] !== "string") return invalid(ZANGALEWA_ID, '{ kind: "pair", code }');
          const s = await client.claim(obj["code"]);
          return done({ kind: "zangalewa_status", summary: "zangalewa-dsl: paired", paired: true, agent: (s.agent ?? null) as Json }, 0);
        }
        if (kind === "unpair") {
          client.unpair();
          return done({ kind: "zangalewa_status", summary: "zangalewa-dsl: unpaired", paired: false }, 0);
        }
        if (kind === "status") {
          const s = await client.status();
          return done({ kind: "zangalewa_status", summary: `zangalewa-dsl: ${s.online ? "online" : s.paired ? "paired, agent offline" : "not paired"}`, ...s } as never, 0);
        }
        if (kind === "generate") {
          if (typeof obj?.["dslId"] !== "string" || typeof obj?.["instructions"] !== "string") {
            return invalid(ZANGALEWA_ID, '{ kind: "generate", dslId, instructions, extent?, drafts?, maxRepairs?, model? }');
          }
          const { kind: _k, ...payload } = obj;
          // One act-budget unit = one draft (contract M6).
          const drafts = typeof payload["drafts"] === "number" ? payload["drafts"] : 1;
          payload["drafts"] = Math.min(drafts, Math.max(1, actBudget));
          const r = (await client.run(payload)) as GenerateResult;
          const accepted = r.chunks?.length ?? 0;
          const rejected = r.rejected?.length ?? 0;
          return {
            ok: r.ok,
            output_delta: {
              ...(r as Record<string, unknown>),
              kind: "zangalewa_generated",
              summary: `zangalewa-dsl: ${accepted} chunk(s) compiled, ${rejected} rejected`,
              retryable: r.stage === "provider",
              executed_on: "local agent",
            },
            residue: accepted + rejected === 0 ? 1 : rejected / (accepted + rejected),
            completed: true,
            ...(r.ok ? {} : { error: r.error ?? "no draft compiled" }),
          };
        }
        return invalid(ZANGALEWA_ID, "{ kind: pair|status|generate|unpair|broker, … }");
      } catch (err) {
        // Transport failures are results, never throws (spec 07, B2).
        return fail([`zangalewa-dsl: ${errorText(err)}`], "remote unreachable");
      }
    },

    outputCell: () => ({ kind: "zangalewa_cell" }),
  };
}
