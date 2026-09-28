/* ============================================================================
 * Remote modules — acts forwarded to a Rust host's registry over
 * `POST /api/dispatch` on buhera-gateway (specification 07 §2).
 *
 * The adapter returns the remote ActResult verbatim (B2), adding only
 * `executed_on` and the remote act id to the delta so the local audit trail
 * shows where the work ran (B3). Transport failures are results, never
 * throws: unreachable / unauthorized / unroutable.
 * ========================================================================== */

import type { ActResult, Descriptor, Instruction, Module } from "../contract.ts";
import { errorText, fail } from "../contract.ts";

export interface GatewayTransport {
  /** Gateway origin, e.g. https://buhera.example. */
  baseUrl(): string;
  /** Session bearer token, or null when signed out. */
  token(): string | null;
  /** Injected for tests; defaults to global fetch. */
  fetch?: typeof fetch;
}

export function makeRemoteModule(descriptor: Omit<Descriptor, "binding">, transport: GatewayTransport): Module {
  const id = descriptor.id;
  return {
    id,
    describe: () => ({ ...descriptor, binding: "remote" }),
    async execute(instruction: Instruction, actBudget = 1): Promise<ActResult> {
      const token = transport.token();
      if (!token) return fail([`${id}: runs on the gateway — sign in first (dispatch("gateway", { kind: "login", … }))`], "remote unauthorized");
      let res: Response;
      try {
        res = await (transport.fetch ?? fetch)(`${transport.baseUrl().replace(/\/+$/, "")}/api/dispatch`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({ module: id, instruction, act_budget: actBudget }),
        });
      } catch (err) {
        return fail([`${id}: gateway unreachable — ${errorText(err)}`], "remote unreachable");
      }
      const body = (await res.json().catch(() => null)) as
        | { result?: ActResult; executed_on?: string; act_id?: number; error?: string }
        | null;
      if (res.status === 401) return fail([`${id}: gateway session expired — sign in again`], "remote unauthorized");
      if (!res.ok || !body?.result) {
        const why = body?.error ?? `HTTP ${res.status}`;
        return fail([`${id}: ${why}`], res.status === 404 ? "remote unknown module" : "remote unroutable");
      }
      const r = body.result;
      return {
        ...r,
        output_delta: r.output_delta ? { ...r.output_delta, executed_on: body.executed_on ?? "gateway", remote_act_id: body.act_id ?? null } : null,
      };
    },
    outputCell: () => ({ kind: `${id}_cell` }),
  };
}

/** Rust-only modules the TypeScript host reaches through the gateway. */
export const GATEWAY_MODULES: Array<Omit<Descriptor, "binding">> = [
  {
    id: "sbs-core",
    description:
      "sbs-core — the SBS observation calculus (Rust port, no DSL), executed on the Buhera gateway: S-entropy, " +
      "coherence R and flux visibility V for an explicit circuit, SBML, or the glycolysis demo.",
    instructions: ['dispatch("sbs-core", "demo")', 'dispatch("sbs-core", { kind: "observe", perturbations: [{ edge: 0, factor: 0.1 }] })'],
  },
];
