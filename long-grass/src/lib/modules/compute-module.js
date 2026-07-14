/* ============================================================================
 * Compute Module
 *
 * Remote dispatch — the mechanism that turns any Buhera-compatible URL into a
 * peer of the local federation. Buhera OS's job is to decide where each unit
 * of work runs; this module is how it actually sends that work somewhere.
 *
 * Instruction shape:
 *
 *   dispatch("compute", {
 *     target: "hetzner",              // catalyst name from the pool
 *     moduleId: "lavoisier",          // module to run remotely
 *     instruction: <anything>,        // what to send the remote module
 *     actBudget: 1,                   // optional
 *     timeoutMs: 30000,               // optional
 *   })
 *
 * Behaviour:
 *   1. Resolve the catalyst by name from the catalyst-registry.
 *   2. POST <catalyst.url>/api/dispatch  with { moduleId, instruction, actBudget }.
 *   3. Attach the catalyst's auth (bearer or basic) to the request.
 *   4. Return the remote ActResult wrapped so the terminal shows which catalyst
 *      answered and how long it took.
 *
 * A "compute" dispatch to target=<catalyst> ends up appearing in this Buhera's
 * audit log with module_id="compute"; the remote catalyst's audit log records
 * its own dispatch to the underlying moduleId. This is the correct behaviour:
 * each Buhera keeps its own history of decisions; the remote work is opaque
 * from the caller's side, transparent from the executor's side.
 * ========================================================================== */

import { getCatalyst } from "./catalyst-registry-module";

const DEFAULT_TIMEOUT_MS = 30_000;

function authHeader(entry) {
  const auth = entry.auth;
  if (!auth || auth.kind === "none") return null;
  if (auth.kind === "bearer") return `Bearer ${auth.token}`;
  if (auth.kind === "basic") {
    // browser-safe base64
    const raw = `${auth.username}:${auth.password}`;
    if (typeof btoa === "function") return `Basic ${btoa(raw)}`;
    if (typeof Buffer !== "undefined") return `Basic ${Buffer.from(raw).toString("base64")}`;
  }
  return null;
}

function withTimeout(promise, ms, tag) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`${tag} timed out after ${ms} ms`)), ms);
    promise.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); },
    );
  });
}

export const computeModule = {
  id: "compute",

  describe() {
    return {
      id: "compute",
      description:
        "Remote dispatch: run a module on a registered catalyst. The " +
        "catalyst can be any Buhera-compatible URL (Hetzner, Vultr, " +
        "Codespaces, Chromebook, another Vercel deployment).",
      instructions: [
        'dispatch("compute", { target: "hetzner", moduleId: "lavoisier", instruction: "demo" })',
        'dispatch("compute", { target: "codespaces", moduleId: "purpose-carry", instruction: { kind: "stats" } })',
        'dispatch("compute", { target: "vultr", moduleId: "echo", instruction: "hello", timeoutMs: 5000 })',
      ],
    };
  },

  async execute(instruction, _actBudget = 1) {
    if (!instruction || typeof instruction !== "object") {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [
            'compute: instruction must be { target, moduleId, instruction, actBudget?, timeoutMs? }',
          ],
        },
        residue: 0,
        completed: true,
        error: "bad-instruction",
      };
    }

    const target = String(instruction.target || "").trim();
    const moduleId = String(instruction.moduleId || "").trim();
    if (!target) {
      return {
        ok: false,
        output_delta: { kind: "text", lines: ["compute: target (catalyst name) is required"] },
        residue: 0,
        completed: true,
        error: "no-target",
      };
    }
    if (!moduleId) {
      return {
        ok: false,
        output_delta: { kind: "text", lines: ["compute: moduleId is required"] },
        residue: 0,
        completed: true,
        error: "no-module-id",
      };
    }

    const catalyst = getCatalyst(target);
    if (!catalyst) {
      return {
        ok: false,
        output_delta: {
          kind: "text",
          lines: [
            `compute: no catalyst named "${target}" in the pool.`,
            'run  dispatch("catalysts", "list")  to see what is registered.',
          ],
        },
        residue: 0,
        completed: true,
        error: "catalyst-not-found",
      };
    }

    const url = catalyst.url.replace(/\/$/, "") + "/api/dispatch";
    const headers = { "Content-Type": "application/json" };
    const auth = authHeader(catalyst);
    if (auth) headers.Authorization = auth;

    const timeoutMs = Number.isFinite(instruction.timeoutMs)
      ? Number(instruction.timeoutMs)
      : DEFAULT_TIMEOUT_MS;

    const started = Date.now();
    let res;
    try {
      res = await withTimeout(
        fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify({
            moduleId,
            instruction: instruction.instruction,
            actBudget: Number.isFinite(instruction.actBudget)
              ? Number(instruction.actBudget)
              : 1,
          }),
        }),
        timeoutMs,
        `compute → ${target}`,
      );
    } catch (err) {
      const elapsedMs = Date.now() - started;
      return {
        ok: false,
        output_delta: {
          kind: "remote_dispatch",
          target,
          moduleId,
          url,
          elapsed_ms: elapsedMs,
          error_stage: "network",
          error_message: err.message || String(err),
        },
        residue: 0,
        completed: true,
        error: err.message || String(err),
      };
    }

    const elapsedMs = Date.now() - started;

    let body;
    try { body = await res.json(); } catch { body = null; }

    if (!res.ok) {
      return {
        ok: false,
        output_delta: {
          kind: "remote_dispatch",
          target,
          moduleId,
          url,
          elapsed_ms: elapsedMs,
          status: res.status,
          error_stage: "upstream",
          error_message: body?.error || `HTTP ${res.status}`,
          remote_body: body,
        },
        residue: 0,
        completed: true,
        error: body?.error || `HTTP ${res.status}`,
      };
    }

    // Successful remote dispatch: body IS the remote ActResult. Wrap it so
    // the caller sees who answered.
    return {
      ok: !!body?.ok,
      output_delta: {
        kind: "remote_dispatch",
        target,
        moduleId,
        url,
        elapsed_ms: elapsedMs,
        status: res.status,
        remote: body,
      },
      residue: typeof body?.residue === "number" ? body.residue : 0,
      completed: true,
      error: body?.ok ? undefined : (body?.error || "remote returned ok=false"),
    };
  },

  outputCell() {
    return { kind: "compute_cell" };
  },
};
