/* ============================================================================
 * The surface's resolver seam.
 *
 * The surface interprets one verb itself — "chart", which needs the page in
 * view and so cannot live anywhere else (see VIS_VERB below) — and routes the
 * work verbs (mail, find, plan, apphub; workVerb in verbs.js) straight to
 * their modules. Everything else
 * it never interprets: it hands the utterance —
 * together with the one page the user was looking at when they started
 * writing — to a resolver, and draws whatever envelope comes back as the next
 * page. The resolver is the intent layer (Layer 4 of the integration paper);
 * the real one (Zangalewa / the MSI / the LLM cascade) is being built
 * elsewhere and installs itself with setResolver(). Until then the default is
 * the existing line router (lib/runtime/run-input.js), which ignores the page.
 *
 * Resolver contract:
 *   (utterance: string, ctx: { runtime, page }) => Promise<Envelope>
 *     runtime  the kernel-owning runtime context (createSurfaceRuntime())
 *     page     pageContext() of the page being viewed, or null if blank —
 *              the only thing about the session the resolver may read
 *     Envelope one of run-input.js's envelopes; must not throw
 * ========================================================================== */

import { runInput } from "@/lib/runtime/run-input";
import { getKernel, replaceKernel } from "@/lib/modules/vahera-module";
import { getModule, dispatch as dispatchModule } from "@/lib/modules/registry";
import { glanceOf, edgeOf } from "@/lib/surface/edges";
import { visInstruction, workVerb } from "@/lib/surface/verbs";

/**
 * The runtime the surface resolves against. Unlike run-input's own
 * createRuntimeContext(), its kernel IS the vahera module's shared kernel, so
 * typed vaHera and dispatch("vahera", …) read and write the same memory.
 * `:clear` assigns `ctx.kernel = new Kernel(…)`; the setter installs that
 * kernel as the shared one.
 */
export function createSurfaceRuntime() {
  return {
    get kernel() { return getKernel(); },
    set kernel(k) { replaceKernel(k); },
    proteinsMode: false,
  };
}

// The surface's own verb (lib/surface/verbs.js): "chart" charts the page in
// view, or a named source. Everything else goes to the installed resolver.
async function chart(utterance, page) {
  const instruction = visInstruction(utterance, page);
  if (!instruction) return null;
  if (instruction.missing) {
    return {
      kind: "text",
      lines: [
        "nothing on screen to chart — flip to a page with data,",
        "or chart memory · chart audit · chart pages.",
      ],
    };
  }
  const res = await dispatchModule("vis", instruction);
  return res?.output_delta ? { kind: "artifact", result: res.output_delta } : { kind: "text", lines: ["(vis: no output)"] };
}

// The work verbs (mail, find, plan, apphub): each names a module and an
// instruction, so they resolve here, before the player sees the words.
async function work(utterance) {
  const verb = workVerb(utterance);
  if (!verb) return null;
  const res = await dispatchModule(verb.module, verb.instruction);
  return res?.output_delta ? { kind: "artifact", result: res.output_delta } : { kind: "text", lines: [`(${verb.module}: no output)`] };
}

async function lineResolver(utterance, { runtime }) {
  return runInput(utterance, runtime);
}

let _resolver = lineResolver;

/** Install the intent layer. Pass null to restore the line router. */
export function setResolver(fn) {
  _resolver = typeof fn === "function" ? fn : lineResolver;
}

/** Resolve one utterance to an envelope. Never throws. */
export async function resolve(utterance, ctx) {
  try {
    const charted = await chart(utterance, ctx?.page ?? null);
    if (charted) return charted;
    const worked = await work(utterance);
    if (worked) return worked;
    const env = await _resolver(utterance, ctx);
    return env || { kind: "text", lines: ["(no result)"] };
  } catch (err) {
    return { kind: "error", message: err?.message || String(err) };
  }
}

/**
 * Open a module picked from an edge: its self-description, plus the result of
 * its read-only glance instruction if it has one. Returns a "module"
 * envelope for the page. Never throws.
 */
export async function openModule(moduleId) {
  const mod = getModule(moduleId);
  if (!mod) return { kind: "error", message: `no module "${moduleId}" is registered` };

  let info;
  try {
    info = typeof mod.describe === "function" ? mod.describe() : { id: moduleId };
  } catch (err) {
    info = { id: moduleId, description: `(describe failed: ${err?.message || err})` };
  }

  const envelope = {
    kind: "module",
    module: {
      id: info.id || moduleId,
      description: info.description || "",
      instructions: Array.isArray(info.instructions) ? info.instructions.map(String) : [],
    },
    edge: edgeOf(moduleId),
    glance: null,
  };

  const glance = glanceOf(moduleId);
  if (glance !== undefined) {
    try {
      const res = await dispatchModule(moduleId, glance);
      envelope.glance = { ok: !!res?.ok, result: res?.output_delta ?? null };
    } catch (err) {
      envelope.glance = { ok: false, result: { kind: "text", lines: [err?.message || String(err)] } };
    }
  }
  return envelope;
}
