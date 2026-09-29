/* ============================================================================
 * System modules — the configuration and connectivity surface.
 *
 * Small browser-side modules the blank surface files under its edges:
 *
 *   network  (top)     is this instance online; can it reach the gateway;
 *                      which LLM providers the server has configured
 *   disk     (bottom)  browser storage quota/usage, and what occupies it
 *   config   (bottom)  where this instance points (gateway, broker), who is
 *                      signed in, how much of the federation is registered
 *   restart  (bottom)  a fresh kernel, or a reload of the whole app
 *   update   (bottom)  whether a newer build is being served, and applying it
 *
 * They report through the generic `kv` (label → value rows) and `text`
 * artifacts rather than bespoke renderers: the surface draws by result
 * shape, not by which module produced it.
 *
 * Everything here reads browser/host state that already exists. Nothing
 * stores content (empty-dictionary principle); restart and update are the
 * only acts with effects, and they only ever run when asked by name.
 * ========================================================================== */

import { listModules, clearAuditLog } from "@/lib/modules/registry";
import { gatewayTransport } from "@/lib/modules/gateway-module";
import { getKernel, resetKernel } from "@/lib/modules/vahera-module";

const hasWindow = () => typeof window !== "undefined";

function done(output_delta, ok = true) {
  return { ok, output_delta, residue: ok ? 1 : 0, completed: true };
}

function kv(title, rows) {
  return { kind: "kv", title, rows: rows.filter(Boolean) };
}

function text(...lines) {
  return { kind: "text", lines };
}

function kindOf(instruction, fallback) {
  if (typeof instruction === "string") return instruction.trim() || fallback;
  return (instruction && instruction.kind) || fallback;
}

function fmtBytes(n) {
  if (typeof n !== "number" || !isFinite(n)) return "?";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

async function fetchWithTimeout(url, ms, init = {}) {
  const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
  const timer = ctl ? setTimeout(() => ctl.abort(), ms) : null;
  const started = Date.now();
  try {
    const res = await fetch(url, { ...init, signal: ctl?.signal });
    return { ok: true, status: res.status, res, ms: Date.now() - started };
  } catch (err) {
    return { ok: false, error: ctl?.signal.aborted ? `no answer in ${ms} ms` : err?.message || String(err), ms: Date.now() - started };
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function providers() {
  const r = await fetchWithTimeout("/api/providers", 4000);
  if (!r.ok) return { error: r.error };
  try {
    return await r.res.json();
  } catch {
    return { error: `HTTP ${r.status}` };
  }
}

// --------------------------------------------------------------------------
// network
// --------------------------------------------------------------------------

export const networkModule = {
  id: "network",
  describe() {
    return {
      id: "network",
      description:
        "Connection status: whether this instance is online, whether the " +
        "Buhera gateway answers, and which LLM providers the server has.",
      instructions: ['dispatch("network", "status")'],
    };
  },
  async execute(instruction) {
    if (kindOf(instruction, "status") !== "status") {
      return done(text(`network: unknown instruction — try dispatch("network", "status")`), false);
    }
    const online = hasWindow() ? navigator.onLine : null;
    const conn = hasWindow() ? navigator.connection : null;
    const base = gatewayTransport.baseUrl();
    const gw = await fetchWithTimeout(base, 5000, { method: "GET", mode: "no-cors" });
    const prov = await providers();
    return done(
      kv("network", [
        ["online", online == null ? "?" : online ? "yes" : "no"],
        conn?.effectiveType && ["link", `${conn.effectiveType}${conn.downlink ? ` · ${conn.downlink} Mb/s` : ""}`],
        ["gateway", base],
        ["gateway answers", gw.ok ? `yes · ${gw.ms} ms` : `no · ${gw.error}`],
        ["LLM providers", prov.error ? `unknown · ${prov.error}` : (prov.available || []).join(", ") || "none configured"],
        !prov.error && ["active provider", prov.active || "—"],
      ])
    );
  },
  outputCell() { return { kind: "network_cell" }; },
};

// --------------------------------------------------------------------------
// disk
// --------------------------------------------------------------------------

export const diskModule = {
  id: "disk",
  describe() {
    return {
      id: "disk",
      description:
        "Disk space: this browser's storage quota and usage for the " +
        "instance, and what occupies local storage.",
      instructions: ['dispatch("disk", "usage")', 'dispatch("disk", "persist")'],
    };
  },
  async execute(instruction) {
    const kind = kindOf(instruction, "usage");
    if (!hasWindow()) return done(text("disk: no browser storage here"), false);

    if (kind === "persist") {
      if (!navigator.storage?.persist) return done(text("disk: this browser cannot pin storage"), false);
      const granted = await navigator.storage.persist();
      return done(text(granted
        ? "disk: storage pinned — the browser will not evict it under pressure."
        : "disk: the browser declined to pin storage (it may still keep it)."));
    }
    if (kind !== "usage") {
      return done(text(`disk: unknown instruction "${kind}" — try "usage" or "persist"`), false);
    }

    const rows = [];
    if (navigator.storage?.estimate) {
      const { usage, quota } = await navigator.storage.estimate();
      rows.push(["used", fmtBytes(usage)]);
      rows.push(["quota", fmtBytes(quota)]);
      if (usage != null && quota) rows.push(["free", `${fmtBytes(quota - usage)} (${(100 - (usage / quota) * 100).toFixed(1)}%)`]);
    } else {
      rows.push(["quota", "not reported by this browser"]);
    }
    if (navigator.storage?.persisted) {
      rows.push(["pinned", (await navigator.storage.persisted()) ? "yes" : "no"]);
    }
    try {
      const keys = Object.keys(window.localStorage);
      const sized = keys
        .map((k) => [k, (window.localStorage.getItem(k) || "").length * 2]) // UTF-16
        .sort((a, b) => b[1] - a[1]);
      const total = sized.reduce((s, [, b]) => s + b, 0);
      rows.push(["local storage", `${fmtBytes(total)} in ${keys.length} key${keys.length === 1 ? "" : "s"}`]);
      for (const [k, b] of sized.slice(0, 8)) rows.push([`  ${k}`, fmtBytes(b)]);
    } catch {
      rows.push(["local storage", "unreadable"]);
    }
    return done(kv("disk", rows));
  },
  outputCell() { return { kind: "disk_cell" }; },
};

// --------------------------------------------------------------------------
// config
// --------------------------------------------------------------------------

export const configModule = {
  id: "config",
  describe() {
    return {
      id: "config",
      description:
        "Setup configuration: where this instance points (gateway, broker), " +
        "who is signed in, the LLM providers, and the registered federation.",
      instructions: ['dispatch("config", "show")', 'dispatch("config", "modules")'],
    };
  },
  async execute(instruction) {
    const kind = kindOf(instruction, "show");
    const mods = listModules();

    if (kind === "modules") {
      return done(kv(`${mods.length} registered modules`, mods
        .map((m) => [m.id, m.description || ""])
        .sort((a, b) => a[0].localeCompare(b[0]))));
    }
    if (kind !== "show") {
      return done(text(`config: unknown instruction "${kind}" — try "show" or "modules"`), false);
    }

    let who = "not signed in";
    try {
      const raw = hasWindow() && window.localStorage.getItem("buhera.gateway.session");
      const s = raw ? JSON.parse(raw) : null;
      if (s?.email) who = s.email;
    } catch { /* noop */ }
    const prov = await providers();
    const kernel = getKernel();

    return done(
      kv("config", [
        ["gateway", gatewayTransport.baseUrl()],
        ["signed in as", who],
        ["zangalewa broker", process.env.NEXT_PUBLIC_ZANGALEWA_BROKER || "default"],
        ["LLM providers", prov.error ? `unknown · ${prov.error}` : (prov.available || []).join(", ") || "none configured"],
        prov.cascade_order && ["provider cascade", prov.cascade_order.join(" → ")],
        ["modules registered", String(mods.length)],
        ["kernel depth", String(kernel.depth)],
        ["kernel objects", String(kernel.store.size)],
        hasWindow() && window.__NEXT_DATA__?.buildId && ["build", window.__NEXT_DATA__.buildId],
      ])
    );
  },
  outputCell() { return { kind: "config_cell" }; },
};

// --------------------------------------------------------------------------
// restart
// --------------------------------------------------------------------------

export const restartModule = {
  id: "restart",
  describe() {
    return {
      id: "restart",
      description:
        "Restart: boot a fresh kernel (stored memory and the audit log are " +
        "dropped; pages are kept — they are snapshots), or reload the whole app.",
      instructions: ['dispatch("restart", "kernel")', 'dispatch("restart", "app")'],
    };
  },
  async execute(instruction) {
    const kind = kindOf(instruction, "");
    if (kind === "kernel") {
      const before = getKernel().store.size;
      resetKernel();
      clearAuditLog();
      return done(text(
        `kernel restarted — ${before} stored object${before === 1 ? "" : "s"} dropped, audit log cleared.`,
        "earlier pages are unchanged: they are snapshots of what was on screen."
      ));
    }
    if (kind === "app") {
      if (!hasWindow()) return done(text("restart: no app to reload here"), false);
      setTimeout(() => window.location.reload(), 300);
      return done(text("reloading the app — your pages will be here when it comes back."));
    }
    return done(text('restart: say what to restart — "kernel" or "app".'), false);
  },
  outputCell() { return { kind: "restart_cell" }; },
};

// --------------------------------------------------------------------------
// update
// --------------------------------------------------------------------------

// The build a deployment serves is named in its HTML (__NEXT_DATA__.buildId).
// Comparing the running build with the one the server hands out now is the
// whole update check: if they differ, a reload picks up the new build.
async function servedBuildId() {
  const r = await fetchWithTimeout("/", 6000, { cache: "no-store" });
  if (!r.ok) return { error: r.error };
  const html = await r.res.text();
  const m = html.match(/"buildId"\s*:\s*"([^"]+)"/);
  return m ? { id: m[1] } : { error: "the served page names no build" };
}

export const updateModule = {
  id: "update",
  describe() {
    return {
      id: "update",
      description:
        "Update: check whether the server is serving a newer build than the " +
        "one running here, and apply it by reloading.",
      instructions: ['dispatch("update", "check")', 'dispatch("update", "apply")'],
    };
  },
  async execute(instruction) {
    const kind = kindOf(instruction, "check");
    if (!hasWindow()) return done(text("update: nothing to update here"), false);
    const running = window.__NEXT_DATA__?.buildId || "unknown";

    if (kind === "apply") {
      setTimeout(() => window.location.reload(), 300);
      return done(text("reloading to pick up the served build — your pages are kept."));
    }
    if (kind !== "check") {
      return done(text(`update: unknown instruction "${kind}" — try "check" or "apply"`), false);
    }
    const served = await servedBuildId();
    const status = served.error
      ? `could not check · ${served.error}`
      : running === "development"
      ? "development build — updates arrive by hot reload"
      : served.id === running
      ? "up to date"
      : `a newer build is being served — apply to update`;
    return done(kv("update", [
      ["running build", running],
      ["served build", served.error ? "?" : served.id],
      ["status", status],
    ]));
  },
  outputCell() { return { kind: "update_cell" }; },
};

export const systemModules = [networkModule, diskModule, configModule, restartModule, updateModule];
