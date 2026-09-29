/* ============================================================================
 * Surface modules — what the top, right and bottom edges hold.
 *
 *   top      devices     the machines this session is open on (gateway)
 *            machine     this machine: cores, memory, GPU, display, storage
 *            network     (system-modules.js) connection, gateway, providers
 *            experiments shared dispatch scopes on the gateway
 *            runtime     the causal knowledge graph the runs are building
 *   right    preferences text size, spacing, column, motion
 *            screen      size, pixel ratio, colour depth, orientation
 *            code        what code steps show (the script, code blocks, trace)
 *            peripherals printer, pointer (the cut gesture), screen → image
 *   bottom   model       the personal model: providers, temperature, notes
 *            projects    the active project; groups (shared experiments)
 *            rag         what retrieval reads
 *            plans       scripts kept from runs, to run again
 *            reports     the account each completed run left
 *
 * Every one is an ordinary registered module, so an edge treats it like any
 * other: its glance opens a page. Settings sections return `controls` deltas —
 * live instruments bound to the settings store (lib/surface/settings.js) — and
 * readouts return `kv` rows, the generic shape.
 * ========================================================================== */

import { gatewayModule } from "@/lib/modules/gateway-module";
import { runtimeSnapshot } from "@/lib/surface/player";

const hasWindow = () => typeof window !== "undefined";
const done = (output_delta, ok = true) => ({ ok, output_delta, residue: ok ? 1 : 0, completed: true });
const kv = (title, rows) => ({ kind: "kv", title, rows: rows.filter(Boolean) });
const controls = (section, extra = {}) => ({ kind: "controls", section, ...extra });

function mod(id, description, instructions, execute) {
  return {
    id,
    describe: () => ({ id, description, instructions }),
    execute,
    outputCell: () => ({ kind: `${id}_cell` }),
  };
}

function fmtBytes(n) {
  if (typeof n !== "number" || !isFinite(n)) return "?";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

// A stable id for this browser, so the devices list can say which one is here.
export function thisDeviceId() {
  if (!hasWindow()) return "server";
  try {
    let id = window.localStorage.getItem("buhera.device.id");
    if (!id) {
      id = `browser-${Math.random().toString(36).slice(2, 10)}`;
      window.localStorage.setItem("buhera.device.id", id);
    }
    return id;
  } catch {
    return "browser";
  }
}

function gpuName() {
  try {
    const c = document.createElement("canvas");
    const gl = c.getContext("webgl2") || c.getContext("webgl");
    if (!gl) return "no WebGL";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  } catch {
    return "?";
  }
}

// ── top ──────────────────────────────────────────────────────────────────

export const devicesModule = mod(
  "devices",
  "Connected devices: the machines this OS session is open on. One account, one session, many machines — each dials the gateway; this browser is one of them.",
  ['dispatch("devices", "list")', 'dispatch("gateway", { kind: "pair", name: "<machine name>" })'],
  async () => {
    const here = kv("this device", [
      ["id", thisDeviceId()],
      hasWindow() && ["browser", navigator.userAgentData?.brands?.map((b) => b.brand).filter((b) => !/Not/.test(b)).join(", ") || navigator.userAgent.split(" ").slice(-1)[0]],
      hasWindow() && ["platform", navigator.userAgentData?.platform || navigator.platform || "?"],
    ]);
    const machines = await gatewayModule.execute("catalysts").catch((e) => done({ kind: "text", lines: [String(e)] }, false));
    return done({ kind: "stack", items: [here, machines.output_delta] });
  }
);

export const machineModule = mod(
  "machine",
  "Local machine: what this machine offers the session — processor cores, memory, graphics, display, storage.",
  ['dispatch("machine", "specs")'],
  async () => {
    if (!hasWindow()) return done(kv("machine", [["where", "server-side: no browser machine"]]));
    const rows = [
      ["processor cores", String(navigator.hardwareConcurrency ?? "?")],
      navigator.deviceMemory && ["memory", `≥ ${navigator.deviceMemory} GB (browser-reported, rounded)`],
      ["graphics", gpuName()],
      ["display", `${screen.width} × ${screen.height} @ ${window.devicePixelRatio}×`],
      ["platform", navigator.userAgentData?.platform || navigator.platform || "?"],
      ["language", navigator.language],
      ["time zone", Intl.DateTimeFormat().resolvedOptions().timeZone],
    ];
    try {
      const { usage, quota } = await navigator.storage.estimate();
      rows.push(["storage for this app", `${fmtBytes(usage)} used of ${fmtBytes(quota)}`]);
    } catch { /* not reported */ }
    try {
      if (navigator.getBattery) {
        const b = await navigator.getBattery();
        rows.push(["battery", `${Math.round(b.level * 100)}%${b.charging ? " · charging" : ""}`]);
      }
    } catch { /* not reported */ }
    return done(kv("this machine", rows));
  }
);

export const experimentsModule = mod(
  "experiments",
  "Experiments: shared dispatch scopes on the gateway. The owner grants other accounts in with a capped capability list; acts inside run against one shared federation and one audit log.",
  [
    'dispatch("experiments", "list")',
    'dispatch("gateway", { kind: "create_experiment", name: "<name>" })',
    'dispatch("gateway", { kind: "grant", experiment: "<id>", email: "<email>", capabilities: ["vahera"] })',
  ],
  async () => {
    const res = await gatewayModule.execute({ kind: "experiments" }).catch((e) => done({ kind: "text", lines: [String(e)] }, false));
    return res;
  }
);

export const runtimeModule = mod(
  "runtime",
  "Runtime: the causal knowledge graph the runs are building. Each run is a line; each thing a run described is a station (one node per τ — the same τ in two runs is one interchange); a station shows whether its chunks have emitted values.",
  ['dispatch("runtime", "map")', 'dispatch("ckg", { op: "report" })'],
  async () => done(await runtimeSnapshot())
);

// ── right ────────────────────────────────────────────────────────────────

export const preferencesModule = mod(
  "preferences",
  "Preferences: how the surface reads — text size, spacing, column width, motion.",
  [],
  async () => done(controls("preferences"))
);

export const screenModule = mod(
  "screen",
  "Screen: the display this surface is drawn on — size, window, pixel ratio, colour depth, orientation.",
  ['dispatch("screen", "size")'],
  async () => {
    if (!hasWindow()) return done(kv("screen", [["where", "server-side: no screen"]]));
    return done(kv("screen", [
      ["screen", `${screen.width} × ${screen.height}`],
      ["available", `${screen.availWidth} × ${screen.availHeight}`],
      ["window", `${window.innerWidth} × ${window.innerHeight}`],
      ["pixel ratio", String(window.devicePixelRatio)],
      ["colour depth", `${screen.colorDepth}-bit`],
      ["orientation", screen.orientation?.type || "?"],
      ["touch points", String(navigator.maxTouchPoints ?? 0)],
    ]));
  }
);

export const codeModule = mod(
  "code",
  "Code visibility: whether a step shows the script it ran, the code inside module results, and the player's retrieval and generation trace.",
  [],
  async () => done(controls("code"))
);

export const peripheralsModule = mod(
  "peripherals",
  "Device connections: printer, pointer (the cut gesture), and saving the screen as an image.",
  [],
  async () => done(controls("peripherals"))
);

// ── bottom ───────────────────────────────────────────────────────────────

export const modelModule = mod(
  "model",
  "Personalised model: which of this server's providers speak for you, how freely, and the standing notes your model reads with every request.",
  [],
  async () => {
    let available = [];
    try {
      const r = await fetch("/api/providers");
      available = (await r.json()).available || [];
    } catch { /* offline: the controls say so */ }
    return done(controls("model", { available }));
  }
);

export const projectsModule = mod(
  "projects",
  "Groups and projects: the active project names your retrieval receiver and scopes your plans and reports; groups are the shared experiments you belong to.",
  [],
  async () => {
    const groups = await gatewayModule.execute({ kind: "experiments" }).catch(() => null);
    return done(controls("projects", { groups: groups?.output_delta || null }));
  }
);

export const ragModule = mod(
  "rag",
  "RAG settings: the folders retrieval reads and the file kinds it takes. Retrieval is four-sided-triangle's individuator: it answers with a grounding status, never a bare passage.",
  [],
  async () => done(controls("rag"))
);

export const plansModule = mod(
  "plans",
  "Plans: the scripts runs were made of — written by you, or by your model from what you wrote. Run one again as a new step.",
  [],
  async () => done(controls("plans"))
);

export const reportsModule = mod(
  "reports",
  "Reports: what each completed run leaves — the request, the script, what retrieval grounded it on, and the graph it built.",
  [],
  async () => done(controls("reports"))
);

export const surfaceModules = [
  devicesModule, machineModule, experimentsModule, runtimeModule,
  preferencesModule, screenModule, codeModule, peripheralsModule,
  modelModule, projectsModule, ragModule, plansModule, reportsModule,
];

/** Ids of modules that configure the surface rather than do work. */
export const SURFACE_IDS = new Set([...surfaceModules.map((m) => m.id), "network", "disk", "config", "restart", "update"]);

