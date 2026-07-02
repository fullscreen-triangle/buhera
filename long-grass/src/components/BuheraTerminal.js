import { useEffect, useRef, useState } from "react";
import { Kernel } from "@/lib/kernel";
import { embedProtein } from "@/lib/substrate";
import { translate } from "@/lib/translator";
import { executeVahera } from "@/lib/vahera";
import { PROTEINS } from "@/lib/proteins";
import { run as runTurbulance, tbToString } from "@/lib/turbulance";
import { register, listModules, dispatch as dispatchModule, getAuditLog } from "@/lib/modules/registry";
import { vaheraModule } from "@/lib/modules/vahera-module";
import { echoModule } from "@/lib/modules/echo-module";
import { lavoisierModule } from "@/lib/modules/lavoisier-module";
// purposeModule intentionally not imported here — it pulls in server-only
// deps (fs, HF SDK). Wired in once /api/purpose-federation lands.
import { zangalewaModule } from "@/lib/modules/zangalewa-module";

// ────────────────────────────────────────────────────────────
//  Kernel boot.
// ────────────────────────────────────────────────────────────

function bootBlank() {
  return new Kernel(12);
}

function loadProteins(kernel) {
  for (const name of Object.keys(PROTEINS)) {
    const coord = embedProtein(name, PROTEINS[name]);
    kernel.allocate(coord, PROTEINS[name], {
      name,
      gene: PROTEINS[name].gene,
      kind: "protein",
    });
  }
}

// ────────────────────────────────────────────────────────────
//  Input router.
//
//  Returns { type, vahera?, meta? }
//    type === "vahera" — `vahera` is source to execute
//    type === "meta"   — `meta` is one of "tour" | "proteins" | "clear"
//                                       | "help"
//    type === "nl"     — caller should fall back to the NL translator
// ────────────────────────────────────────────────────────────

const VAHERA_PREFIXES = [
  "describe ",
  "resolve ",
  "spawn ",
  "navigate ",
  "complete ",
  "memory ",
  "demon ",
  "controller ",
  "kernel ",
  "process ",
];

// A line starting with one of these is a turbulance (kwasa-kwasa) script.
const TURBULANCE_PREFIXES = [
  "funxn ",
  "item ",
  "proposition ",
  "hypothesis ",
  "point ",
  "given ",
  "considering ",
  "within ",
  "research ",
  "for each ",
];

function stripQuotes(s) {
  const t = s.trim();
  if (t.length >= 2 && t.startsWith('"') && t.endsWith('"')) return t.slice(1, -1);
  return t;
}

export function routeInput(line) {
  const trimmed = line.trim();
  if (!trimmed) return { type: "noop" };

  const lower = trimmed.toLowerCase();

  // Meta commands.
  if (trimmed === ":quit" || trimmed === ":exit") return { type: "meta", meta: "quit" };
  if (trimmed === ":help") return { type: "meta", meta: "help" };
  if (trimmed === ":clear") return { type: "meta", meta: "clear" };
  if (trimmed === ":tour") return { type: "meta", meta: "tour" };
  if (trimmed === ":proteins") return { type: "meta", meta: "proteins" };
  if (trimmed === ":modules") return { type: "meta", meta: "modules" };
  if (trimmed === ":audit") return { type: "meta", meta: "audit" };

  // Turbulance script (kwasa-kwasa). Multi-line scripts are supported via
  // the textarea; a single-line input also routes here if it starts with
  // a turbulance keyword.
  if (TURBULANCE_PREFIXES.some((p) => lower.startsWith(p))) {
    return { type: "turbulance", source: trimmed };
  }

  // Already vaHera.
  if (VAHERA_PREFIXES.some((p) => lower.startsWith(p))) {
    return { type: "vahera", vahera: trimmed };
  }

  // store <name> = "<text>"
  if (lower.startsWith("store ")) {
    const rest = trimmed.slice(6);
    const eq = rest.indexOf("=");
    if (eq >= 0) {
      const name = rest.slice(0, eq).trim();
      const value = stripQuotes(rest.slice(eq + 1));
      if (name && value) {
        return { type: "vahera", vahera: `memory store "${name}" = "${value}"` };
      }
    }
  }

  // find "<text>" [k=N]
  if (lower.startsWith("find ")) {
    const rest = trimmed.slice(5);
    const m = rest.match(/\sk=(\d+)\s*$/);
    let k = 3;
    let text = rest;
    if (m) {
      k = parseInt(m[1], 10);
      text = rest.slice(0, m.index);
    }
    const t = stripQuotes(text.trim());
    return { type: "vahera", vahera: `memory find nearest "${t}" k=${k}` };
  }

  // dump <name>
  if (lower.startsWith("dump ")) {
    return { type: "vahera", vahera: `memory dump ${trimmed.slice(5).trim()}` };
  }

  const single = {
    list: "memory list",
    sort: "demon sort",
    stats: "kernel stats",
    trace: "kernel trace",
    procs: "process list",
    ps: "process list",
    verify: "controller verify",
  }[lower];
  if (single) return { type: "vahera", vahera: single };

  // Otherwise leave it for the NL translator (when proteins mode is
  // on) or treat as a search query.
  return { type: "nl", text: trimmed };
}

const TOUR_VAHERA = `
memory store "weekend"   = "I need to do laundry and clean the kitchen this weekend"
memory store "groceries" = "buy milk eggs bread and coffee from the supermarket"
memory store "exercise"  = "go for a run on Saturday morning before it gets hot"
memory store "travel"    = "book a flight to Munich for the conference next month"
memory store "code"      = "refactor the database connection pool to use async"
memory find nearest "shopping list" k=3
memory find nearest "morning workout" k=3
memory find nearest "flight to Germany" k=3
kernel stats
`.trim();

// ────────────────────────────────────────────────────────────
//  Welcome banner.
// ────────────────────────────────────────────────────────────

const WELCOME = `\
buhera-os web demo · in-browser kernel, no install

what it does
  files text by its categorical address — three numbers
  derived from the meaning of what you write. ask later,
  it finds what you stored, ranked by closeness.

quick try
  store note  = "remember to write up the proposal"
  store other = "buy milk and bread from the corner shop"
  find "writing"
  find "shopping"

other commands
  list                    show everything you've stored
  stats                   per-subsystem statistics
  dump <name>             show one object in detail
  sort                    zero-cost categorical sort
  :modules                list the federation
  :audit                  show recent acts
  :tour                   load five sample notes and search them
  :proteins               load a hardcoded biology database for the
                          natural-language demo
  :clear                  reset the kernel
  :help                   show every vaHera statement

you can also type vaHera directly:
  memory store "name" = "text"
  memory find nearest "query" k=5
  describe X with "text", spawn p from X, navigate to penultimate,
  complete trajectory, kernel stats, controller verify ...

or a turbulance (kwasa-kwasa) script:
  funxn double(x): return x * 2
  item r = double(21)
  proposition Greeting: motion Hello("world")

  // call any registered Buhera module:
  item out = dispatch("echo", "hello federation")
  print(out.output_delta.value)

  // route vaHera through the orchestrator:
  item r = dispatch("vahera", "memory store \"n\" = \"hi\"")

  // run a virtual mass-spec experiment (lavoisier):
  item ms = dispatch("lavoisier", "demo")
  print("records: {}", ms.output_delta.summary.count)
`;

const HELP = `\
vaHera statements (15):
  describe <name> with "<text>"
  resolve <name>
  spawn <program> from <name>
  navigate to penultimate
  complete trajectory
  memory create at S(<k>,<t>,<e>)
  memory store "<name>" = "<text>"
  memory find nearest "<text>" k=<n>
  memory list
  memory dump <name>
  demon sort
  controller verify
  kernel stats
  kernel trace
  process list

shortcuts:
  store <name> = "<text>"
  find "<text>" [k=N]
  list / dump <name> / sort / stats / trace / procs / verify

meta:
  :tour  :proteins  :clear  :help  :quit
`;

// ────────────────────────────────────────────────────────────
//  Artifact renderers.
// ────────────────────────────────────────────────────────────

function ProteinHeader({ name, p }) {
  return (
    <div className="flex items-baseline gap-6 mb-3 flex-wrap">
      <span className="text-white text-base">{name}</span>
      {p.gene && <span className="text-gray-500 text-xs">{p.gene}</span>}
      {p.uniprot && <span className="text-gray-600 text-xs">{p.uniprot}</span>}
      {p.length && <span className="text-gray-600 text-xs">{p.length} aa</span>}
      {p.role && <span className="text-gray-500 text-xs italic">{p.role}</span>}
    </div>
  );
}

function Field({ label, value }) {
  if (!value || (Array.isArray(value) && value.length === 0)) return null;
  return (
    <div className="flex mb-1">
      <span className="text-gray-500 w-28 shrink-0">{label}</span>
      <span className="text-gray-300 flex-1">
        {Array.isArray(value) ? value.join(", ") : value}
      </span>
    </div>
  );
}

function ArtifactProtein({ name, payload, aspect }) {
  const p = payload;
  if (aspect === "function") {
    return (<div><ProteinHeader name={name} p={p} /><Field label="function" value={p.function} /></div>);
  }
  if (aspect === "diseases") {
    return (<div><ProteinHeader name={name} p={p} /><Field label="diseases" value={p.diseases} /></div>);
  }
  if (aspect === "interacts") {
    return (<div><ProteinHeader name={name} p={p} /><Field label="interacts" value={p.interacts} /></div>);
  }
  if (aspect === "domains") {
    return (<div><ProteinHeader name={name} p={p} /><Field label="domains" value={p.domains} /></div>);
  }
  return (
    <div>
      <ProteinHeader name={name} p={p} />
      <Field label="function" value={p.function} />
      <Field label="domains" value={p.domains} />
      <Field label="diseases" value={p.diseases} />
      <Field label="interacts" value={p.interacts} />
      <Field label="pathway" value={p.pathway} />
      <Field label="location" value={p.localization} />
    </div>
  );
}

function ArtifactCompare({ a, b }) {
  const rows = [
    ["role", a.payload.role, b.payload.role],
    ["length", `${a.payload.length} aa`, `${b.payload.length} aa`],
    ["function", a.payload.function, b.payload.function],
    ["domains", a.payload.domains?.join(", "), b.payload.domains?.join(", ")],
    ["diseases", a.payload.diseases?.join(", "), b.payload.diseases?.join(", ")],
    ["pathway", a.payload.pathway, b.payload.pathway],
  ];
  return (
    <div>
      <div className="flex items-baseline gap-6 mb-4">
        <span className="text-white text-base">{a.name}</span>
        <span className="text-gray-500 text-xs">vs</span>
        <span className="text-white text-base">{b.name}</span>
      </div>
      {rows.map(([label, va, vb]) => (
        <div key={label} className="grid grid-cols-[7rem_1fr_1fr] gap-4 mb-2 text-xs">
          <span className="text-gray-500">{label}</span>
          <span className="text-gray-300">{va || "—"}</span>
          <span className="text-gray-300">{vb || "—"}</span>
        </div>
      ))}
    </div>
  );
}

function ArtifactFind({ query, items }) {
  if (!items.length) {
    return (<div className="text-gray-500 italic">no hits for &quot;{query}&quot;</div>);
  }
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">
        nearest to <span className="text-gray-300">&quot;{query}&quot;</span>
      </div>
      <ul className="text-gray-300">
        {items.map((it, i) => (
          <li key={i} className="py-1.5 grid grid-cols-[1.5rem_8rem_5rem_1fr] gap-3 items-baseline text-xs">
            <span className="text-gray-600">[{i + 1}]</span>
            <span className="text-gray-200 truncate">{it.name}</span>
            <span className="text-gray-600">d={it.distance.toFixed(3)}</span>
            <span className="text-gray-400 truncate">{it.source || ""}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactNote({ name, text, address, tier }) {
  return (
    <div>
      <div className="flex items-baseline gap-6 mb-2 flex-wrap">
        <span className="text-white text-base">{name}</span>
        <span className="text-gray-600 text-xs">addr {address.slice(0, 12)}</span>
        <span className="text-gray-600 text-xs">{tier}</span>
      </div>
      <div className="text-gray-300">{text}</div>
    </div>
  );
}

function ArtifactObjectList({ items }) {
  if (!items.length) {
    return <div className="text-gray-500 italic">memory is empty</div>;
  }
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">memory ({items.length})</div>
      <ul>
        {items.map((it, i) => (
          <li key={i} className="py-1 grid grid-cols-[9rem_8rem_4rem_1fr] gap-3 items-baseline text-xs">
            <span className="text-gray-600">{it.address.slice(0, 12)}</span>
            <span className="text-gray-200 truncate">{it.name}</span>
            <span className="text-gray-600">{it.tier}</span>
            <span className="text-gray-400 truncate">{it.source || ""}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactDump({ name, object }) {
  if (!object) {
    return <div className="text-gray-500 italic">dump {name}: not found</div>;
  }
  const o = object;
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">dump {name}</div>
      <Field label="address" value={o.address} />
      <Field label="coord" value={`S(${o.coord.k.toFixed(3)},${o.coord.t.toFixed(3)},${o.coord.e.toFixed(3)})`} />
      <Field label="tier" value={o.tier} />
      {typeof o.payload === "string" ? (
        <Field label="text" value={o.payload} />
      ) : (
        <div className="flex mb-1">
          <span className="text-gray-500 w-28 shrink-0">payload</span>
          <pre className="text-gray-300 flex-1 whitespace-pre-wrap font-mono text-xs">
            {JSON.stringify(o.payload, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function ArtifactStats({ stats }) {
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">kernel stats</div>
      <Field label="objects" value={String(stats.objects)} />
      <Field label="PVE ok" value={String(stats.pveOk)} />
      <Field label="PVE rejected" value={String(stats.pveRej)} />
      <Field label="TEM samples" value={String(stats.tem)} />
    </div>
  );
}

function ArtifactTrace({ log }) {
  if (!log.length) return <div className="text-gray-500 italic">no activity</div>;
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">activity ({log.length})</div>
      <ul className="text-gray-400 text-xs leading-relaxed font-mono">
        {log.slice(-20).map((l, i) => (<li key={i}>{l}</li>))}
      </ul>
    </div>
  );
}

function ArtifactSorted({ items }) {
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">sorted by S-distance to origin</div>
      <ul>
        {items.map((it, i) => (
          <li key={i} className="py-1 grid grid-cols-[2rem_9rem_8rem] gap-3 items-baseline text-xs">
            <span className="text-gray-600">{i + 1}</span>
            <span className="text-gray-600">{it.address.slice(0, 12)}</span>
            <span className="text-gray-200">{it.name}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactProcesses({ items }) {
  if (!items.length) return <div className="text-gray-500 italic">no processes</div>;
  return (
    <ul className="text-gray-300 text-xs font-mono">
      {items.map((p, i) => (
        <li key={i}>
          {p.name} <span className="text-gray-500">state={p.state}</span>
        </li>
      ))}
    </ul>
  );
}

function ArtifactVerify({ samples, message }) {
  return (
    <div>
      <div className="text-gray-300">{message}</div>
      <div className="text-gray-600 text-xs">samples observed: {samples}</div>
    </div>
  );
}

function ArtifactText({ lines }) {
  return (
    <div className="text-gray-300">
      {lines.map((line, i) => (<p key={i}>{line}</p>))}
    </div>
  );
}

function ArtifactTurbulance({ tb }) {
  if (!tb) return null;
  const lines = (tb.output || []).map((v) => {
    try { return tbToString ? tbToString(v) : String(v); }
    catch { return String(v); }
  });
  return (
    <div className="text-gray-300">
      {tb.ok === false && tb.error && (
        <p className="text-red-400 mb-2">
          turbulance error{tb.error.line ? ` (line ${tb.error.line})` : ""}: {tb.error.message}
        </p>
      )}
      {lines.length > 0 && (
        <pre className="font-mono text-sm whitespace-pre-wrap">{lines.join("\n")}</pre>
      )}
      {tb.propositions && tb.propositions.length > 0 && (
        <div className="mt-3 text-xs text-gray-500">
          <span className="text-gray-400">propositions: </span>
          {tb.propositions.map((p, i) => (
            <span key={i}>{p.name}{i < tb.propositions.length - 1 ? ", " : ""}</span>
          ))}
        </div>
      )}
      {tb.points && tb.points.length > 0 && (
        <div className="mt-1 text-xs text-gray-500">
          <span className="text-gray-400">points: </span>{tb.points.length}
        </div>
      )}
      {lines.length === 0 && !tb.error && (
        <p className="text-gray-500">(script ran; no output emitted)</p>
      )}
    </div>
  );
}

function ArtifactLavoisier({ summary, records, config }) {
  const [expanded, setExpanded] = useState(false);
  if (!summary) return null;

  const perClass = Object.entries(summary.perClass || {});
  const perAdduct = Object.entries(summary.perAdduct || {});
  const [mzLo, mzHi] = summary.mzRange || [0, 0];
  const [iLo, iHi] = summary.intensityRange || [0, 0];
  const fmt = (n) => (typeof n === "number" ? n.toFixed(4) : String(n));

  return (
    <div className="text-gray-300">
      <div className="mb-2">
        <span className="text-gray-400">
          {config?.experimentType || "run"}, {config?.analyser || "?"},
          polarity {config?.polarity || "?"}, CE{" "}
          {config?.collisionEnergy_eV ?? "?"} eV
        </span>
      </div>

      <div className="text-sm">
        <p>records: <span className="text-white">{summary.count}</span></p>
        <p>
          m/z range: <span className="text-white">{fmt(mzLo)} – {fmt(mzHi)}</span>
        </p>
        <p>
          intensity range:{" "}
          <span className="text-white">{fmt(iLo)} – {fmt(iHi)}</span>
        </p>
        <p>
          mean partition entropy:{" "}
          <span className="text-white">{fmt(summary.avgEntropy)}</span>
        </p>
      </div>

      {perClass.length > 0 && (
        <div className="mt-2 text-xs text-gray-500">
          <span className="text-gray-400">by class: </span>
          {perClass.map(([k, v], i) => (
            <span key={k}>
              {k}={v}
              {i < perClass.length - 1 ? ", " : ""}
            </span>
          ))}
        </div>
      )}
      {perAdduct.length > 0 && (
        <div className="mt-1 text-xs text-gray-500">
          <span className="text-gray-400">by adduct: </span>
          {perAdduct.map(([k, v], i) => (
            <span key={k}>
              {k}={v}
              {i < perAdduct.length - 1 ? ", " : ""}
            </span>
          ))}
        </div>
      )}
      {summary.shellsHistogram && summary.shellsHistogram.length > 0 && (
        <div className="mt-1 text-xs text-gray-500">
          <span className="text-gray-400">principal shells: </span>
          {summary.shellsHistogram.map((b, i) => (
            <span key={b.n}>
              n={b.n}:{b.count}
              {i < summary.shellsHistogram.length - 1 ? ", " : ""}
            </span>
          ))}
        </div>
      )}

      {records && records.length > 0 && (
        <div className="mt-3">
          <button
            onClick={() => setExpanded((e) => !e)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {expanded ? "hide records" : `show ${records.length} records`}
          </button>
          {expanded && (
            <pre className="mt-2 text-xs font-mono whitespace-pre-wrap text-gray-400">
              {records.slice(0, 50).map((r) =>
                `${(r.name || r.analyteClass || "?").padEnd(14)} ` +
                `${(r.adduct || "").padEnd(8)} ` +
                `m/z=${fmt(r.precursorMz)}  ` +
                `I=${fmt(r.intensity)}`
              ).join("\n")}
              {records.length > 50 ? `\n… and ${records.length - 50} more` : ""}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function ArtifactPurpose({ synthesis, model, federation, floor }) {
  return (
    <div className="text-gray-300">
      <div className="mb-2 text-xs text-gray-500">
        <span className="text-gray-400">model:</span> {model || "?"}
        {typeof floor === "number" && (
          <>
            {" · "}
            <span className="text-gray-400">floor:</span> {floor.toFixed(2)}
          </>
        )}
        {federation?.active_drafts?.length > 0 && (
          <>
            {" · "}
            <span className="text-gray-400">federation:</span>{" "}
            {federation.active_drafts.length} drafts
          </>
        )}
      </div>
      <pre className="whitespace-pre-wrap text-sm font-mono">{synthesis}</pre>
    </div>
  );
}

function Artifact({ result }) {
  if (!result) return null;
  switch (result.kind) {
    case "protein":         return <ArtifactProtein name={result.name} payload={result.payload} aspect={result.aspect} />;
    case "protein_compare": return <ArtifactCompare a={result.a} b={result.b} />;
    case "find":            return <ArtifactFind query={result.query} items={result.items} />;
    case "note":            return <ArtifactNote name={result.name} text={result.text} address={result.address} tier={result.tier} />;
    case "list_objects":    return <ArtifactObjectList items={result.items} />;
    case "dump":            return <ArtifactDump name={result.name} object={result.object} />;
    case "sorted_objects":  return <ArtifactSorted items={result.items} />;
    case "stats":           return <ArtifactStats stats={result.stats} />;
    case "trace":           return <ArtifactTrace log={result.log} />;
    case "processes":       return <ArtifactProcesses items={result.items} />;
    case "verify":          return <ArtifactVerify samples={result.samples} message={result.message} />;
    case "turbulance_result": return <ArtifactTurbulance tb={result.tb} />;
    case "lavoisier_run":   return <ArtifactLavoisier summary={result.summary} records={result.records} config={result.config} />;
    case "purpose_synthesis": return <ArtifactPurpose synthesis={result.synthesis} model={result.model} federation={result.federation} floor={result.floor} />;
    case "text":            return <ArtifactText lines={result.lines} />;
    case "list":            return <ArtifactFind query={result.title || ""} items={result.items} />;
    default:                return null;
  }
}

// ────────────────────────────────────────────────────────────
//  Welcome panel.
// ────────────────────────────────────────────────────────────

function WelcomePanel() {
  return (
    <pre className="text-gray-400 text-xs leading-relaxed whitespace-pre-wrap font-mono">
{WELCOME}
    </pre>
  );
}

// ────────────────────────────────────────────────────────────
//  Terminal.
// ────────────────────────────────────────────────────────────

export default function BuheraTerminal() {
  const kernelRef = useRef(null);
  const inputRef = useRef(null);
  const historyRef = useRef(null);
  const [entries, setEntries] = useState([]);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [proteinsMode, setProteinsMode] = useState(false);
  const [history, setHistory] = useState([]);
  const [cursor, setCursor] = useState(-1);

  useEffect(() => {
    kernelRef.current = bootBlank();
    // Register the federation.
    register(vaheraModule);
    register(echoModule);
    register(lavoisierModule);
    // purposeModule is server-side (reads knowledge packs via fs, calls HF).
    // Landing it needs an API route wrapper — register once /api/purpose-federation
    // exists. The adapter file is ready.
    register(zangalewaModule);
  }, []);

  useEffect(() => {
    if (inputRef.current) inputRef.current.focus();
    const onClick = () => inputRef.current && inputRef.current.focus();
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  useEffect(() => {
    if (historyRef.current) {
      historyRef.current.scrollTop = historyRef.current.scrollHeight;
    }
  }, [entries]);

  function pushEntry(entry) {
    setEntries((e) => [...e, { id: Date.now() + Math.random(), ...entry }]);
  }

  async function dispatch(text) {
    const route = routeInput(text);
    if (route.type === "noop") return;

    pushEntry({ obs: text, thinking: true });

    // Small artificial delay so the screen doesn't flicker.
    await new Promise((r) => setTimeout(r, 80));

    function patchLast(patch) {
      setEntries((es) => {
        const out = [...es];
        for (let i = out.length - 1; i >= 0; i--) {
          if (out[i].obs === text && out[i].thinking) {
            out[i] = { ...out[i], thinking: false, ...patch };
            break;
          }
        }
        return out;
      });
    }

    try {
      if (route.type === "meta") {
        if (route.meta === "help") {
          patchLast({ result: { kind: "text", lines: HELP.split("\n") } });
        } else if (route.meta === "clear") {
          kernelRef.current = bootBlank();
          setProteinsMode(false);
          patchLast({ result: { kind: "text", lines: ["(kernel reset)"] } });
        } else if (route.meta === "tour") {
          const out = executeVahera(TOUR_VAHERA, kernelRef.current, {
            useProteinDb: false,
            rerank: true,
          });
          patchLast({ multi: out.results });
        } else if (route.meta === "proteins") {
          if (!proteinsMode) {
            loadProteins(kernelRef.current);
            setProteinsMode(true);
          }
          patchLast({
            result: {
              kind: "text",
              lines: [
                "(proteins demo loaded; try \"tell me about TP53\" or \"compare BRCA1 and BRCA2\")",
              ],
            },
          });
        } else if (route.meta === "modules") {
          const mods = listModules();
          const lines = mods.length === 0
            ? ["(no modules registered)"]
            : mods.flatMap((m) => [
                `[${m.id}]${m.description ? "  " + m.description : ""}`,
                ...(m.instructions || []).map((i) => "    " + i),
              ]);
          patchLast({ result: { kind: "text", lines } });
        } else if (route.meta === "audit") {
          const log = getAuditLog().slice(-15);
          const lines = log.length === 0
            ? ["(audit log is empty)"]
            : log.map((e) => `#${e.act_id} ${e.module_id} (${e.wall_clock_ms}ms) — ${
                typeof e.instruction === "string"
                  ? e.instruction.slice(0, 60)
                  : "[non-string instruction]"
              }`);
          patchLast({ result: { kind: "text", lines } });
        } else if (route.meta === "quit") {
          patchLast({ result: { kind: "text", lines: ["(can't quit a browser tab from here)"] } });
        }
        return;
      }

      if (route.type === "vahera") {
        const out = executeVahera(route.vahera, kernelRef.current, {
          useProteinDb: proteinsMode,
          rerank: true,
        });
        if (out.results.length === 1) {
          patchLast({ result: out.results[0] });
        } else if (out.results.length > 1) {
          patchLast({ multi: out.results });
        } else if (out.lastResult) {
          patchLast({ result: out.lastResult });
        } else {
          patchLast({ result: { kind: "text", lines: ["ok"] } });
        }
        return;
      }

      if (route.type === "turbulance") {
        const tb = await runTurbulance(route.source);
        patchLast({ result: { kind: "turbulance_result", tb } });
        return;
      }

      // NL input.
      if (proteinsMode) {
        // Use the legacy translator for the proteins demo.
        const vh = translate(route.text);
        const out = executeVahera(vh, kernelRef.current, {
          useProteinDb: true,
          rerank: true,
        });
        if (out.lastResult) {
          patchLast({ result: out.lastResult });
        } else if (out.results.length) {
          patchLast({ multi: out.results });
        } else {
          patchLast({ result: { kind: "text", lines: ["no categorical match."] } });
        }
        return;
      }

      // Otherwise: bare line, no proteins mode → search.
      const safe = route.text.replace(/"/g, "'");
      const vh = `memory find nearest "${safe}" k=3`;
      const out = executeVahera(vh, kernelRef.current, {
        useProteinDb: false,
        rerank: true,
      });
      patchLast({ result: out.lastResult });
    } catch (err) {
      patchLast({ error: err.message || String(err) });
    }
  }

  async function submit(text) {
    if (!text.trim() || busy) return;
    setBusy(true);
    setDraft("");
    setHistory((h) => (h[h.length - 1] === text ? h : [...h, text]));
    setCursor(-1);

    await dispatch(text);

    setBusy(false);
    if (inputRef.current) inputRef.current.focus();
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(draft);
    } else if (e.key === "ArrowUp" && !draft.includes("\n")) {
      if (!history.length) return;
      e.preventDefault();
      const next = cursor < 0 ? history.length - 1 : Math.max(0, cursor - 1);
      setCursor(next);
      setDraft(history[next]);
    } else if (e.key === "ArrowDown" && cursor >= 0) {
      e.preventDefault();
      const next = cursor + 1;
      if (next >= history.length) {
        setCursor(-1);
        setDraft("");
      } else {
        setCursor(next);
        setDraft(history[next]);
      }
    }
  }

  function onInput(e) {
    setDraft(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = e.target.scrollHeight + "px";
  }

  const empty = entries.length === 0;

  return (
    <div className="fixed inset-0 bg-black text-gray-300 flex flex-col px-16 py-10 md:px-8 md:py-6 font-mono text-sm leading-relaxed">
      <div
        ref={historyRef}
        className="flex-1 overflow-y-auto pb-4"
        style={{ scrollbarWidth: "none" }}
      >
        {empty && (
          <div className="mb-8 animate-fade">
            <WelcomePanel />
          </div>
        )}
        {entries.map((e) => (
          <div key={e.id} className="mb-8 animate-fade">
            <div className="text-gray-200 whitespace-pre-wrap mb-2">{e.obs}</div>
            {e.thinking && <span className="text-gray-600 italic">...</span>}
            {e.error && <p className="text-gray-500">[{e.error}]</p>}
            {e.result && <Artifact result={e.result} />}
            {e.multi && (
              <div className="space-y-4">
                {e.multi.map((r, i) => (
                  <div key={i}><Artifact result={r} /></div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="flex items-start pt-3">
        <span className="text-gray-700 mr-2 select-none pt-0.5">{proteinsMode ? "🧬" : ""}</span>
        <textarea
          ref={inputRef}
          value={draft}
          onChange={onInput}
          onKeyDown={onKeyDown}
          rows={1}
          spellCheck={false}
          autoFocus
          placeholder={empty ? "type something to store, ask a question, or paste vaHera…" : ""}
          className="flex-1 bg-transparent border-none outline-none resize-none text-gray-200 font-mono text-sm leading-relaxed placeholder-gray-700"
          style={{ caretColor: "#2a9d8f" }}
        />
      </div>

      <style jsx>{`
        .animate-fade {
          animation: fade 0.3s ease;
        }
        @keyframes fade {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        div::-webkit-scrollbar { display: none; }
      `}</style>
    </div>
  );
}
