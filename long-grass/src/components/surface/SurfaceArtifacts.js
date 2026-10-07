/* ============================================================================
 * SurfaceArtifacts — renderers for what the surface itself produces.
 *
 *   stack        several results, one after another
 *   controls     a settings section: live instruments bound to the settings
 *                store (lib/surface/settings.js) — preferences, code,
 *                peripherals, model, projects, rag, plans, reports
 *   runtime_map  the causal knowledge graph as a transit map
 *   player_run   one step of the player: the script, what grounded it, its
 *                results, and the line it laid in the runtime
 *   report_view  a kept report, opened as a page
 *   mail_*       accounts, a search, a message, a sync         (Mail.js)
 *   planning_*   the board, one item, what find found          (Planning.js)
 *   lattice_*    jobs on AppHub: tasks, plan, unit, results    (Jobs.js)
 *   web_*        search results, a page read, a site, the library (Reader.js)
 *   spec_*       a specification's model, diagrams, a comparison (Spec.js)
 *
 * <Artifact> delegates unknown kinds here and passes itself in as `Artifact`,
 * so nested results render with the same renderer without an import cycle.
 * Controls act on the settings store or start new steps through
 * SurfaceActions; they never change the page they sit on.
 * ========================================================================== */

import { useState } from "react";
import { useSettings, update } from "@/lib/surface/settings";
import { useSurfaceActions } from "@/components/surface/actions";
import RuntimeMap from "@/components/runtime/RuntimeMap";
import { Button, Choice, Row, Slider, Toggle, pct, when } from "@/components/surface/controls";
import { MailAccounts, MailMessage, MailSearch, MailSync } from "@/components/surface/Mail";
import { PlanningBoard, PlanningFind, PlanningItem } from "@/components/surface/Planning";
import { WebLibrary, WebPage, WebSearch, WebSite } from "@/components/surface/Reader";
import { SpecCompare, SpecDiagram, SpecModel } from "@/components/surface/Spec";
import { LatticeHome, LatticeLog, LatticePlan, LatticeResults, LatticeTasks, LatticeUnits, LatticeWrapped } from "@/components/surface/Jobs";

// ── sections ─────────────────────────────────────────────────────────────

function Preferences() {
  const { preferences: p } = useSettings();
  const set = (patch) => update("preferences", patch);
  return (
    <div>
      <Row label="text size" hint="scales everything on the surface">
        <Slider value={p.textScale} min={0.8} max={1.5} step={0.05} format={pct} onChange={(v) => set({ textScale: v })} />
      </Row>
      <Row label="spacing" hint="line height and the space between blocks">
        <Slider value={p.spacing} min={0.85} max={1.6} step={0.05} format={pct} onChange={(v) => set({ spacing: v })} />
      </Row>
      <Row label="column" hint="how wide a page's text runs">
        <Choice value={p.width} options={["narrow", "normal", "wide"]} onChange={(v) => set({ width: v })} />
      </Row>
      <Row label="motion" hint="page turns, drawer slides, flying marks">
        <Toggle on={p.motion} onChange={(v) => set({ motion: v })} />
      </Row>
    </div>
  );
}

function Code() {
  const { code: c } = useSettings();
  const set = (patch) => update("code", patch);
  return (
    <div>
      <Row label="the script a step ran" hint="the vaHera above each step's result — yours, or your model's">
        <Toggle on={c.showScript} onChange={(v) => set({ showScript: v })} />
      </Row>
      <Row label="code in results" hint="generated code inside module results (interceptor, DSL writer)">
        <Toggle on={c.showCodeBlocks} onChange={(v) => set({ showCodeBlocks: v })} />
      </Row>
      <Row label="the player's trace" hint="what retrieval returned and how the script was generated">
        <Toggle on={c.showTrace} onChange={(v) => set({ showTrace: v })} />
      </Row>
    </div>
  );
}

// The screen, as it is: the strip at its current scroll, cropped to the
// window. (A DOM clone loses scroll offsets, so the clone is laid out at full
// height and shifted up by the scroll, then cut to the viewport.)
async function captureScreen() {
  const { toPng } = await import("html-to-image");
  const strip = document.querySelector("[data-strip]");
  const node = strip || document.querySelector("[data-surface-root]") || document.body;
  const opts = { backgroundColor: "#000000", pixelRatio: window.devicePixelRatio || 1 };
  if (strip) {
    Object.assign(opts, {
      width: window.innerWidth,
      height: window.innerHeight,
      style: {
        transform: `translateY(-${strip.scrollTop}px)`,
        height: `${strip.scrollHeight}px`,
        overflow: "visible",
        inset: "auto",
        top: "0",
        left: "0",
        width: `${window.innerWidth}px`,
      },
    });
  }
  const url = await toPng(node, opts);
  const a = document.createElement("a");
  a.href = url;
  a.download = `buhera-screen-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
  a.click();
}

function Peripherals() {
  const { pointer: p } = useSettings();
  const [status, setStatus] = useState("");
  return (
    <div>
      <Row label="printer" hint="prints what is on the screen">
        <Button onClick={() => window.print()}>print the screen</Button>
      </Row>
      <Row label="screen → image" hint="saves the screen as a PNG">
        <span className="inline-flex items-center gap-3">
          <Button onClick={async () => {
            setStatus("saving…");
            try { await captureScreen(); setStatus("saved"); } catch (e) { setStatus(`failed: ${e.message || e}`); }
          }}>save as image</Button>
          <span className="text-xs text-gray-500">{status}</span>
        </span>
      </Row>
      <Row label="pointer: cut with the wheel" hint="press the scroll wheel and drag over a frame to lift that part onto the screen">
        <Toggle on={p.cutWithWheel} onChange={(v) => update("pointer", { cutWithWheel: v })} />
      </Row>
      <Row label="pointer: cut with Alt" hint="Alt + drag does the same — for trackpads">
        <Toggle on={p.cutWithAlt} onChange={(v) => update("pointer", { cutWithAlt: v })} />
      </Row>
    </div>
  );
}

function Model({ available = [] }) {
  const { model: m } = useSettings();
  const set = (patch) => update("model", patch);
  const chosen = new Set(m.providers);
  const toggle = (p) => {
    const next = new Set(chosen);
    next.has(p) ? next.delete(p) : next.add(p);
    set({ providers: [...next] });
  };
  return (
    <div>
      <Row label="providers" hint="which of this server's models draft your scripts; none chosen = all of them, in federation">
        {available.length === 0 ? (
          <span className="text-xs text-gray-500">no model is configured on this server</span>
        ) : (
          <span className="inline-flex flex-wrap gap-2">
            {available.map((p) => (
              <Toggle key={p} on={chosen.has(p)} onChange={() => toggle(p)} label={p} />
            ))}
          </span>
        )}
      </Row>
      <Row label="temperature" hint="how freely your model writes">
        <Slider value={m.temperature} min={0} max={1.2} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ temperature: v })} />
      </Row>
      <Row label="standing notes" hint="read by your model with every request">
        <textarea defaultValue={m.instructions} rows={3} spellCheck={false}
          onBlur={(e) => set({ instructions: e.target.value })}
          className="w-full bg-transparent border border-gray-800 focus:border-gray-600 rounded p-2 text-xs text-gray-200 outline-none" />
      </Row>
    </div>
  );
}

function Projects({ groups, Artifact }) {
  const s = useSettings();
  const [name, setName] = useState("");
  const known = [...new Set(["default", s.project, ...s.plans.map((p) => p.project), ...s.reports.map((r) => r.project)].filter(Boolean))];
  return (
    <div>
      <Row label="active project" hint="names your retrieval receiver and scopes your plans and reports">
        <Choice value={s.project} options={known} onChange={(v) => update("project", v)} />
      </Row>
      <Row label="new project">
        <span className="inline-flex gap-2">
          <input value={name} onChange={(e) => setName(e.target.value)} spellCheck={false} placeholder="name"
            className="bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-xs text-gray-200 px-1 w-40" />
          <Button disabled={!name.trim()} onClick={() => { update("project", name.trim()); setName(""); }}>create and switch</Button>
        </span>
      </Row>
      <div className="mt-6 text-xs text-gray-600 mb-2">groups — the shared experiments you belong to</div>
      {groups ? <Artifact result={groups} /> : <p className="text-xs text-gray-500">sign in to the gateway to see your groups.</p>}
    </div>
  );
}

function Rag() {
  const { rag: r } = useSettings();
  const [folder, setFolder] = useState("");
  const set = (patch) => update("rag", patch);
  return (
    <div>
      <Row label="retrieval" hint="off: your model writes from your words alone">
        <Toggle on={r.enabled} onChange={(v) => set({ enabled: v })} />
      </Row>
      <Row label="folders" hint="read by this server; a folder named here is read only when the server runs on this machine">
        <div>
          {r.folders.length === 0 && <div className="text-xs text-gray-500 mb-1">none yet</div>}
          {r.folders.map((f) => (
            <div key={f} className="flex items-center gap-2 text-xs text-gray-300">
              <span className="font-mono break-all">{f}</span>
              <button type="button" className="text-gray-600 hover:text-rose-300" onClick={() => set({ folders: r.folders.filter((x) => x !== f) })}>×</button>
            </div>
          ))}
          <span className="inline-flex gap-2 mt-2">
            <input value={folder} onChange={(e) => setFolder(e.target.value)} spellCheck={false} placeholder="C:\path\to\notes"
              className="bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-xs text-gray-200 px-1 w-72 font-mono" />
            <Button disabled={!folder.trim()} onClick={() => { set({ folders: [...new Set([...r.folders, folder.trim()])] }); setFolder(""); }}>add</Button>
          </span>
        </div>
      </Row>
      <Row label="file kinds" hint="comma-separated extensions">
        <input defaultValue={r.extensions.join(", ")} spellCheck={false}
          onBlur={(e) => set({ extensions: e.target.value.split(",").map((x) => x.trim()).filter(Boolean).map((x) => (x.startsWith(".") ? x : `.${x}`)) })}
          className="bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-xs text-gray-200 px-1 w-72 font-mono" />
      </Row>
    </div>
  );
}

function Plans() {
  const s = useSettings();
  const actions = useSurfaceActions();
  const mine = s.plans.filter((p) => p.project === s.project);
  if (!mine.length) return <p className="text-gray-500 italic">no plans in “{s.project}” yet — every script a step runs is kept here.</p>;
  return (
    <div className="space-y-4">
      {mine.map((p) => (
        <div key={p.id} className="border-t border-gray-900 pt-2">
          <div className="flex flex-wrap items-baseline gap-3 text-xs">
            <span className="text-gray-600">{when(p.at)}</span>
            <span className="text-gray-500">{p.by === "model" ? "your model wrote it" : "you wrote it"}{p.dsl ? ` · ${p.dsl}` : ""}</span>
            {actions && <Button onClick={() => actions.write(p.script)}>run again</Button>}
            {actions && <Button onClick={() => actions.draft(p.script)}>edit first</Button>}
          </div>
          <div className="text-sm text-gray-300 mt-1">{p.source}</div>
          {p.script !== p.source && s.code.showScript && (
            <pre className="mt-1 text-xs font-mono text-teal-100/80 whitespace-pre-wrap">{p.script}</pre>
          )}
        </div>
      ))}
    </div>
  );
}

function Reports() {
  const s = useSettings();
  const actions = useSurfaceActions();
  const mine = s.reports.filter((r) => r.project === s.project);
  if (!mine.length) return <p className="text-gray-500 italic">no reports in “{s.project}” yet — each completed run leaves one.</p>;
  return (
    <div className="space-y-3">
      {mine.map((r) => (
        <div key={r.id} className="border-t border-gray-900 pt-2">
          <div className="flex flex-wrap items-baseline gap-3 text-xs">
            <span className="text-gray-600">{when(r.at)}</span>
            <span className="text-gray-500">{r.summary}</span>
            {actions && (
              <Button onClick={() => actions.derive(`report: ${r.source}`, async () => ({ kind: "artifact", result: { kind: "report_view", report: r } }))}>
                open
              </Button>
            )}
          </div>
          <div className="text-sm text-gray-300 mt-1">{r.source}</div>
        </div>
      ))}
    </div>
  );
}

// ── player run & report ──────────────────────────────────────────────────

const STATUS_WORD = {
  grounded: "grounded (three or more independent sources agree)",
  "single-sourced": "single-sourced",
  "two-sourced": "two-sourced",
  contested: "contested — your sources disagree",
  declined: "declined",
};

function Grounding({ retrieval, full }) {
  if (!retrieval) return null;
  return (
    <div className="text-xs text-gray-500 mb-3">
      <span className="text-gray-400">grounding:</span> {STATUS_WORD[retrieval.status] || retrieval.status}
      {retrieval.support?.length ? ` · ${retrieval.support.length} source${retrieval.support.length === 1 ? "" : "s"}` : ""}
      {retrieval.reason ? ` — ${retrieval.reason}` : ""}
      {retrieval.refused ? ` · ${retrieval.refused}` : ""}
      {full && retrieval.claim && <div className="mt-1 text-gray-400 whitespace-pre-wrap">{retrieval.claim}</div>}
      {full && retrieval.support?.map((c, i) => (
        <div key={i} className="font-mono text-gray-600">{c.source} · power {c.power}</div>
      ))}
      {full && retrieval.warning && <div className="text-yellow-500/80">{retrieval.warning}</div>}
    </div>
  );
}

function Script({ script, by }) {
  return (
    <div className="mb-4">
      <div className="text-[11px] text-gray-600 mb-1">{by === "model" ? "the script your model wrote" : "your script"}</div>
      <pre className="p-3 rounded border border-gray-900 bg-white/[0.02] text-xs font-mono text-teal-100/90 whitespace-pre-wrap">{script}</pre>
    </div>
  );
}

function PlayerRun({ r, Artifact }) {
  const { code } = useSettings();
  if (!r.ok) {
    return (
      <div>
        <p className="text-rose-300/80 mb-2">[{r.stage}] {r.error}</p>
        <Grounding retrieval={r.retrieval} full />
        {r.script && <Script script={r.script} by="model" />}
      </div>
    );
  }
  const onlyRun = r.run ? [r.run] : [];
  const graph = r.graph && r.run ? { ...r.graph, nodes: r.graph.nodes.filter((n) => r.run.taus.includes(n.tau)) } : null;
  return (
    <div>
      {r.by === "model" && <Grounding retrieval={r.retrieval} full={code.showTrace} />}
      {code.showTrace && r.generation && (
        <div className="text-xs text-gray-600 mb-3">
          model: {r.generation.providers?.join(", ")} · {r.generation.attempts} attempt(s), {r.generation.repairs} repair(s)
          {r.generation.confidence != null ? ` · confidence ${r.generation.confidence}` : ""}
        </div>
      )}
      {code.showScript && <Script script={r.script} by={r.by} />}
      {r.error && <p className="text-rose-300/80 mb-3">[{r.error}]</p>}
      <div className="space-y-5">
        {(r.results || []).map((res, i) => <div key={i}><Artifact result={res} /></div>)}
      </div>
      {r.run && (r.run.taus.length > 0 || r.run.spawned.length > 0) && (
        <div className="mt-8">
          <div className="text-[11px] text-gray-600 mb-2">the line this run laid in the runtime</div>
          <RuntimeMap runs={onlyRun} graph={graph} showDetails={false} />
          {r.run.spawned.filter((s) => !s.attached).map((s, i) => (
            <div key={i} className="text-xs text-gray-500">spawn {s.program} from {s.target} — {s.note}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function ReportView({ report: r }) {
  return (
    <div>
      <div className="text-xs text-gray-600 mb-3">{when(r.at)} · project {r.project} · {r.summary}</div>
      <div className="text-sm text-gray-200 mb-3">{r.source}</div>
      <Grounding retrieval={r.retrieval} full />
      <Script script={r.script} by={r.by} />
      {r.run && (
        <div className="text-xs text-gray-400">
          <div>nodes: {r.run.taus.join(", ") || "—"}</div>
          {r.run.spawned.map((s, i) => (
            <div key={i}>spawn {s.program} from {s.target} {s.attached ? "→ chunk attached" : `— ${s.note}`}</div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── dispatch ─────────────────────────────────────────────────────────────

const SECTIONS = { preferences: Preferences, code: Code, peripherals: Peripherals, model: Model, projects: Projects, rag: Rag, plans: Plans, reports: Reports };

export default function SurfaceArtifact({ result, Artifact }) {
  switch (result.kind) {
    case "stack":
      return <div className="space-y-6">{(result.items || []).map((it, i) => <div key={i}><Artifact result={it} /></div>)}</div>;
    case "controls": {
      const Section = SECTIONS[result.section];
      return Section ? <Section {...result} Artifact={Artifact} /> : null;
    }
    case "runtime_map":
      return <RuntimeMap runs={result.runs} graph={result.graph} />;
    case "player_run":
      return <PlayerRun r={result} Artifact={Artifact} />;
    case "report_view":
      return <ReportView report={result.report} />;
    case "mail_accounts": return <MailAccounts {...result} />;
    case "mail_search": return <MailSearch {...result} />;
    case "mail_message": return <MailMessage message={result.message} />;
    case "mail_sync": return <MailSync {...result} />;
    case "planning_board": return <PlanningBoard />;
    case "planning_item": return <PlanningItem id={result.id} created={result.created} />;
    case "planning_find": return <PlanningFind {...result} />;
    case "lattice_home": return <LatticeHome />;
    case "web_search": return <WebSearch {...result} />;
    case "web_page": return <WebPage {...result} />;
    case "web_site": return <WebSite {...result} />;
    case "web_library": return <WebLibrary {...result} />;
    case "spec_model": return <SpecModel {...result} />;
    case "spec_diagram": return <SpecDiagram {...result} />;
    case "spec_compare": return <SpecCompare {...result} />;
    case "lattice_tasks": return <LatticeTasks {...result} />;
    case "lattice_plan": return <LatticePlan {...result} />;
    case "lattice_wrapped": return <LatticeWrapped {...result} />;
    case "lattice_units": return <LatticeUnits {...result} />;
    case "lattice_results": return <LatticeResults {...result} />;
    case "lattice_log": return <LatticeLog {...result} />;
    default:
      return null;
  }
}
