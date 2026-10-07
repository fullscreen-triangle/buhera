/* ============================================================================
 * Jobs — AppHub through lattice: your repositories, a repository's tasks,
 * a plan (what a task needs there), a wrapped unit and its results.
 *
 * The path of a job, and where each part happens:
 *   here     tasks → plan (writes nothing) → wrap (commits .lattice/<unit>/
 *            on your current branch and pushes it to the university's Gitea)
 *   AppHub   you start a session; in its terminal: git pull, then
 *            bash .lattice/<unit>/run.sh --detach
 *   here     results (what the session pushed back) → get (copy the outputs)
 * Wrapping and getting change a repository, so each asks once more first.
 * ========================================================================== */

import { useState } from "react";
import { useSettings, update } from "@/lib/surface/settings";
import { addJob, itemsInProject, usePlanning } from "@/lib/surface/planning";
import { useStep } from "@/components/surface/actions";
import { Button, Row } from "@/components/surface/controls";

const STATE_INK = { done: "text-teal-300", running: "text-amber-300", pending: "text-gray-500", failed: "text-rose-300" };

function Notes({ notes = [], error }) {
  return (
    <>
      {error && <p className="text-sm text-rose-300/90 mb-2 whitespace-pre-wrap">{error}</p>}
      {notes.length > 0 && (
        <div className="text-[11px] text-gray-500 mb-3 space-y-0.5">
          {notes.map((n, i) => <div key={i}>lattice: {n}</div>)}
        </div>
      )}
    </>
  );
}

function RepoName({ repo }) {
  const name = String(repo || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop();
  return <span><span className="text-gray-200">{name}</span> <span className="text-[11px] text-gray-600 font-mono break-all">{repo}</span></span>;
}

export function LatticeHome() {
  const { lattice } = useSettings();
  const step = useStep();
  const [path, setPath] = useState("");
  const add = () => {
    const p = path.trim().replace(/\\/g, "/");
    if (!p) return;
    update("lattice", { repos: [...new Set([...lattice.repos, p])] });
    setPath("");
  };
  return (
    <div>
      <p className="text-sm text-gray-400 mb-4 leading-relaxed">
        AppHub takes no connection from outside, so a job travels by git: lattice wraps a task into a unit and pushes it
        to the university&apos;s Gitea; you start an AppHub session and run it there; it pushes its results back, and they
        show up here.
      </p>
      {lattice.repos.length === 0 && <p className="text-xs text-gray-500 mb-2">no repositories yet — add one that has a remote on git.uni-greifswald.de.</p>}
      {lattice.repos.map((r) => (
        <Row key={r} label={<RepoName repo={r} />}>
          <span className="inline-flex flex-wrap gap-2">
            {step && <Button onClick={() => step(`apphub tasks`, "lattice", { kind: "tasks", repo: r })}>tasks</Button>}
            {step && <Button onClick={() => step(`apphub units`, "lattice", { kind: "units", repo: r })}>units &amp; results</Button>}
            <button type="button" className="text-[11px] text-gray-700 hover:text-rose-300"
              onClick={() => update("lattice", { repos: lattice.repos.filter((x) => x !== r) })}>forget</button>
          </span>
        </Row>
      ))}
      <div className="flex gap-2 mt-4">
        <input value={path} onChange={(e) => setPath(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") add(); }}
          placeholder="C:/Users/you/Documents/project" spellCheck={false}
          className="flex-1 bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-xs font-mono text-gray-200 px-1" />
        <Button disabled={!path.trim()} onClick={add}>add a repository</Button>
      </div>
    </div>
  );
}

// The shape of a run: what `plan` and `wrap` take.
function SpecForm({ repo, task, onPlan }) {
  const [matrix, setMatrix] = useState("");
  const [each, setEach] = useState("");
  const [outputs, setOutputs] = useState("");
  const list = (s) => s.split(/\s+/).map((x) => x.trim()).filter(Boolean);
  const input = "bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-xs font-mono text-gray-200 px-1";
  return (
    <div className="ml-4 mt-1 mb-3 grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1 text-[11px] text-gray-500 items-center">
      <span>matrix</span><input className={input} value={matrix} onChange={(e) => setMatrix(e.target.value)} placeholder="seed=1..5 model=a,b" />
      <span>once per file</span><input className={input} value={each} onChange={(e) => setEach(e.target.value)} placeholder="data/*.csv" />
      <span>send back</span><input className={input} value={outputs} onChange={(e) => setOutputs(e.target.value)} placeholder="results/**" />
      <span />
      <span><Button onClick={() => onPlan({ repo, task, matrix: list(matrix), each: each.trim() || undefined, outputs: list(outputs) })}>plan it — writes nothing</Button></span>
    </div>
  );
}

export function LatticeTasks({ repo, tasks = [], notes, error }) {
  const step = useStep();
  const [open, setOpen] = useState(null);
  return (
    <div>
      <div className="mb-3"><RepoName repo={repo} /></div>
      <Notes notes={notes} error={error} />
      {tasks.length === 0 && !error && <p className="text-xs text-gray-500">no tasks in .vscode/tasks.json or pixi.</p>}
      {tasks.map((t) => (
        <div key={t.name} className="border-t border-gray-900 py-1">
          <button type="button" className="text-sm text-gray-200 hover:text-white" onClick={() => setOpen(open === t.name ? null : t.name)}>
            {t.name} <span className="text-[11px] text-gray-600">· {t.source}</span>
          </button>
          {open === t.name && step && <SpecForm repo={repo} task={t.name} onPlan={(spec) => step(`apphub plan ${t.name}`, "lattice", { kind: "plan", ...spec })} />}
        </div>
      ))}
    </div>
  );
}

function Summary({ s }) {
  if (!s) return null;
  const rows = [
    ["unit", s.unit], ["from", s.from], ["runs", s.runs], ["environment", s.environment], ["compute", s.compute],
    ["profile", s.profile], ["parallel", s.parallel], ["models", s.models], ["secrets", s.secrets], ["results", s.results],
  ].filter(([, v]) => v);
  return (
    <div>
      <div className="grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <span className="text-gray-600 text-xs">{k}</span>
            <span className={k === "runs" ? "font-mono text-xs text-teal-100/80 break-all" : "text-gray-300"}>{v}</span>
          </div>
        ))}
      </div>
      {s.notes.length > 0 && <div className="mt-2 text-xs text-gray-400">{s.notes.map((n, i) => <div key={i}>– {n}</div>)}</div>}
      {s.warnings.length > 0 && <div className="mt-2 text-xs text-amber-300/80">{s.warnings.map((n, i) => <div key={i}>! {n}</div>)}</div>}
    </div>
  );
}

function OnAppHub({ s }) {
  if (!s?.steps?.length) return null;
  return (
    <div className="mt-5">
      <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-1">on AppHub (Code-Server → Terminal)</div>
      {s.steps.map((st) => (
        <div key={st.n} className="text-xs mb-1">
          <span className="text-gray-500">{st.n}. {st.what}</span>
          <pre className="ml-4 font-mono text-teal-100/90 whitespace-pre-wrap">{st.command}</pre>
          {st.more.map((m, i) => <div key={i} className="ml-4 text-gray-600">{m}</div>)}
        </div>
      ))}
    </div>
  );
}

function KeepJob({ repo, unit }) {
  const items = usePlanning();
  const { project } = useSettings();
  const [kept, setKept] = useState(null);
  const open = itemsInProject(items, project).filter((i) => i.status !== "done" && i.status !== "dropped");
  if (kept) return <span className="text-[11px] text-teal-400/80">on “{kept}”</span>;
  if (!open.length) return null;
  return (
    <select defaultValue="" onChange={(e) => { const i = open.find((x) => x.id === e.target.value); if (i) { addJob(i.id, { repo, unit }); setKept(i.title); } }}
      className="bg-black border border-gray-800 rounded text-[11px] text-gray-400 px-1 py-0.5">
      <option value="" disabled>+ this job belongs to a plan…</option>
      {open.map((i) => <option key={i.id} value={i.id}>{i.title}</option>)}
    </select>
  );
}

export function LatticePlan({ repo, spec, summary, notes, error, ok }) {
  const step = useStep();
  const [sure, setSure] = useState(false);
  return (
    <div>
      <div className="mb-3"><RepoName repo={repo} /></div>
      <Notes notes={notes} error={error} />
      <Summary s={summary} />
      <OnAppHub s={summary} />
      {ok && step && (
        <div className="mt-6">
          {sure ? (
            <span className="inline-flex flex-wrap items-center gap-3 text-xs text-gray-400">
              this commits <span className="font-mono">.lattice/{summary?.name}/</span> on your current branch — and nothing else — and pushes it to Gitea.
              <Button onClick={() => step(`apphub wrap ${summary?.name}`, "lattice", { kind: "wrap", ...spec, repo, confirm: true })}>wrap and push</Button>
              <button type="button" className="text-gray-600 hover:text-gray-300" onClick={() => setSure(false)}>not yet</button>
            </span>
          ) : (
            <Button onClick={() => setSure(true)}>wrap it…</Button>
          )}
        </div>
      )}
    </div>
  );
}

export function LatticeWrapped({ repo, summary, notes, error, ok, pushed }) {
  const step = useStep();
  return (
    <div>
      <div className="mb-3"><RepoName repo={repo} /></div>
      <Notes notes={notes} error={error} />
      {ok && <p className="text-sm text-teal-300/90 mb-4">{pushed ? "wrapped and pushed — AppHub can pull it now." : "wrapped and committed, not pushed."}</p>}
      <OnAppHub s={summary} />
      <div className="mt-5"><Summary s={summary} /></div>
      {ok && (
        <div className="mt-6 flex flex-wrap items-center gap-3">
          {step && <Button onClick={() => step(`apphub results ${summary?.name}`, "lattice", { kind: "results", repo, unit: summary?.name })}>results</Button>}
          <KeepJob repo={repo} unit={summary?.name} />
        </div>
      )}
    </div>
  );
}

export function LatticeUnits({ repo, units = [], notes, error }) {
  const step = useStep();
  return (
    <div>
      <div className="mb-3"><RepoName repo={repo} /></div>
      <Notes notes={notes} error={error} />
      {units.length === 0 && !error && <p className="text-xs text-gray-500">no units yet — pick a task and wrap it.</p>}
      {units.map((u) => (
        <div key={u.name} className="flex flex-wrap items-baseline gap-3 py-1 border-t border-gray-900">
          <span className="text-sm text-gray-200 font-mono">{u.name}</span>
          {u.error ? <span className="text-xs text-rose-300/80">{u.error}</span>
            : <span className="text-[11px] text-gray-500">{u.shards} shard{u.shards === 1 ? "" : "s"} · GPU {u.gpu} · {u.from}</span>}
          {step && <Button onClick={() => step(`apphub results ${u.name}`, "lattice", { kind: "results", repo, unit: u.name })}>results</Button>}
          <KeepJob repo={repo} unit={u.name} />
        </div>
      ))}
    </div>
  );
}

const STATE_WORD = {
  complete: "every shard is done.",
  incomplete: "not finished — some shards are still running, failed, or not reported yet.",
  "nothing-pushed": "nothing pushed back yet. Is the unit running on AppHub?",
};

export function LatticeResults({ repo, unit, state, results, notes, error, ok }) {
  const step = useStep();
  const [sure, setSure] = useState(false);
  const r = results || {};
  return (
    <div>
      <div className="mb-3"><RepoName repo={repo} /> <span className="text-gray-600">· unit</span> <span className="font-mono text-gray-200">{unit || r.unit}</span></div>
      <Notes notes={state === "nothing-pushed" ? [] : notes} error={ok ? null : error} />
      <p className={`text-sm mb-3 ${state === "complete" ? "text-teal-300/90" : "text-gray-400"}`}>{STATE_WORD[state] || state}</p>
      {r.commit && <div className="text-[11px] text-gray-600 mb-2">results at {r.commit} ({r.age} ago)</div>}
      {r.runs?.map((x) => (
        <div key={x.part} className="text-xs text-gray-400">
          part {x.part} · {x.state} on {x.host} · {x.shards} shard(s), {x.parallel} at a time, {x.cpus} CPU, {x.gpus} GPU · updated {x.updated} ago
          {x.stale && <span className="text-amber-300/70"> · an earlier version of the unit</span>}
        </div>
      ))}
      {r.shards?.length > 0 && (
        <table className="mt-3 text-xs">
          <thead><tr className="text-gray-600 text-left"><th className="pr-6 font-normal">shard</th><th className="pr-4 font-normal">state</th><th className="pr-4 font-normal">exit</th><th className="pr-4 font-normal">took</th><th className="font-normal">host</th><th /></tr></thead>
          <tbody>
            {r.shards.map((s) => (
              <tr key={s.id} className="border-t border-gray-900">
                <td className="pr-6 font-mono text-gray-300">{s.id}</td>
                <td className={`pr-4 ${STATE_INK[s.state] || "text-rose-300"}`}>{s.state}</td>
                <td className="pr-4 text-gray-400">{s.exit ?? ""}</td>
                <td className="pr-4 text-gray-400">{s.took ?? ""}</td>
                <td className="pr-4 text-gray-600">{s.host ?? ""}</td>
                <td>{step && s.state !== "pending" && <button type="button" className="text-gray-500 hover:text-teal-300" onClick={() => step(`apphub log ${s.id}`, "lattice", { kind: "log", repo, unit: unit || r.unit, shard: s.id })}>log</button>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {r.totals && <div className="text-xs text-gray-500 mt-2">{r.totals.done} done, {r.totals.failed} failed, {r.totals.running} running, {r.totals.pending} not reported, of {r.totals.of}</div>}
      {step && (
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <Button onClick={() => step(`apphub results ${unit}`, "lattice", { kind: "results", repo, unit })}>look again</Button>
          {r.shards?.some((s) => s.state === "done") && (sure
            ? <Button onClick={() => step(`apphub get ${unit}`, "lattice", { kind: "get", repo, unit, confirm: true })}>copy the outputs into the working tree</Button>
            : <Button onClick={() => setSure(true)}>get the outputs…</Button>)}
        </div>
      )}
    </div>
  );
}

export function LatticeLog({ shard, log, notes, error }) {
  return (
    <div>
      <div className="text-xs text-gray-500 mb-2">shard <span className="font-mono text-gray-300">{shard}</span></div>
      <Notes notes={notes} error={error} />
      <pre className="p-3 rounded border border-gray-900 bg-white/[0.02] text-xs font-mono text-gray-300 whitespace-pre-wrap break-all">{log || "(empty)"}</pre>
    </div>
  );
}
