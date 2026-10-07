/* ============================================================================
 * Planning — the board, one plan item, what `find` found, and the control
 * that keeps a found thing on a plan.
 *
 * The board and an item are live views of the planning store
 * (lib/surface/planning.js): a frame that shows an item keeps showing the
 * item as it is now, the way a settings page does. `find` results are a
 * snapshot, like every other frame; keeping one of them on a plan writes to
 * the store and leaves the frame as it was.
 * ========================================================================== */

import { useMemo, useState } from "react";
import { useSettings } from "@/lib/surface/settings";
import {
  STATUSES, addItem, addJob, addRef, addStep, itemsInProject, removeItem, removeRef, removeStep,
  toMarkdown, toggleStep, updateItem, usePlanning,
} from "@/lib/surface/planning";
import { useStep } from "@/components/surface/actions";
import { Button, Choice, when } from "@/components/surface/controls";
import { SpraypaintResult } from "@/components/sandboxes/spraypaint/SpraypaintResult";

const VERDICT_INK = { covered: "text-teal-300", partial: "text-amber-300", declined: "text-rose-300" };

function Field({ value, onSave, placeholder, className = "", multiline = false }) {
  const props = {
    defaultValue: value,
    placeholder,
    spellCheck: false,
    onBlur: (e) => { if (e.target.value !== value) onSave(e.target.value); },
    className: `bg-transparent border-b border-gray-900 focus:border-gray-600 outline-none text-gray-200 ${className}`,
  };
  return multiline ? <textarea rows={3} {...props} /> : <input {...props} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }} />;
}

// ── keeping a found thing ────────────────────────────────────────────────

/**
 * "keep on a plan": pick an open item of the active project, or start a new
 * one from this. `commit`, if given, runs first — a search whose answer is
 * kept is a search relied on, so it is committed (spraypaint's count) then.
 */
export function KeepOnPlan({ found, commit }) {
  const items = usePlanning();
  const { project } = useSettings();
  const [open, setOpen] = useState(false);
  const [title, setTitle] = useState("");
  const [kept, setKept] = useState(null);
  const openItems = itemsInProject(items, project).filter((i) => i.status !== "done" && i.status !== "dropped");

  async function keep(itemId) {
    let extra = {};
    if (commit) {
      try { extra = (await commit()) || {}; } catch { /* the ref is kept either way */ }
    }
    const item = itemId ? items.find((i) => i.id === itemId) : addItem({ title: title || found.title, kind: "task" });
    addRef(item.id, { ...found, ...extra });
    setKept(item.title);
    setOpen(false);
  }

  if (kept) return <span className="text-[11px] text-teal-400/80">kept on “{kept}”</span>;
  return (
    <span className="relative inline-block">
      <button type="button" onClick={() => setOpen((o) => !o)} className="text-[11px] text-gray-500 hover:text-teal-300">
        + keep on a plan
      </button>
      {open && (
        <div className="absolute z-20 mt-1 left-0 w-72 border border-gray-800 bg-black rounded p-2 shadow-lg">
          {openItems.slice(0, 8).map((i) => (
            <button key={i.id} type="button" onClick={() => keep(i.id)}
              className="block w-full text-left text-xs text-gray-300 hover:text-white py-0.5 truncate">
              {i.title} <span className="text-gray-600">· {i.kind}</span>
            </button>
          ))}
          <div className="flex gap-2 mt-1 pt-1 border-t border-gray-900">
            <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="a new plan…" spellCheck={false}
              onKeyDown={(e) => { if (e.key === "Enter") keep(null); }}
              className="flex-1 bg-transparent outline-none text-xs text-gray-200" />
            <button type="button" onClick={() => keep(null)} className="text-[11px] text-gray-500 hover:text-teal-300">new</button>
          </div>
        </div>
      )}
    </span>
  );
}

// ── the board ────────────────────────────────────────────────────────────

export function PlanningBoard() {
  const items = usePlanning();
  const { project } = useSettings();
  const step = useStep();
  const [title, setTitle] = useState("");
  const [kind, setKind] = useState("experiment");
  const mine = itemsInProject(items, project);
  const groups = STATUSES.map((s) => [s, mine.filter((i) => i.status === s)]).filter(([, l]) => l.length);

  function create() {
    if (!title.trim()) return;
    const item = addItem({ title, kind });
    setTitle("");
    step?.(`plan ${kind} ${item.title}`, "planning", { kind: "item", id: item.id });
  }

  return (
    <div>
      <div className="text-xs text-gray-600 mb-3">project {project} · {mine.length} item{mine.length === 1 ? "" : "s"}</div>
      <div className="flex flex-wrap items-center gap-3 mb-6">
        <Choice value={kind} options={["experiment", "task"]} onChange={setKind} />
        <input value={title} onChange={(e) => setTitle(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") create(); }}
          placeholder={kind === "experiment" ? "what is the experiment?" : "what needs doing?"} spellCheck={false}
          className="flex-1 min-w-[16rem] bg-transparent border-b border-gray-800 focus:border-gray-600 outline-none text-sm text-gray-200 px-1" />
        <Button disabled={!title.trim()} onClick={create}>plan it</Button>
      </div>

      {!mine.length && (
        <p className="text-gray-500 italic">
          nothing planned in “{project}” yet. Write <span className="not-italic text-gray-400">plan experiment …</span> or{" "}
          <span className="not-italic text-gray-400">find …</span> on the blank screen, or start one above.
        </p>
      )}

      {groups.map(([status, list]) => (
        <div key={status} className="mb-5">
          <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-1">{status}</div>
          {list.map((i) => {
            const doneSteps = i.steps.filter((s) => s.done).length;
            return (
              <button key={i.id} type="button" onClick={() => step?.(`plan: ${i.title}`, "planning", { kind: "item", id: i.id })}
                className="w-full text-left flex flex-wrap items-baseline gap-x-3 py-1.5 border-t border-gray-900 hover:bg-white/[0.02]">
                <span className="text-gray-200">{i.title}</span>
                <span className="text-[11px] text-gray-600">{i.kind}</span>
                {i.steps.length > 0 && <span className="text-[11px] text-gray-500">{doneSteps}/{i.steps.length} steps</span>}
                {i.refs.length > 0 && <span className="text-[11px] text-gray-500">{i.refs.length} found</span>}
                {i.jobs.length > 0 && <span className="text-[11px] text-gray-500">{i.jobs.length} job{i.jobs.length === 1 ? "" : "s"}</span>}
                {i.due && <span className="text-[11px] text-amber-300/70">due {i.due}</span>}
              </button>
            );
          })}
        </div>
      ))}

      {step && (
        <div className="mt-6 text-[11px] text-gray-600">
          <button type="button" className="hover:text-gray-300" onClick={() => step("plans", "plans", "show")}>the scripts your steps ran →</button>
        </div>
      )}
    </div>
  );
}

// ── one item ─────────────────────────────────────────────────────────────

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function RefRow({ itemId, r }) {
  const step = useStep();
  const openable = r.mail || r.corpus === "mail";
  return (
    <div className="py-2 border-t border-gray-900">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <span className="text-[11px] text-gray-600 w-12">{r.source}</span>
        <span className="text-sm text-gray-200">{r.title || r.cite}</span>
        {r.verdict && <span className={`text-[11px] ${VERDICT_INK[r.verdict] || "text-gray-400"}`}>{r.verdict}</span>}
        {r.committed_count != null && <span className="text-[11px] text-gray-600">committed act #{r.committed_count}</span>}
        {openable && step && (
          <button type="button" className="text-[11px] text-gray-500 hover:text-teal-300"
            onClick={() => step(`mail: ${r.title || r.cite}`, "mail", r.mail ? { kind: "read", ...r.mail } : { kind: "open", path: r.path })}>
            open
          </button>
        )}
        <button type="button" className="text-[11px] text-gray-700 hover:text-rose-300" onClick={() => removeRef(itemId, r.id)}>remove</button>
      </div>
      <div className="ml-12 text-[11px] font-mono text-teal-300/70 break-all">{r.cite}</div>
      {r.snippet && <div className="ml-12 text-xs text-gray-500 whitespace-pre-wrap line-clamp-3">{r.snippet}</div>}
      {r.query && <div className="ml-12 text-[11px] text-gray-700">found by “{r.query}”</div>}
    </div>
  );
}

export function PlanningItem({ id, created }) {
  const items = usePlanning();
  const settings = useSettings();
  const step = useStep();
  const item = items.find((i) => i.id === id);
  const [stepText, setStepText] = useState("");
  const [job, setJob] = useState({ repo: settings.lattice.repos[0] || "", unit: "" });
  const [sure, setSure] = useState(false);

  if (!item) return <p className="text-gray-500 italic">this plan item was deleted.</p>;
  const set = (patch) => updateItem(item.id, patch);

  return (
    <div>
      {created && <div className="text-[11px] text-teal-400/70 mb-2">new {item.kind}, in project {item.project}</div>}
      <Field value={item.title} onSave={(v) => set({ title: v.trim() || item.title })} className="w-full text-lg" />
      <div className="flex flex-wrap items-center gap-4 mt-3 text-xs text-gray-500">
        <span>{item.kind}</span>
        <Choice value={item.status} options={STATUSES} onChange={(v) => set({ status: v })} />
        <label className="inline-flex items-center gap-2">
          due
          <input type="date" value={item.due || ""} onChange={(e) => set({ due: e.target.value || null })}
            className="bg-transparent border-b border-gray-900 text-gray-300 outline-none [color-scheme:dark]" />
        </label>
        <span className="text-gray-700">started {when(item.created)}</span>
      </div>

      <div className="mt-5">
        <Field value={item.notes} onSave={(v) => set({ notes: v })} multiline placeholder="notes: the question, the hypothesis, what matters" className="w-full text-sm border rounded p-2 border-gray-900" />
      </div>

      <section className="mt-6">
        <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-1">steps</div>
        {item.steps.map((s) => (
          <div key={s.id} className="group flex items-baseline gap-3 py-0.5">
            <input type="checkbox" checked={s.done} onChange={() => toggleStep(item.id, s.id)} className="accent-teal-500" />
            <span className={s.done ? "text-gray-600 line-through" : "text-gray-300"}>{s.text}</span>
            <button type="button" onClick={() => removeStep(item.id, s.id)} className="opacity-0 group-hover:opacity-100 text-[11px] text-gray-700 hover:text-rose-300">×</button>
          </div>
        ))}
        <input value={stepText} onChange={(e) => setStepText(e.target.value)} placeholder="+ a step" spellCheck={false}
          onKeyDown={(e) => { if (e.key === "Enter" && stepText.trim()) { addStep(item.id, stepText); setStepText(""); } }}
          className="mt-1 bg-transparent outline-none text-sm text-gray-300 placeholder:text-gray-700" />
      </section>

      <section className="mt-6">
        <div className="flex items-baseline gap-3 mb-1">
          <span className="text-[11px] uppercase tracking-wider text-gray-600">what we found</span>
          {step && <button type="button" className="text-[11px] text-gray-500 hover:text-teal-300" onClick={() => step(`find ${item.title}`, "planning", { kind: "find", query: item.title })}>find for this →</button>}
        </div>
        {item.refs.length === 0 && <p className="text-xs text-gray-600">nothing kept yet — write <span className="text-gray-400">find …</span> and keep what bears on this.</p>}
        {item.refs.map((r) => <RefRow key={r.id} itemId={item.id} r={r} />)}
      </section>

      <section className="mt-6">
        <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-1">jobs on AppHub</div>
        {item.jobs.map((j) => (
          <div key={`${j.repo}:${j.unit}`} className="flex flex-wrap items-baseline gap-3 py-0.5 text-sm">
            <span className="text-gray-200 font-mono">{j.unit}</span>
            <span className="text-[11px] text-gray-600 font-mono break-all">{j.repo}</span>
            {step && <button type="button" className="text-[11px] text-gray-500 hover:text-teal-300" onClick={() => step(`apphub results ${j.unit}`, "lattice", { kind: "results", repo: j.repo, unit: j.unit, remote: j.remote })}>results</button>}
          </div>
        ))}
        <div className="flex flex-wrap items-center gap-2 mt-1">
          <input value={job.repo} onChange={(e) => setJob({ ...job, repo: e.target.value })} placeholder="repository path" spellCheck={false}
            list="lattice-repos" className="w-72 bg-transparent border-b border-gray-900 focus:border-gray-600 outline-none text-xs font-mono text-gray-300" />
          <datalist id="lattice-repos">{settings.lattice.repos.map((r) => <option key={r} value={r} />)}</datalist>
          <input value={job.unit} onChange={(e) => setJob({ ...job, unit: e.target.value })} placeholder="unit" spellCheck={false}
            className="w-32 bg-transparent border-b border-gray-900 focus:border-gray-600 outline-none text-xs font-mono text-gray-300" />
          <Button disabled={!job.repo.trim() || !job.unit.trim()} onClick={() => { addJob(item.id, { repo: job.repo.trim(), unit: job.unit.trim() }); setJob({ ...job, unit: "" }); }}>add a job</Button>
        </div>
      </section>

      <div className="mt-8 flex flex-wrap gap-3">
        <Button onClick={() => download(`${item.title.replace(/[^\w.-]+/g, "-").slice(0, 60)}.md`, toMarkdown(item))}>save as markdown</Button>
        {sure
          ? <Button onClick={() => removeItem(item.id)}>yes, delete “{item.title}”</Button>
          : <Button onClick={() => setSure(true)}>delete</Button>}
      </div>
    </div>
  );
}

// ── what find found ──────────────────────────────────────────────────────

const SOURCE_TITLE = { mail: "your mail", files: "your files", web: "the web", plans: "your plans" };

function MailHits({ live, query }) {
  const step = useStep();
  if (!live) return null;
  return (
    <div>
      {live.accounts?.filter((a) => a.error).map((a) => (
        <div key={a.account} className="text-xs text-rose-300/80">{a.account}: {a.error}</div>
      ))}
      {live.messages?.length === 0 && <div className="text-xs text-gray-500">no message matches, in any account.</div>}
      {live.messages?.map((m) => (
        <div key={`${m.account}/${m.mailbox}/${m.uid}`} className="flex flex-wrap items-baseline gap-x-3 py-1 border-t border-gray-900">
          <span className="text-[11px] text-gray-600 w-20 shrink-0">{m.date ? m.date.slice(0, 10) : ""}</span>
          <button type="button" className="text-sm text-gray-200 hover:text-white text-left"
            onClick={() => step?.(`mail: ${m.subject}`, "mail", { kind: "read", account: m.account, mailbox: m.mailbox, uid: m.uid })}>
            {m.subject || "(no subject)"}
          </button>
          <span className="text-[11px] text-gray-500">{m.from?.[0]?.name || m.from?.[0]?.address}</span>
          <span className="text-[11px] text-gray-700">{m.account}</span>
          <KeepOnPlan found={{ source: "mail", cite: `mail:${m.account}/${m.mailbox}/${m.uid}`, title: m.subject, snippet: `from ${m.from?.[0]?.name || m.from?.[0]?.address || "?"}, ${m.date?.slice(0, 10) || ""}`, query, mail: { account: m.account, mailbox: m.mailbox, uid: m.uid } }} />
        </div>
      ))}
    </div>
  );
}

const firstLine = (s) => {
  const l = String(s || "").split("\n").find((x) => x.trim())?.replace(/^#+\s*/, "").trim() || "";
  return l.length > 80 ? `${l.slice(0, 79)}…` : l;
};

// A passage kept from a spraypaint search: committed when kept (the ask relied on).
function passageFound(r, result, source, corpus) {
  const from = r.evidence_start_line ?? r.start_line;
  const to = r.evidence_end_line ?? r.end_line;
  return {
    source,
    cite: `${corpus === "mail" ? "kept mail: " : ""}${r.path}:${from}-${to}`,
    title: firstLine(r.snippet) || r.path,
    snippet: r.snippet,
    verdict: result.coverage?.verdict || null,
    query: result.query,
    path: r.path,
    corpus,
  };
}

async function commitAsk(corpus, query) {
  const res = await fetch(corpus === "mail" ? "/api/mail" : "/api/spraypaint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action: "ask", query, dry_run: false, budget: 1 }),
  });
  const j = await res.json().catch(() => null);
  const n = j?.output_delta?.committed_count;
  return typeof n === "number" ? { committed_count: n } : {};
}

function Passages({ result, source, corpus }) {
  return (
    <SpraypaintResult {...result} compact
      rowAction={(r) => <KeepOnPlan found={passageFound(r, result, source, corpus)} commit={() => commitAsk(corpus, result.query)} />} />
  );
}

function Section({ s, query }) {
  const step = useStep();
  let body;
  if (!s.ok && !s.live && !s.kept) body = <p className="text-xs text-rose-300/80">{s.error}</p>;
  else if (s.source === "mail") {
    body = (
      <div className="space-y-3">
        <MailHits live={s.live} query={query} />
        {s.kept && (
          <div className="pt-2">
            <div className="text-[11px] text-gray-600 mb-1">in the kept mail, with a verdict</div>
            <Passages result={s.kept} source="mail" corpus="mail" />
          </div>
        )}
        {s.keptNote && <p className="text-[11px] text-gray-600">{s.keptNote}</p>}
      </div>
    );
  } else if (s.source === "files") body = <Passages result={s.result} source="files" corpus="files" />;
  else if (s.source === "web") {
    const w = s.result || {};
    body = (
      <div>
        <p className="text-sm text-gray-300 whitespace-pre-wrap leading-relaxed">{w.content}</p>
        {!w.grounded && <p className="text-[11px] text-amber-300/70 mt-1">no citations came back — treat this as the model&apos;s own words.</p>}
        {w.sources?.slice(0, 5).map((src) => (
          <div key={src.index} className="text-[11px] flex gap-3">
            <a href={src.uri} target="_blank" rel="noreferrer" className="text-gray-400 hover:text-teal-300 truncate">{src.title || src.uri}</a>
            <KeepOnPlan found={{ source: "web", cite: src.uri || src.title, title: src.title || src.uri, snippet: "", query }} />
          </div>
        ))}
      </div>
    );
  } else if (s.source === "plans") {
    body = s.items?.length
      ? s.items.map((i) => (
          <button key={i.id} type="button" onClick={() => step?.(`plan: ${i.title}`, "planning", { kind: "item", id: i.id })}
            className="block text-sm text-gray-300 hover:text-white py-0.5">
            {i.title} <span className="text-[11px] text-gray-600">· {i.kind} · {i.status}</span>
          </button>
        ))
      : <p className="text-xs text-gray-600">no plan of yours mentions it.</p>;
  }
  return (
    <section className="mb-8">
      <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-2">{SOURCE_TITLE[s.source] || s.source}</div>
      {body}
    </section>
  );
}

export function PlanningFind({ query, sections }) {
  const step = useStep();
  const order = useMemo(() => ["mail", "files", "plans", "web"], []);
  const sorted = [...(sections || [])].sort((a, b) => order.indexOf(a.source) - order.indexOf(b.source));
  return (
    <div>
      <div className="text-xs text-gray-500 mb-5">
        found for <span className="text-white">“{query}”</span> — previews: nothing is committed until you keep it.
        {step && <button type="button" className="ml-3 hover:text-teal-300" onClick={() => step(`plan task ${query}`, "planning", { kind: "new", type: "task", title: query })}>plan this →</button>}
      </div>
      {sorted.map((s) => <Section key={s.source} s={s} query={query} />)}
    </div>
  );
}
