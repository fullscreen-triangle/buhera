/* ============================================================================
 * Spec — a specification understood: what it defines, its diagrams, and
 * what a profile changes about the specification it extends.
 *
 * Every diagram here is drawn from what the specification states (the model,
 * lib/server/spec-model.js) or from your own words, and says which. Changing
 * the view (another class, another depth, the workflow) is a new frame.
 * ========================================================================== */

import { useState } from "react";
import { useStep } from "@/components/surface/actions";
import { Button, Choice } from "@/components/surface/controls";
import { KeepOnPlan } from "@/components/surface/Planning";
import MermaidView from "@/components/surface/MermaidView";

const host = (u) => { try { const x = new URL(u); return x.hostname + x.pathname; } catch { return u; } };

export function SpecModel({ url, title, version, source, classes = [], properties }) {
  const step = useStep();
  const withProps = classes.filter((c) => c.properties > 0);
  const bare = classes.filter((c) => c.properties === 0);
  return (
    <div>
      <div className="text-xs text-gray-500 mb-3">
        <span className="text-gray-200">{title}</span>{version ? ` ${version}` : ""} · {classes.length} classes, {properties} properties · read from its {source === "linkml" ? "LinkML schema" : "property tables"}
      </div>
      <p className="text-xs text-gray-500 mb-4">Mandatory means a minimum cardinality of 1 (in LinkML, <span className="font-mono">required</span>) — the specification&apos;s own word for it, not a reading.</p>
      <table className="text-xs">
        <thead><tr className="text-gray-600 text-left"><th className="font-normal pr-4">class</th><th className="font-normal pr-4">is a</th><th className="font-normal pr-4">properties</th><th className="font-normal">mandatory</th></tr></thead>
        <tbody>
          {withProps.map((c) => (
            <tr key={c.name} className="border-t border-gray-900 align-top">
              <td className="pr-4 py-1">
                <button type="button" className="text-gray-200 hover:text-teal-300" onClick={() => step?.(`diagram ${url} around ${c.name}`, "spec", { kind: "diagram", url, view: "classes", focus: c.name })}>{c.label}</button>
              </td>
              <td className="pr-4 py-1 text-gray-500">{c.isA.join(", ")}</td>
              <td className="pr-4 py-1 text-gray-400">{c.properties}</td>
              <td className="py-1 text-gray-300">{c.mandatory.join(", ") || <span className="text-gray-700">—</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {bare.length > 0 && <p className="text-xs text-gray-600 mt-3">named, with no properties of their own here: {bare.map((c) => c.label).join(", ")}</p>}
    </div>
  );
}

export function SpecDiagram({ url, title, view, focus, depth, attributes, against, highlight = [], classes = [], mermaid, by, provider, model, attempts, error, from }) {
  const step = useStep();
  const [pick, setPick] = useState(focus || "");
  const again = (patch) => step?.(`${patch.view === "flow" ? "workflow" : "diagram"} ${url}${patch.focus ? ` around ${patch.focus}` : ""}`, "spec", { kind: "diagram", url, view, focus, depth, attributes, against, ...patch });
  const origin = by === "model"
    ? `drafted by your model (${provider}${model ? ` ${model}` : ""}${attempts > 1 ? `, ${attempts} attempts` : ""})${from ? ` from the notes of “${from.title}”` : ""} — check it against your sources`
    : by === "you" ? "written by you"
    : view === "flow" ? "the workflow the specification's PROV-O terms state: used → activity → generated, carried out by, at" : "the classes the specification states";

  return (
    <div>
      <div className="text-xs text-gray-500 mb-1"><span className="text-gray-200">{title}</span>{url ? <> · <span className="font-mono">{host(url)}</span></> : null}</div>
      <div className="text-[11px] text-gray-600 mb-3">{origin}{highlight.length ? ` · teal: added by this specification (${highlight.length} classes)` : ""}</div>

      {by === "spec" && step && (
        <div className="flex flex-wrap items-center gap-3 mb-3 text-[11px] text-gray-500">
          <Choice value={view} options={["classes", "flow"]} onChange={(v) => again({ view: v })} />
          {view === "classes" && (
            <>
              <span className="inline-flex items-center gap-1">around
                <select value={pick} onChange={(e) => { setPick(e.target.value); again({ focus: e.target.value || null }); }}
                  className="bg-black border border-gray-800 rounded text-gray-300 px-1 py-0.5">
                  <option value="">every class</option>
                  {classes.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </span>
              <Choice value={String(depth)} options={["1", "2"]} onChange={(v) => again({ depth: Number(v) })} />
              <Choice value={attributes} options={["mandatory", "all", "none"]} onChange={(v) => again({ attributes: v })} />
            </>
          )}
        </div>
      )}
      {error && by !== "spec" && <p className="text-xs text-rose-300/80 mb-2">did not parse after {attempts || 1} attempt(s): {error}</p>}

      <MermaidView text={mermaid} name={`${title}-${view || "diagram"}${focus ? `-${focus}` : ""}`} />

      <div className="mt-3">
        <KeepOnPlan found={{ source: "diagram", cite: url ? `${url} (${view}${focus ? ` around ${focus}` : ""})` : `diagram: ${title}`, title: `${title} — ${view === "flow" ? "workflow" : view === "classes" ? `classes${focus ? ` around ${focus}` : ""}` : "diagram"}`, snippet: origin, mermaid }} />
      </div>
    </div>
  );
}

function Change({ c, aUrl }) {
  const [open, setOpen] = useState(false);
  const link = (anchor) => (anchor && aUrl ? `${aUrl.replace(/#.*$/, "")}#${anchor}` : null);
  return (
    <div className="py-1.5 border-t border-gray-900 text-xs">
      <div className="flex flex-wrap items-baseline gap-x-3">
        <span className="text-gray-200">{c.class}</span>
        {c.added.length > 0 && <button type="button" className="text-gray-500 hover:text-gray-200" onClick={() => setOpen((v) => !v)}>{open ? "▾" : "▸"} +{c.added.length} propert{c.added.length === 1 ? "y" : "ies"}</button>}
      </div>
      {c.stricter.map((p) => <div key={p.name} className="ml-4 text-amber-300/90">{p.name}: {p.from} → {p.to} — now required {link(p.aAnchor) && <a href={link(p.aAnchor)} target="_blank" rel="noreferrer noopener" className="text-gray-600 hover:text-teal-300">(base)</a>}</div>)}
      {c.narrowed.map((p) => <div key={p.name} className="ml-4 text-teal-300/90">{p.name}: range {p.from} → {p.to}{p.subclass ? " (a subclass)" : ""} {link(p.aAnchor) && <a href={link(p.aAnchor)} target="_blank" rel="noreferrer noopener" className="text-gray-600 hover:text-teal-300">(base)</a>}</div>)}
      {c.looser.map((p) => <div key={p.name} className="ml-4 text-gray-400">{p.name}: {p.from} → {p.to} — looser</div>)}
      {open && c.added.map((p) => <div key={p.name} className="ml-4 text-gray-400"><span className="text-gray-300">{p.name}</span> → {p.range} [{p.card}]{p.definition ? <span className="text-gray-600"> — {p.definition}</span> : null}</div>)}
    </div>
  );
}

export function SpecCompare({ aUrl, bUrl, a, b, addedClasses = [], missingClasses = [], changes = [], counts, flow, flowBase }) {
  const step = useStep();
  const key = changes.filter((c) => c.stricter.length || c.narrowed.length || c.looser.length);
  const summary = [
    `${b.title}${b.version ? ` ${b.version}` : ""} against ${a.title}:`,
    `${counts.addedClasses} classes added, ${counts.addedProperties} properties added to classes both have,`,
    `${counts.stricter} made required, ${counts.narrowed} ranges narrowed.`,
    ...key.flatMap((c) => [...c.stricter.map((p) => `${c.class}.${p.name}: ${p.from} → ${p.to}`), ...c.narrowed.map((p) => `${c.class}.${p.name}: ${p.from} → ${p.to}`)]),
  ].join(" ");
  return (
    <div>
      <div className="text-xs text-gray-500 mb-4">
        what <span className="text-gray-200">{b.title}{b.version ? ` ${b.version}` : ""}</span> changes about <span className="text-gray-200">{a.title}</span>
        {" · "}classes matched by name, properties by URI — check a surprising match against both sources
      </div>
      <div className="grid grid-cols-4 md:grid-cols-2 gap-3 mb-6">
        {[["classes added", counts.addedClasses], ["properties added", counts.addedProperties], ["made required", counts.stricter], ["ranges narrowed", counts.narrowed]].map(([k, v]) => (
          <div key={k} className="border border-gray-900 rounded p-3"><div className="text-2xl text-gray-100">{v}</div><div className="text-[11px] text-gray-500">{k}</div></div>
        ))}
      </div>

      {key.length > 0 && (
        <section className="mb-6">
          <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-1">what it changes in properties the base already had</div>
          {key.map((c) => <Change key={c.class} c={{ ...c, added: [] }} aUrl={aUrl} />)}
        </section>
      )}

      <section className="mb-6">
        <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-2">the workflow each states</div>
        <div className="text-[11px] text-gray-500 mb-1">{a.title}</div>
        <MermaidView text={flowBase} name={`${a.title}-workflow`} compact />
        <div className="text-[11px] text-gray-500 mt-4 mb-1">{b.title} — teal: classes it adds</div>
        <MermaidView text={flow} name={`${b.title}-workflow`} />
      </section>

      <section className="mb-6">
        <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-1">classes it adds</div>
        {addedClasses.map((c) => (
          <div key={c.name} className="py-1 border-t border-gray-900 text-xs flex flex-wrap gap-x-3">
            <button type="button" className="text-gray-200 hover:text-teal-300" onClick={() => step?.(`diagram ${bUrl} around ${c.name}`, "spec", { kind: "diagram", url: bUrl, view: "classes", focus: c.name, against: aUrl })}>{c.name}</button>
            {c.isA.length > 0 && <span className="text-gray-600">is a {c.isA.join(", ")}</span>}
            <span className="text-gray-500">{c.description}</span>
          </div>
        ))}
      </section>

      <section className="mb-6">
        <div className="text-[11px] uppercase tracking-wider text-gray-600 mb-1">properties it adds, by class</div>
        {changes.filter((c) => c.added.length).map((c) => <Change key={c.class} c={{ ...c, stricter: [], narrowed: [], looser: [] }} aUrl={aUrl} />)}
      </section>

      {missingClasses.length > 0 && <p className="text-xs text-gray-600 mb-4">in {a.title} but not named in {b.title}: {missingClasses.map((c) => c.label || c.name).join(", ")}</p>}

      <KeepOnPlan found={{ source: "comparison", cite: `${aUrl} → ${bUrl}`, title: `What ${b.title} changes about ${a.title}`, snippet: summary, mermaid: flow }} />
    </div>
  );
}
