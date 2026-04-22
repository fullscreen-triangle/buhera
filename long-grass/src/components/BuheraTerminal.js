import { useEffect, useRef, useState } from "react";
import { Kernel } from "@/lib/kernel";
import { embedProtein, embedText } from "@/lib/substrate";
import { translate } from "@/lib/translator";
import { executeVahera } from "@/lib/vahera";
import { PROTEINS } from "@/lib/proteins";

// ────────────────────────────────────────────────────────────
//  Boot the kernel with proteins.
// ────────────────────────────────────────────────────────────

function bootKernel() {
  const k = new Kernel(12);
  for (const name of Object.keys(PROTEINS)) {
    const coord = embedProtein(name, PROTEINS[name]);
    k.allocate(coord, PROTEINS[name], { name, gene: PROTEINS[name].gene });
  }
  return k;
}

// ────────────────────────────────────────────────────────────
//  Artifact renderers.
// ────────────────────────────────────────────────────────────

function ProteinHeader({ name, p }) {
  return (
    <div className="flex items-baseline gap-6 mb-3">
      <span className="text-white text-base">{name}</span>
      <span className="text-gray-500 text-xs">{p.gene}</span>
      <span className="text-gray-600 text-xs">{p.uniprot}</span>
      <span className="text-gray-600 text-xs">{p.length} aa</span>
      <span className="text-gray-500 text-xs italic">{p.role}</span>
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
    return (
      <div>
        <ProteinHeader name={name} p={p} />
        <Field label="function" value={p.function} />
      </div>
    );
  }
  if (aspect === "diseases") {
    return (
      <div>
        <ProteinHeader name={name} p={p} />
        <Field label="diseases" value={p.diseases} />
      </div>
    );
  }
  if (aspect === "interacts") {
    return (
      <div>
        <ProteinHeader name={name} p={p} />
        <Field label="interacts" value={p.interacts} />
      </div>
    );
  }
  if (aspect === "domains") {
    return (
      <div>
        <ProteinHeader name={name} p={p} />
        <Field label="domains" value={p.domains} />
      </div>
    );
  }

  // full record
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

function ArtifactList({ items }) {
  return (
    <ul className="text-gray-300">
      {items.map((it, i) => {
        const p = it.payload;
        return (
          <li key={i} className="py-1 flex items-baseline gap-4">
            <span className="w-24 text-gray-200">{it.name}</span>
            {p && p.role && (
              <span className="text-gray-500 italic text-xs w-40">{p.role}</span>
            )}
            {p && p.function && (
              <span className="text-gray-400 text-xs flex-1 truncate">{p.function}</span>
            )}
            <span className="text-gray-600 text-xs">d={it.distance.toFixed(3)}</span>
          </li>
        );
      })}
    </ul>
  );
}

function ArtifactText({ lines }) {
  return (
    <div className="text-gray-300">
      {lines.map((line, i) => (
        <p key={i}>{line}</p>
      ))}
    </div>
  );
}

function Artifact({ result }) {
  if (!result) return null;
  switch (result.kind) {
    case "protein":
      return <ArtifactProtein name={result.name} payload={result.payload} aspect={result.aspect} />;
    case "protein_compare":
      return <ArtifactCompare a={result.a} b={result.b} />;
    case "list":
      return <ArtifactList items={result.items} />;
    case "text":
      return <ArtifactText lines={result.lines} />;
    default:
      return null;
  }
}

// ────────────────────────────────────────────────────────────
//  Terminal component.
// ────────────────────────────────────────────────────────────

export default function BuheraTerminal() {
  const kernelRef = useRef(null);
  const inputRef = useRef(null);
  const historyRef = useRef(null);
  const [entries, setEntries] = useState([]);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    kernelRef.current = bootKernel();
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

  async function submit(text) {
    if (!text.trim() || busy) return;
    setBusy(true);
    const eId = Date.now();
    setEntries((e) => [...e, { id: eId, obs: text, result: null, thinking: true, error: null }]);
    setDraft("");

    await new Promise((r) => setTimeout(r, 140 + Math.random() * 180));

    try {
      const vahera = translate(text);
      const result = executeVahera(vahera, kernelRef.current);
      setEntries((e) =>
        e.map((x) => (x.id === eId ? { ...x, result, thinking: false } : x))
      );
    } catch (err) {
      setEntries((e) =>
        e.map((x) =>
          x.id === eId ? { ...x, error: err.message || String(err), thinking: false } : x
        )
      );
    }
    setBusy(false);
    if (inputRef.current) inputRef.current.focus();
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit(draft);
    }
  }

  function onInput(e) {
    setDraft(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = e.target.scrollHeight + "px";
  }

  return (
    <div className="fixed inset-0 bg-black text-gray-300 flex flex-col px-16 py-10 md:px-8 md:py-6 font-mono text-sm leading-relaxed">
      <div
        ref={historyRef}
        className="flex-1 overflow-y-auto pb-4"
        style={{ scrollbarWidth: "none" }}
      >
        {entries.map((e) => (
          <div key={e.id} className="mb-8 animate-fade">
            <div className="text-gray-200 whitespace-pre-wrap mb-2">{e.obs}</div>
            {e.thinking && <span className="text-gray-600 italic">...</span>}
            {e.error && <p className="text-gray-500">[{e.error}]</p>}
            {e.result && <Artifact result={e.result} />}
          </div>
        ))}
      </div>

      <div className="flex items-start pt-3">
        <span className="text-gray-700 mr-2 select-none pt-0.5"></span>
        <textarea
          ref={inputRef}
          value={draft}
          onChange={onInput}
          onKeyDown={onKeyDown}
          rows={1}
          spellCheck={false}
          autoFocus
          className="flex-1 bg-transparent border-none outline-none resize-none text-gray-200 font-mono text-sm leading-relaxed"
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
