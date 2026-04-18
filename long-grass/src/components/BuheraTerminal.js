import { useEffect, useRef, useState } from "react";
import { Kernel } from "@/lib/kernel";
import { embedMolecule } from "@/lib/substrate";
import { translate } from "@/lib/translator";
import { executeVahera } from "@/lib/vahera";
import { COMPOUNDS } from "@/lib/compounds";

const PROP_LABELS = {
  molecular_weight: "mw",
  boiling_point_c: "bp (°C)",
  melting_point_c: "mp (°C)",
  density_g_cm3: "density (g/cm³)",
  n_atoms: "atoms",
};

function bootKernel() {
  const k = new Kernel(12);
  for (const name of Object.keys(COMPOUNDS)) {
    const coord = embedMolecule(name, COMPOUNDS[name]);
    k.allocate(coord, COMPOUNDS[name], { name, formula: COMPOUNDS[name].formula });
  }
  return k;
}

function ArtifactMolecule({ compound }) {
  return (
    <div className="text-gray-300">
      <div className="mb-1">
        <span className="text-white">{compound.name}</span>
        {compound.formula && (
          <span className="ml-4 text-gray-500">{compound.formula}</span>
        )}
      </div>
      <div className="text-sm">
        {Object.keys(PROP_LABELS).map((key) => {
          if (compound.payload[key] === undefined) return null;
          return (
            <div key={key}>
              <span className="text-gray-500">{PROP_LABELS[key]}</span>
              <span className="ml-3 text-gray-300">{String(compound.payload[key])}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ArtifactList({ items }) {
  return (
    <ul className="text-gray-300">
      {items.map((it, i) => (
        <li key={i} className="py-0.5 flex">
          <span className="flex-1">{it.name}</span>
          <span className="text-gray-600 ml-6">d={it.distance.toFixed(3)}</span>
        </li>
      ))}
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
    case "molecule": return <ArtifactMolecule compound={result.compound} />;
    case "list":     return <ArtifactList items={result.items} />;
    case "text":     return <ArtifactText lines={result.lines} />;
    default:         return null;
  }
}

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

    await new Promise((r) => setTimeout(r, 120 + Math.random() * 160));

    try {
      const vahera = translate(text);
      const result = executeVahera(vahera, kernelRef.current, COMPOUNDS);
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
          <div key={e.id} className="mb-7 animate-fade">
            <div className="text-gray-200 whitespace-pre-wrap mb-1">{e.obs}</div>
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
