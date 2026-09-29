/* ============================================================================
 * PageView — draws one page of the book.
 *
 * A page is an inert snapshot (lib/surface/book.js): the words that started
 * the step and the result that came back. Everything is drawn flat — the
 * whole tree sits inside <FlatContext.Provider value={true}>, so no section
 * of any artifact is hidden behind a toggle. What is on the page is all there
 * is to know about it.
 *
 * The only live element a page carries is on a module page: its actions.
 * Picking one does not change this page; it starts a new step (onAct), which
 * forks to the end of the book like any other step.
 * ========================================================================== */

import { FlatContext } from "@/components/artifacts/disclosure";
import { Artifact } from "@/components/artifacts/Artifact";
import { isTemplate } from "@/lib/surface/edges";

function Envelope({ envelope, onAct }) {
  if (!envelope) return null;
  switch (envelope.kind) {
    case "artifact":
      return envelope.result ? <Artifact result={envelope.result} /> : <Quiet>(no result)</Quiet>;
    case "multi":
      return (
        <div className="space-y-6">
          {(envelope.results || []).map((r, i) => (
            <div key={i}><Artifact result={r} /></div>
          ))}
        </div>
      );
    case "text":
      return <Artifact result={{ kind: "text", lines: envelope.lines || [] }} />;
    case "error":
      return <p className="text-rose-300/80">[{envelope.message}]</p>;
    case "external":
      return <Quiet>{envelope.message}</Quiet>;
    case "module":
      return <ModuleCard envelope={envelope} onAct={onAct} />;
    default:
      return <Quiet>(nothing to show)</Quiet>;
  }
}

function Quiet({ children }) {
  return <p className="text-gray-500 italic">{children}</p>;
}

const EDGE_LABEL = { top: "top edge", right: "right edge", bottom: "bottom edge" };

function ModuleCard({ envelope, onAct }) {
  const { module: m, edge, glance } = envelope;
  return (
    <div>
      <div className="flex items-baseline gap-4 mb-2">
        <span className="text-white text-base">{m.id}</span>
        {edge && <span className="text-gray-600 text-xs">{EDGE_LABEL[edge]}</span>}
      </div>
      {m.description && <p className="text-gray-400 mb-6 max-w-2xl">{m.description}</p>}

      {glance && (
        <div className="mb-8">
          {glance.result ? <Artifact result={glance.result} /> : <Quiet>(no state reported)</Quiet>}
        </div>
      )}

      {m.instructions.length > 0 && (
        <div>
          <div className="text-gray-600 text-xs mb-2">actions</div>
          <ul className="space-y-1">
            {m.instructions.map((ins, i) => {
              const template = isTemplate(ins);
              return (
                <li key={i}>
                  <button
                    type="button"
                    onClick={() => onAct?.(ins, template)}
                    className="text-left text-xs font-mono text-gray-400 hover:text-teal-300 break-all"
                    title={template ? "fill in and run" : "run as a new step"}
                  >
                    {ins}
                    {template && <span className="ml-2 text-gray-700">…</span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

function SourceLine({ page }) {
  const src = page.source || {};
  const words = src.type === "module" ? src.moduleId : src.text;
  return (
    <div className="mb-6">
      <div className="text-gray-200 whitespace-pre-wrap">{words}</div>
      {page.from != null && (
        <div className="text-gray-700 text-xs mt-1">continued from page {page.from}</div>
      )}
    </div>
  );
}

export default function PageView({ page, onAct }) {
  return (
    <FlatContext.Provider value={true}>
      <SourceLine page={page} />
      <Envelope envelope={page.envelope} onAct={onAct} />
    </FlatContext.Provider>
  );
}
