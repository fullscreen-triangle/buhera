/* ============================================================================
 * RunnableCell
 *
 * A code block in a tutorial page that runs against the real federation.
 * Shows the source, a Run button, and (once run) the actual result inline.
 * Uses the same `runInput` + `Artifact` machinery as the main terminal, so
 * everything you can do in the terminal you can do here.
 *
 * Multi-line cells are supported (the buhera terminal takes multi-line
 * input; scripts get joined with \n).
 * ========================================================================== */

import { useState, useRef } from "react";
import dynamic from "next/dynamic";
import { runInput } from "@/lib/runtime/run-input";

// The Artifact component pulls in the terminal's whole render tree
// (charts, workspaces, etc.). Load it dynamically, ssr-off, so the
// tutorial pages stay SSG-friendly and only pay the bundle cost when a
// cell is actually run.
const Artifact = dynamic(
  () => import("@/components/BuheraTerminal").then((m) => m.Artifact),
  { ssr: false }
);

export default function RunnableCell({ source, ctxRef }) {
  const [busy, setBusy] = useState(false);
  const [output, setOutput] = useState(null);
  const [error, setError] = useState(null);
  const [ran, setRan] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(source);
  const textareaRef = useRef(null);

  async function onRun() {
    if (busy) return;
    setBusy(true);
    setError(null);
    setOutput(null);
    try {
      const ctx = ctxRef.current;
      const envelope = await runInput(editing ? draft : source, ctx);
      if (envelope.kind === "error") {
        setError(envelope.message);
      } else {
        setOutput(envelope);
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setBusy(false);
      setRan(true);
    }
  }

  function onEdit() {
    setEditing(true);
    // Focus after next tick so the textarea exists.
    setTimeout(() => textareaRef.current?.focus(), 0);
  }

  function onReset() {
    setEditing(false);
    setDraft(source);
    setOutput(null);
    setError(null);
    setRan(false);
  }

  const displayed = editing ? draft : source;

  return (
    <div className="my-4 border border-gray-800 rounded overflow-hidden bg-gray-900">
      <div className="flex items-center justify-between px-3 py-1 bg-gray-950 border-b border-gray-800 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <button
            onClick={onRun}
            disabled={busy}
            className={`px-2 py-0.5 rounded font-mono ${
              busy
                ? "bg-gray-800 text-gray-500 cursor-not-allowed"
                : "bg-green-900 text-green-300 hover:bg-green-800"
            }`}
          >
            {busy ? "…" : "▶ run"}
          </button>
          {!editing && (
            <button
              onClick={onEdit}
              className="text-gray-500 hover:text-gray-300"
            >
              edit
            </button>
          )}
          {ran && (
            <button
              onClick={onReset}
              className="text-gray-500 hover:text-gray-300"
            >
              reset
            </button>
          )}
        </div>
        <span className="font-mono">buhera</span>
      </div>

      {editing ? (
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          rows={Math.max(2, draft.split("\n").length)}
          spellCheck={false}
          className="w-full bg-gray-900 border-0 outline-none resize-y text-sm font-mono text-gray-100 p-3 leading-relaxed"
          style={{ caretColor: "#2a9d8f" }}
        />
      ) : (
        <pre className="text-sm font-mono text-gray-100 p-3 whitespace-pre-wrap overflow-x-auto">
          {displayed}
        </pre>
      )}

      {(output || error) && (
        <div className="border-t border-gray-800 p-3 bg-black">
          {error && (
            <div className="text-red-400 text-sm">
              <span className="text-red-500">error:</span> {error}
            </div>
          )}
          {output?.kind === "text" && (
            <div className="text-gray-300 text-sm">
              {output.lines.map((l, i) => (
                <p key={i} className="whitespace-pre-wrap">{l}</p>
              ))}
            </div>
          )}
          {output?.kind === "artifact" && output.result && (
            <div className="text-sm">
              <Artifact result={output.result} />
            </div>
          )}
          {output?.kind === "multi" && Array.isArray(output.results) && (
            <div className="space-y-4">
              {output.results.map((r, i) => (
                <div key={i} className="text-sm">
                  <Artifact result={r} />
                </div>
              ))}
            </div>
          )}
          {output?.kind === "external" && (
            <p className="text-gray-400 text-sm italic">{output.message}</p>
          )}
          {output?.kind === "noop" && (
            <p className="text-gray-500 text-sm italic">(no-op — empty input)</p>
          )}
        </div>
      )}
    </div>
  );
}
