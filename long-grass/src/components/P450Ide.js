/* ============================================================================
 * P450Ide — a VSCode-styled workspace that runs the P450 model.
 *
 * Three panes: an activity/file-tree rail on the left, an editor in the
 * middle, and an output panel on the right/below. "Running" a file feeds its
 * source through runInput() against the live federation and renders the
 * result with the terminal's own <Artifact> — so the DSL output here is byte
 * for byte the output the terminal produces.
 *
 * Some files carry a `program` (a sequence of setup commands, e.g. building
 * the ckg graph) run before the final `source`; only the last result is shown.
 *
 * The IDE owns a single runtime context so files see each other's state — the
 * ckg graph built by build-cycle.ckg is the graph graph.ckg reads back.
 * ========================================================================== */

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { createRuntimeContext, runInput } from "@/lib/runtime/run-input";
import { bootstrapFederation } from "@/lib/runtime/bootstrap";
import { IDE_FILES, fileByPath, DEFAULT_FILE } from "@/lib/p450-ide-files";

const Artifact = dynamic(
  () => import("@/components/BuheraTerminal").then((m) => m.Artifact),
  { ssr: false }
);

function FileTree({ activePath, onOpen }) {
  return (
    <div className="text-[13px] leading-6 select-none">
      <div className="px-3 py-2 text-[11px] uppercase tracking-wider text-gray-500">
        P450 · Explorer
      </div>
      {IDE_FILES.map((group) => (
        <div key={group.folder} className="mb-1">
          <div className="px-3 py-0.5 text-gray-300 flex items-center gap-1.5">
            <span className="text-gray-500">▾</span>
            <span aria-hidden>{group.icon}</span>
            <span className="font-medium">{group.folder}</span>
          </div>
          <div className="text-gray-500 text-[10px] px-3 pb-1 pl-8 italic">
            {group.hint}
          </div>
          {group.files.map((f) => {
            const path = `${group.folder}/${f.name}`;
            const active = path === activePath;
            return (
              <button
                key={path}
                onClick={() => onOpen(path)}
                className={`w-full text-left pl-8 pr-3 py-0.5 flex items-center gap-2 ${
                  active
                    ? "bg-[#37373d] text-white"
                    : "text-gray-400 hover:bg-[#2a2d2e] hover:text-gray-200"
                }`}
              >
                <span className="text-gray-600" aria-hidden>
                  ▪
                </span>
                <span className="font-mono truncate">{f.name}</span>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function OutputPanel({ state }) {
  if (state.status === "idle") {
    return (
      <p className="text-gray-600 text-sm italic p-4">
        ▶ Run a file to dispatch it against the live federation. The output is
        the same artifact the terminal renders.
      </p>
    );
  }
  if (state.status === "running") {
    return (
      <p className="text-yellow-500/80 text-sm p-4 font-mono">
        running {state.path}
        {state.step ? `  · step ${state.step}` : ""}…
      </p>
    );
  }
  if (state.status === "error") {
    return (
      <div className="p-4 text-sm">
        <span className="text-red-500 font-mono">error:</span>{" "}
        <span className="text-red-400">{state.message}</span>
      </div>
    );
  }
  // done
  return (
    <div className="p-4 space-y-4">
      {state.note && (
        <p className="text-emerald-400/80 text-xs font-mono">{state.note}</p>
      )}
      {state.envelope?.kind === "text" &&
        state.envelope.lines.map((l, i) => (
          <p key={i} className="text-gray-300 text-sm whitespace-pre-wrap font-mono">
            {l}
          </p>
        ))}
      {state.envelope?.kind === "artifact" && state.envelope.result && (
        <div className="text-sm">
          <Artifact result={state.envelope.result} />
        </div>
      )}
      {state.envelope?.kind === "multi" &&
        Array.isArray(state.envelope.results) && (
          <div className="space-y-4">
            {state.envelope.results.map((r, i) => (
              <div key={i} className="text-sm">
                <Artifact result={r} />
              </div>
            ))}
          </div>
        )}
      {state.envelope?.kind === "external" && (
        <p className="text-gray-400 text-sm italic">{state.envelope.message}</p>
      )}
      {state.envelope?.kind === "noop" && (
        <p className="text-gray-600 text-sm italic">(no-op)</p>
      )}
    </div>
  );
}

export default function P450Ide() {
  const ctxRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [activePath, setActivePath] = useState(DEFAULT_FILE);
  const [output, setOutput] = useState({ status: "idle" });

  useEffect(() => {
    ctxRef.current = createRuntimeContext();
    bootstrapFederation();
    setReady(true);
  }, []);

  const file = useMemo(() => fileByPath(activePath), [activePath]);

  async function runFile() {
    if (!ready || !file) return;
    setOutput({ status: "running", path: activePath });
    try {
      // Files with a `program` run their setup steps first (e.g. building the
      // ckg graph), then the final `source` produces the shown artifact.
      const program = file.program || [];
      for (let i = 0; i < program.length; i++) {
        setOutput({ status: "running", path: activePath, step: i + 1 });
        const env = await runInput(program[i], ctxRef.current);
        if (env.kind === "error") {
          setOutput({ status: "error", message: `step ${i + 1}: ${env.message}` });
          return;
        }
      }
      const envelope = await runInput(file.source, ctxRef.current);
      if (envelope.kind === "error") {
        setOutput({ status: "error", message: envelope.message });
        return;
      }
      const note = program.length
        ? `ran ${program.length} setup step${program.length === 1 ? "" : "s"}, then ${activePath}`
        : `ran ${activePath}`;
      setOutput({ status: "done", envelope, note });
    } catch (err) {
      setOutput({ status: "error", message: err.message || String(err) });
    }
  }

  return (
    <div className="rounded-lg overflow-hidden border border-[#2b2b2b] bg-[#1e1e1e] text-gray-200 shadow-2xl">
      {/* title bar */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-[#323233] border-b border-[#2b2b2b] text-xs">
        <span className="flex gap-1.5">
          <span className="w-3 h-3 rounded-full bg-[#ff5f56]" />
          <span className="w-3 h-3 rounded-full bg-[#ffbd2e]" />
          <span className="w-3 h-3 rounded-full bg-[#27c93f]" />
        </span>
        <span className="ml-2 text-gray-400 font-mono">
          buhera — cytochrome-p450 workspace
        </span>
        <span className="ml-auto text-gray-500">
          {ready ? "● federation live" : "booting…"}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[220px_minmax(0,1fr)] lg:grid-cols-[220px_minmax(0,1.1fr)_minmax(0,1fr)]">
        {/* explorer rail */}
        <div className="bg-[#252526] border-r border-[#2b2b2b] py-1 max-h-[560px] overflow-y-auto">
          <FileTree activePath={activePath} onOpen={setActivePath} />
        </div>

        {/* editor */}
        <div className="flex flex-col border-r border-[#2b2b2b] min-w-0">
          <div className="flex items-center bg-[#2d2d2d] border-b border-[#2b2b2b] text-xs">
            <div className="px-3 py-1.5 bg-[#1e1e1e] text-gray-200 font-mono border-r border-[#2b2b2b]">
              {file ? file.name : "—"}
            </div>
            <button
              onClick={runFile}
              disabled={!ready}
              className={`ml-auto mr-2 my-1 px-3 py-1 rounded font-mono text-xs ${
                ready
                  ? "bg-[#0e639c] hover:bg-[#1177bb] text-white"
                  : "bg-gray-700 text-gray-400 cursor-not-allowed"
              }`}
            >
              ▶ Run
            </button>
          </div>
          {file?.blurb && (
            <div className="px-4 py-2 text-[12px] text-gray-400 leading-relaxed border-b border-[#2b2b2b] bg-[#1e1e1e]">
              {file.blurb}
            </div>
          )}
          <pre className="flex-1 p-4 overflow-auto text-[13px] font-mono text-gray-100 leading-relaxed max-h-[440px]">
            <code>{file ? file.editor : ""}</code>
          </pre>
        </div>

        {/* output */}
        <div className="bg-[#181818] min-w-0 max-h-[560px] overflow-y-auto lg:border-t-0 border-t border-[#2b2b2b]">
          <div className="px-3 py-1.5 bg-[#2d2d2d] border-b border-[#2b2b2b] text-[11px] uppercase tracking-wider text-gray-500">
            Output — live federation
          </div>
          <OutputPanel state={output} />
        </div>
      </div>
    </div>
  );
}
