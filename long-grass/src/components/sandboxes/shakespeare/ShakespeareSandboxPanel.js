/* ============================================================================
 * ShakespeareSandboxPanel — the REAL shakespeare (.shk) sandbox, embedded.
 *
 * Runs the real Shakespeare interpreter in the browser
 * (performPlay(src, receiver, lesson) -> { console, charts }). The interpreter
 * parses the play live and, for heavy verbs (track/fold), replays the lesson's
 * baked oracle — the monograph's own validated numbers. Seeded with the real
 * `07_electron-chain.shk` lesson (its source AND oracle travel together, as the
 * source app requires), whose output is the femtosecond electron-transfer trace
 * (et-trace) + the electron-transfer-chain GLB structure.
 *
 * Interpreter + renderers reproduced from levinthal/cytochrome (PRIVATE source)
 * as grounding for this page — not copied wholesale.
 * ========================================================================== */

import { useState, useCallback, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import { performPlay, newReceiver } from "@/lib/sandboxes/shakespeare/shakespeare";
import ShakespeareChart from "./ShakespeareChart";
import SandboxFrame from "../SandboxFrame";

// StructureOutput mounts a WebGL Canvas — load it client-side only.
const StructureOutput = dynamic(() => import("./StructureOutput"), { ssr: false });

const ACCENT = "#58E6D9";

// The real 07_electron-chain lesson — source + baked oracle. The interpreter
// needs BOTH: `track` replays lesson.oracle rather than recomputing.
const ELECTRON_CHAIN_LESSON = {
  id: "07_electron-chain",
  n: 7,
  title: "The electron-transfer chain",
  subtitle: "femtosecond multi-hop electronic leaves",
  paper: "Paper 4 · multi-hop-et-chain",
  verbs: ["electronic", "track"],
  charts: ["et-trace"],
  requires: "06_ch-rebound",
  src: `-- 07_electron-chain.shk
-- Electrons are their own leaf class, addressed by partition
-- coordinates (n, l, m, s). The reductase delivers two electrons
-- to the heme through a multi-hop chain, resolved to femtoseconds.

receiver bio
floor 3.7e-4

e1 := electronic "FAD -> FMN -> heme"

trace := track e1 in E
           until converge
           yield trajectory

observe trace
-- a time-indexed sequence of partition cells (fs resolution)
`,
  oracle: {
    verdict: "PASS",
    hops: ["FAD", "FMN", "heme-Fe"],
    timescale: "femtosecond",
    selection_rule: "delta s_orbital = 0 (conserved)",
    note: "electronic-leaf trajectory; QND under categorical observation",
  },
};

const LINE_COLOR = {
  cmd: "#c586c0", ok: "#6a9955", pass: "#4ec9b0", err: "#f48771",
  clock: "#dcdcaa", dim: "#6b7280", log: "#d4d4d4",
};

export default function ShakespeareSandboxPanel() {
  const [source, setSource] = useState(ELECTRON_CHAIN_LESSON.src);
  const [output, setOutput] = useState(null); // { console, charts }

  const run = useCallback((code) => {
    const src = typeof code === "string" ? code : source;
    // A fresh receiver each run keeps the panel self-contained (the full IDE
    // accumulates across lessons; here one play is the whole apparatus).
    const recv = newReceiver();
    const lesson = { ...ELECTRON_CHAIN_LESSON, src };
    setOutput(performPlay(src, recv, lesson));
  }, [source]);

  // Auto-run the seeded electron-chain play on mount.
  useEffect(() => { run(ELECTRON_CHAIN_LESSON.src); /* eslint-disable-next-line */ }, []);

  const { structure, chartEntries } = useMemo(() => {
    const charts = output?.charts ?? {};
    let structure = null;
    const chartEntries = [];
    for (const [name, data] of Object.entries(charts)) {
      if (name === "structure") structure = data;
      else chartEntries.push([name, data]);
    }
    return { structure, chartEntries };
  }, [output]);

  return (
    <SandboxFrame
      title="shakespeare"
      subtitle="07_electron-chain.shk — femtosecond electron-transfer trace"
      accent={ACCENT}
    >
      {({ editorCollapsed }) => (
        <div
          className="grid min-h-0 flex-1"
          style={{ gridTemplateColumns: editorCollapsed ? "100%" : "40% 60%" }}
        >
          {/* Editor + console */}
          {!editorCollapsed && (
            <div className="flex min-h-0 flex-col border-r border-neutral-800">
              <div className="flex items-center justify-between px-3 py-1.5">
                <span className="text-[10px] uppercase tracking-widest text-neutral-500">source · .shk</span>
                <button
                  onClick={() => run(source)}
                  className="rounded bg-[#58E6D9] px-3 py-0.5 text-[12px] font-medium text-[#0a0a0a] hover:brightness-110"
                >
                  ▶ run
                </button>
              </div>
              <textarea
                value={source}
                onChange={(e) => setSource(e.target.value)}
                spellCheck={false}
                className="min-h-0 flex-1 resize-none bg-neutral-950 p-3 font-mono text-[12px] leading-relaxed text-neutral-200 outline-none"
                style={{ minHeight: 180 }}
              />
              <div className="h-48 shrink-0 overflow-y-auto border-t border-neutral-800 bg-black p-2 font-mono text-[11px] leading-relaxed">
                {!output ? (
                  <span className="text-neutral-600">Press run to perform the play.</span>
                ) : output.console.map((l, i) => (
                  <div key={i} className="whitespace-pre-wrap" style={{ color: LINE_COLOR[l.kind] || "#d4d4d4" }}>{l.text}</div>
                ))}
              </div>
            </div>
          )}

          {/* Structure + charts */}
          <div className="min-h-0 overflow-y-auto p-3">
            {editorCollapsed && (
              <div className="mb-3 flex justify-end">
                <button
                  onClick={() => run(source)}
                  className="rounded bg-[#58E6D9] px-3 py-0.5 text-[12px] font-medium text-[#0a0a0a] hover:brightness-110"
                >
                  ▶ run
                </button>
              </div>
            )}
            {structure && (
              <div className="mb-3">
                <StructureOutput data={structure} />
              </div>
            )}
            <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
              {chartEntries.map(([name, data]) => (
                <ShakespeareChart key={name} name={name} data={data} />
              ))}
            </div>
            {!structure && chartEntries.length === 0 && (
              <div className="flex h-full items-center justify-center text-sm text-neutral-600">
                Run the play to emit its structure + charts.
              </div>
            )}
          </div>
        </div>
      )}
    </SandboxFrame>
  );
}
