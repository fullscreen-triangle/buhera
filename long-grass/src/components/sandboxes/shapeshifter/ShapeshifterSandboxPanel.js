/* ============================================================================
 * ShapeshifterSandboxPanel — the REAL shapeshifter sandbox, embedded.
 *
 * Runs the real Shapeshifter compiler in the browser (already vendored as
 * @lavoisier/shapeshifter/compiler): compileStage(source) -> {ok, ast, ir, term}
 * then executeStage(ast) -> {result, logs, term, workspace}. The workspace's
 * `records` entry is a live PredictedRecord[] — the same array the live site
 * feeds its dashboard. Seeded with the real `proteomics_experiment.ss`
 * (run_proteomics over HSA/HBB/ENO1).
 *
 * Four tabs:
 *   Charts  — SandboxCharts.js, lifted verbatim (zero-dep, 9 SVG frames) over
 *             the live records.
 *   Records — a lighter records/summary view over the SAME PredictedRecord[]
 *             (stats + class legend + partition scatter + table). This is NOT a
 *             pixel-clone of the live site's crossfilter/GPU ResultsDashboard —
 *             it is an honest lighter view over the same live records, so the
 *             panel carries no heavy crossfilter/three deps.
 *   Console — the interpreter's execution logs + terminal stream.
 *   IR      — the compiled intermediate representation (compileStage.ir).
 * ========================================================================== */

import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { compileStage, executeStage } from "@lavoisier/shapeshifter/compiler";
import SandboxCharts from "./SandboxCharts";

const T = {
  editor: "#1e1e1e", editorFg: "#d4d4d4",
  tabInactive: "#2d2d2d", tabActive: "#1e1e1e",
  tabFg: "#969696", tabFgActive: "#ffffff", border: "#3c3c3c",
  accent: "#0e639c", accentBright: "#007acc", gutter: "#858585",
};

const CLASS_COLORS = {
  PC: "#5fa8d3", PE: "#e07a7a", PS: "#b388eb", PG: "#e493b3",
  SM: "#7cc77c", Cer: "#e6a456", TAG: "#cdc15c", DAG: "#a07a5e",
  LPC: "#a8b2bd", CE: "#9cc4d8", FA: "#e8c598",
  HSA: "#60a5fa", HBB: "#f87171", ENO1: "#34d399", CYCS: "#a78bfa", CASE: "#fb923c",
};

// The real proteomics_experiment.ss seed script.
const SS_PROTEOMICS = `\
// Proteomics virtual experiment.
// Predicts tryptic peptide library from common plasma protein standards.
// Multiply-charged ESI adducts are selected by peptide mass.

import lavoisier.instrument

objective PlasmaProteomics:
    target: "predict tryptic peptide library from plasma proteins"
    success_criteria: "records > 50"

instrument OrbitrapFusion:
    analyzer: "orbitrap"
    polarity: "+"
    collision_energy: 28

phase Design:
    proteins = ["HSA", "HBB", "ENO1"]

phase VirtualRun:
    records = lavoisier.instrument.run_proteomics(
        proteins: proteins,
        length_min: 7,
        length_max: 20,
        mc_max: 1,
        polarity: "+",
        analyser: "orbitrap",
        collision_energy: 28
    )
`;

/* ── Partition scatter over the live records ── */
function PartitionScatter({ records }) {
  if (!records.length) return (
    <div className="flex items-center justify-center text-[11px]" style={{ height: 150, color: "#555" }}>no records</div>
  );
  const W = 380, H = 150, PL = 32, PR = 8, PT = 8, PB = 28;
  const pW = W - PL - PR, pH = H - PT - PB;
  const mzMin = Math.min(...records.map(r => r.precursorMz));
  const mzMax = Math.max(...records.map(r => r.precursorMz));
  const nMax = Math.max(...records.map(r => r.n), 1);
  const toX = mz => PL + ((mz - mzMin) / (mzMax - mzMin || 1)) * pW;
  const toY = n => H - PB - ((n - 1) / nMax) * pH;
  const sample = records.length > 400
    ? records.filter((_, i) => i % Math.ceil(records.length / 400) === 0)
    : records;
  const nTicks = [1, 2, 3, 4, 5].filter(v => v <= nMax + 1);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ fontFamily: "monospace", display: "block" }}>
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke="#3a3a3a" />
      <line x1={PL} y1={PT} x2={PL} y2={H - PB} stroke="#3a3a3a" />
      {nTicks.map(v => (
        <g key={v}>
          <line x1={PL - 3} y1={toY(v)} x2={PL} y2={toY(v)} stroke="#3a3a3a" />
          <text x={PL - 5} y={toY(v) + 4} textAnchor="end" fontSize={8} fill="#666">{v}</text>
        </g>
      ))}
      <text x={PL - 14} y={H / 2 + 4} fontSize={8} fill="#555" transform={`rotate(-90,${PL - 14},${H / 2})`}>n</text>
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={8} fill="#555">m/z</text>
      {sample.map((r, i) => (
        <circle key={i} cx={toX(r.precursorMz)} cy={toY(r.n)} r={1.8}
          fill={CLASS_COLORS[r.analyteClass] || "#555"} opacity={0.55} />
      ))}
    </svg>
  );
}

/* ── Lighter records view over the live PredictedRecord[] ── */
function RecordsPanel({ records }) {
  const summary = useMemo(() => {
    if (!records.length) return { count: 0, perClass: {}, mzRange: [0, 0] };
    const perClass = {};
    let lo = Infinity, hi = -Infinity;
    for (const r of records) {
      const c = r.analyteClass || "?";
      perClass[c] = (perClass[c] || 0) + 1;
      if (r.precursorMz < lo) lo = r.precursorMz;
      if (r.precursorMz > hi) hi = r.precursorMz;
    }
    return { count: records.length, perClass, mzRange: [lo, hi] };
  }, [records]);

  if (!records.length) {
    return <div className="flex h-full items-center justify-center text-[12px]" style={{ color: "#444" }}>Run a .ss script to see records</div>;
  }
  const classes = Object.keys(summary.perClass);
  const top = records.slice(0, 20);
  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-3">
      <div className="grid grid-cols-3 gap-2">
        {[
          ["Records", summary.count],
          ["Classes", classes.length],
          ["m/z range", `${summary.mzRange[0]?.toFixed(0)}–${summary.mzRange[1]?.toFixed(0)}`],
        ].map(([label, val]) => (
          <div key={label} className="rounded p-2 text-center" style={{ background: "#2a2d2e", border: `1px solid ${T.border}` }}>
            <div className="text-[9px] uppercase tracking-wider" style={{ color: "#666" }}>{label}</div>
            <div className="font-mono text-[13px]" style={{ color: T.editorFg }}>{val}</div>
          </div>
        ))}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {classes.map(cls => (
          <span key={cls} className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px]" style={{ background: "#2a2d2e", border: `1px solid ${T.border}` }}>
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: CLASS_COLORS[cls] || "#555" }} />
            <span style={{ color: T.editorFg }}>{cls}</span>
            <span style={{ color: "#666" }}>{summary.perClass[cls]}</span>
          </span>
        ))}
      </div>
      <PartitionScatter records={records} />
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#666", borderBottom: `1px solid ${T.border}` }}>
              {["Analyte", "Class", "m/z", "Adduct", "n", "ℓ", "Sk"].map(h => (
                <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {top.map((r, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #2a2a2a", color: T.editorFg }}>
                <td className="truncate py-0.5 pr-3" style={{ maxWidth: 120 }}>{r.analyte}</td>
                <td className="pr-3" style={{ color: CLASS_COLORS[r.analyteClass] || "#555" }}>{r.analyteClass}</td>
                <td className="pr-3">{r.precursorMz?.toFixed(3)}</td>
                <td className="pr-3" style={{ color: "#9cdcfe" }}>{r.adduct}</td>
                <td className="pr-3">{r.n}</td>
                <td className="pr-3">{r.l}</td>
                <td className="pr-3">{r.sentropyVec?.sk?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {records.length > 20 && (
          <div className="pt-1 text-[10px]" style={{ color: "#555" }}>+ {records.length - 20} more records</div>
        )}
      </div>
    </div>
  );
}

/* ── Terminal / console stream ── */
function TerminalView({ term }) {
  const endRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView({ block: "end" }); }, [term]);
  const style = (s) => s === "stderr" ? { color: "#f48771" } : s === "stage" ? { color: "#4ec9b0", fontWeight: 600 } : { color: "#d4d4d4" };
  return (
    <div className="h-full overflow-y-auto p-2 font-mono text-[12px] leading-[1.55]">
      {term.length === 0 ? (
        <div className="px-1 pt-1" style={{ color: "#5a5a5a" }}>Press Run to execute the active script.</div>
      ) : term.map((l, i) => l.stream === "stage" ? (
        <div key={i} className="mt-1 flex items-center gap-2"><span style={{ color: "#ce9178" }}>$</span><span style={style("stage")}>{l.text}</span></div>
      ) : (
        <div key={i} className="whitespace-pre-wrap break-words px-1" style={style(l.stream)}>{l.text}</div>
      ))}
      <div ref={endRef} />
    </div>
  );
}

/* ── Editor ── */
function Editor({ value, onChange }) {
  const gutterRef = useRef(null);
  const lines = value.split("\n");
  const syncScroll = (e) => { if (gutterRef.current) gutterRef.current.scrollTop = e.target.scrollTop; };
  return (
    <div className="flex min-h-0 flex-1" style={{ background: T.editor }}>
      <div ref={gutterRef} className="select-none overflow-hidden py-3 text-right font-mono text-[13px] leading-[1.5]" style={{ color: T.gutter, minWidth: 44, paddingRight: 12 }}>
        {lines.map((_, i) => <div key={i}>{i + 1}</div>)}
      </div>
      <textarea
        value={value} onChange={(e) => onChange(e.target.value)} onScroll={syncScroll} spellCheck={false}
        className="min-h-0 flex-1 resize-none border-0 bg-transparent py-3 pr-4 font-mono text-[13px] leading-[1.5] outline-none"
        style={{ color: T.editorFg, tabSize: 2, caretColor: "#fff" }}
      />
    </div>
  );
}

export default function ShapeshifterSandboxPanel() {
  const [source, setSource] = useState(SS_PROTEOMICS);
  const [ir, setIr] = useState("");
  const [logs, setLogs] = useState([]);
  const [term, setTerm] = useState([]);
  const [workspace, setWorkspace] = useState([]);
  const [tab, setTab] = useState("charts");
  const [running, setRunning] = useState(false);

  const records = useMemo(() => {
    const rec = (workspace || []).find(w => w.kind === "records");
    return rec ? rec.value : [];
  }, [workspace]);

  const run = useCallback(async (code) => {
    const src = typeof code === "string" ? code : source;
    setRunning(true);
    const { ok, ast, ir: i, term: compileTerm } = compileStage(src);
    setIr(i || "");
    if (!ok) {
      setTerm(compileTerm || [{ stream: "stderr", text: "compile failed" }]);
      setWorkspace([]); setLogs([]); setRunning(false);
      return;
    }
    await new Promise(r => setTimeout(r, 16));
    const { logs: l, term: execTerm, workspace: ws } = executeStage(ast);
    setLogs(l || []);
    setTerm([...(compileTerm || []), ...(execTerm || [])]);
    setWorkspace(ws || []);
    setRunning(false);
  }, [source]);

  // Auto-run the seeded proteomics experiment on mount.
  useEffect(() => { run(SS_PROTEOMICS); /* eslint-disable-next-line */ }, []);

  const tabs = [
    { id: "charts", label: "Charts" },
    { id: "records", label: "Records" },
    { id: "console", label: "Console" },
    { id: "ir", label: "IR" },
  ];
  const levelColor = { log: "#d4d4d4", info: "#9cdcfe", warn: "#dcdcaa", error: "#f48771" };

  return (
    <div className="flex flex-col overflow-hidden rounded border" style={{ height: 560, background: T.editor, borderColor: T.border }}>
      <div className="flex h-8 shrink-0 items-center gap-3 px-3" style={{ background: "#2d2d2d", borderBottom: `1px solid ${T.border}` }}>
        <span className="font-mono text-[12px] font-bold" style={{ color: "#c586c0" }}>shapeshifter</span>
        <span className="text-[11px]" style={{ color: "#999" }}>proteomics_experiment.ss — run_proteomics (HSA / HBB / ENO1)</span>
        {records.length > 0 && <span className="ml-auto font-mono text-[11px]" style={{ color: "#888" }}>{records.length} records</span>}
      </div>
      <div className="flex min-h-0 flex-1">
        <div className="flex min-w-0 flex-col" style={{ width: "44%" }}>
          <div className="flex items-center justify-between px-3 py-1.5" style={{ background: T.tabInactive }}>
            <span className="text-[10px] uppercase tracking-widest" style={{ color: T.tabFg }}>source · .ss</span>
            <button onClick={() => run(source)} disabled={running}
              className="rounded px-3 py-0.5 text-[12px] font-medium text-white disabled:opacity-40" style={{ background: running ? "#555" : T.accent }}>
              {running ? "Running…" : "▶ Run"}
            </button>
          </div>
          <Editor value={source} onChange={setSource} />
        </div>

        <div className="flex min-w-0 flex-1 flex-col" style={{ borderLeft: `1px solid ${T.border}` }}>
          <div className="flex h-9 shrink-0 items-center" style={{ background: T.tabInactive }}>
            {tabs.map(({ id, label }) => {
              const active = tab === id;
              const badge = id === "console" ? logs.length : 0;
              return (
                <button key={id} onClick={() => setTab(id)}
                  className="relative flex items-center gap-1.5 px-3 text-[12px]"
                  style={{ color: active ? T.tabFgActive : T.tabFg, background: active ? T.tabActive : "transparent" }}>
                  {label}
                  {badge > 0 && <span className="rounded-full px-1.5 text-[10px]" style={{ background: T.accent, color: "#fff" }}>{badge}</span>}
                  {active && <span className="absolute left-0 top-0 h-0.5 w-full" style={{ background: T.accentBright }} />}
                </button>
              );
            })}
          </div>
          <div className="min-h-0 flex-1">
            {tab === "charts" && <SandboxCharts records={records} />}
            {tab === "records" && <RecordsPanel records={records} />}
            {tab === "console" && (
              <div className="flex h-full flex-col">
                <div className="max-h-1/2 overflow-y-auto p-2 font-mono text-[12px]" style={{ borderBottom: `1px solid ${T.border}` }}>
                  {logs.length === 0
                    ? <div className="px-1 pt-1" style={{ color: "#5a5a5a" }}>Execution log appears here.</div>
                    : logs.map((l, i) => (
                        <div key={i} className="border-b px-1 py-0.5" style={{ color: levelColor[l.level] || "#d4d4d4", borderColor: "#2a2a2a" }}>
                          <span className="mr-2 opacity-50">{l.level}</span>{l.message}
                        </div>
                      ))}
                </div>
                <div className="min-h-0 flex-1"><TerminalView term={term} /></div>
              </div>
            )}
            {tab === "ir" && (
              <pre className="h-full overflow-auto p-3 font-mono text-[11px] leading-[1.5]" style={{ color: T.editorFg }}>
                {ir || "No IR — Run a .ss file first"}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
