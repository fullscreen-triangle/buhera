import { useEffect, useState } from "react";

// One act, step by step, through Registry::dispatch — rules R1–R4 of
// specification 03, the same in both libraries.

interface Step {
  title: string;
  rule: string;
  text: string;
  active: string[];
}

const STEPS: Step[] = [
  { title: "Instruction", rule: "I1–I4", text: "A caller — terminal, tutorial cell, generated script, remote host — calls dispatch(moduleId, instruction, actBudget). The instruction is JSON: a string (source or verb) or { kind, … }.", active: ["caller", "e1"] },
  { title: "Lookup", rule: "R1", text: "The registry resolves moduleId. Unknown id → the caller gets an UnknownModule error (TS throws, Rust returns Err). Nothing is audited: this is a caller error, not a module failure.", active: ["registry", "e1", "unknown"] },
  { title: "Execute", rule: "M1–M6", text: "module.execute(instruction, actBudget) — a thin adapter over the vendored engine. Invalid input returns the standard invalid-instruction result; it never throws.", active: ["registry", "module", "e2", "engine", "e3"] },
  { title: "Contain", rule: "R2", text: "If execute throws (TS) or panics (Rust), the registry records { ok:false, output_delta:null, residue:0, completed:true, error }. This is the only null-delta path and marks a defect in the module.", active: ["module", "contain", "e4"] },
  { title: "Audit", rule: "R3", text: "An AuditEntry is appended: act_id (monotone from 1, never reused), module_id, instruction, act_budget, result, wall_clock_ms, RFC 3339 timestamp.", active: ["registry", "audit", "e5"] },
  { title: "Hooks", rule: "R4", text: "Each post-dispatch hook sees the entry, in registration order — the purpose-carry feeder, the desk observer. A failing hook is contained and never affects the caller.", active: ["audit", "hooks", "e6"] },
  { title: "Result", rule: "A1–A5", text: "The ActResult returns to the caller. output_delta.kind picks the renderer; residue is in the module's declared unit; ok:false means the act could not be performed, not that the answer was 'no'.", active: ["caller", "e7", "registry"] },
];

const box = (id: string, x: number, y: number, w: number, label: string, sub: string, on: boolean) => (
  <g key={id}>
    <rect x={x} y={y} width={w} height={54} rx={10} fill={on ? "var(--accent)" : "var(--panel)"} stroke={on ? "var(--accent)" : "var(--rule)"} strokeWidth={on ? 2 : 1} style={{ transition: "fill 200ms" }} />
    <text x={x + w / 2} y={y + 24} fontSize={13.5} fontWeight={600} textAnchor="middle" style={{ fill: on ? "var(--accent-ink)" : "var(--ink)" }}>
      {label}
    </text>
    <text x={x + w / 2} y={y + 41} fontSize={10.5} textAnchor="middle" style={{ fill: on ? "var(--accent-ink)" : "var(--ink-3)", fontFamily: "var(--mono)" }}>
      {sub}
    </text>
  </g>
);

export function DispatchPipeline() {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setI((k) => (k + 1) % STEPS.length), 2200);
    return () => clearInterval(t);
  }, [playing]);
  const s = STEPS[i]!;
  const on = (id: string) => s.active.includes(id);
  const edge = (id: string, d: string, dashed = false) => (
    <path key={id} d={d} fill="none" stroke={on(id) ? "var(--accent)" : "var(--rule)"} strokeWidth={on(id) ? 2.5 : 1.5} strokeDasharray={dashed ? "5 4" : undefined} markerEnd="url(#arrow)" />
  );

  return (
    <div>
      <div className="diagram">
        <svg viewBox="0 0 1000 320" role="img" aria-label="Dispatch pipeline">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
            </marker>
          </defs>
          {edge("e1", "M150,77 L250,77")}
          {edge("e2", "M430,77 L530,77")}
          {edge("e3", "M710,77 L800,77")}
          {edge("e4", "M620,104 L620,140", true)}
          {edge("e5", "M340,104 L340,245")}
          {edge("e6", "M430,272 L530,272")}
          {edge("e7", "M250,60 C200,20 170,20 150,60")}
          {box("caller", 20, 50, 130, "Caller", "dispatch(…)", on("caller"))}
          {box("registry", 250, 50, 180, "Registry", "the one funnel", on("registry"))}
          {box("module", 530, 50, 180, "Module", "execute()", on("module"))}
          {box("engine", 800, 50, 180, "Vendored engine", "upstream @ commit", on("engine"))}
          {box("contain", 530, 140, 180, "Containment", "R2: null delta", on("contain"))}
          {box("audit", 250, 245, 180, "Audit log", "act_id++", on("audit"))}
          {box("hooks", 530, 245, 180, "Hooks", "in order, isolated", on("hooks"))}
          {on("unknown") && (
            <g>
              <rect x={20} y={160} width={180} height={44} rx={8} fill="var(--panel)" stroke="var(--remote)" strokeDasharray="4 3" />
              <text x={110} y={187} fontSize={12} textAnchor="middle" style={{ fill: "var(--remote)" }}>
                unknown id → error, not audited
              </text>
            </g>
          )}
        </svg>
      </div>
      <div className="step-controls">
        <button onClick={() => setI((k) => (k - 1 + STEPS.length) % STEPS.length)}>◀ Prev</button>
        <button onClick={() => setPlaying((p) => !p)}>{playing ? "Pause" : "Play"}</button>
        <button onClick={() => setI((k) => (k + 1) % STEPS.length)}>Next ▶</button>
        <span>
          <b>
            {i + 1}/{STEPS.length} · {s.title}
          </b>{" "}
          <code>{s.rule}</code>
        </span>
      </div>
      <p style={{ color: "var(--ink-2)", fontSize: 14, marginBottom: 0 }}>{s.text}</p>
    </div>
  );
}
