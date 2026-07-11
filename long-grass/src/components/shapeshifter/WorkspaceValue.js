/**
 * WorkspaceValue — inline renderer for one Shape Shifter workspace entry.
 *
 * Ported from the lavoisier-web Sandbox (src/sandbox/ShapeshifterSandbox.js).
 * In the Sandbox these panels tile a whole-program dashboard; in Buhera's
 * REPL/notebook model each dispatch produces ONE workspace value, so this
 * component renders a single entry inline in the terminal transcript. There is
 * no store, no crossfilter, no dashboard — just the per-kind panel.
 *
 * The panel theme colours are inlined (the Sandbox `T` object is not carried
 * over) and the async spinner is a text glyph (no lucide-react dependency).
 */

import React from "react";
import SandboxCharts from "@/components/shapeshifter/SandboxCharts";

const BORDER = "#3c3c3c";
const FG = "#d4d4d4";

/* ── records → the record chart grid ─────────────────────────────────────── */
function RecordsPanel({ records }) {
  return <SandboxCharts records={records} />;
}

/* ── ΔP cell registry ────────────────────────────────────────────────────── */
function CellsPanel({ cells }) {
  return (
    <div>
      <div className="mb-3 text-[10px] uppercase tracking-wider" style={{ color: "#666" }}>
        ΔP Cell Registry — {cells.length} cells
      </div>
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#666", borderBottom: `1px solid ${BORDER}` }}>
              {["Target", "m/z", "±Da", "ω (Hz)", "ΔP lo", "ΔP hi", "τ_min ms"].map(h => (
                <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {cells.map((c, i) => (
              <tr key={i} style={{ borderBottom: `1px solid #2a2a2a`, color: FG }}>
                <td className="py-0.5 pr-3" style={{ color: "#9cdcfe" }}>{c.name}</td>
                <td className="pr-3">{c.mz}</td>
                <td className="pr-3" style={{ color: "#dcdcaa" }}>±{c.window_da}</td>
                <td className="pr-3" style={{ color: "#b388eb" }}>{c.omega_hz}</td>
                <td className="pr-3" style={{ color: "#e07a7a" }}>{c.dp_lo}</td>
                <td className="pr-3" style={{ color: "#7cc77c" }}>{c.dp_hi}</td>
                <td className="pr-3" style={{ color: "#e6a456" }}>{c.tau_min_ms}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── partition addresses ─────────────────────────────────────────────────── */
function AddressesPanel({ addresses }) {
  return (
    <div>
      <div className="mb-3 text-[10px] uppercase tracking-wider" style={{ color: "#666" }}>
        Partition Addresses — Φ: ℤ⁺ → 𝒫
      </div>
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#666", borderBottom: `1px solid ${BORDER}` }}>
              {["Name", "Mass (Da)", "Adduct", "n", "ℓ", "m", "s"].map(h => (
                <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {addresses.map((a, i) => (
              <tr key={i} style={{ borderBottom: `1px solid #2a2a2a`, color: FG }}>
                <td className="py-0.5 pr-3" style={{ color: "#9cdcfe" }}>{a.name}</td>
                <td className="pr-3">{a.mass}</td>
                <td className="pr-3" style={{ color: "#dcdcaa" }}>{a.adduct}</td>
                <td className="pr-3" style={{ color: "#5fa8d3" }}>{a.n}</td>
                <td className="pr-3" style={{ color: "#7cc77c" }}>{a.l}</td>
                <td className="pr-3" style={{ color: "#e07a7a" }}>{a.m}</td>
                <td className="pr-3" style={{ color: "#b388eb" }}>{a.s > 0 ? "+½" : "−½"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── generic value visualisations ────────────────────────────────────────── */

function Card({ label, value, color = FG }) {
  return (
    <div className="rounded p-2" style={{ background: "#2a2d2e", border: `1px solid ${BORDER}` }}>
      <div className="text-[9px] uppercase tracking-wider mb-0.5" style={{ color: "#666" }}>{label}</div>
      <div className="font-mono text-[12px] break-words" style={{ color }}>{value}</div>
    </div>
  );
}

/** Horizontal mini-bar chart from {label, value} rows. */
function MiniBars({ rows, max }) {
  const m = max ?? Math.max(...rows.map(r => Math.abs(r.value)), 1e-9);
  return (
    <div className="space-y-1">
      {rows.map((r, i) => (
        <div key={i} className="flex items-center gap-2 text-[10px]">
          <span className="w-24 shrink-0 truncate font-mono" style={{ color: "#cccccc" }}>{r.label}</span>
          <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "#2a2d2e" }}>
            <div className="h-full rounded-full"
              style={{ width: `${Math.min(100, (Math.abs(r.value) / m) * 100)}%`, background: r.color || "#5fa8d3" }} />
          </div>
          <span className="w-16 text-right font-mono" style={{ color: "#888" }}>
            {typeof r.value === "number" ? r.value.toFixed(3) : r.value}
          </span>
        </div>
      ))}
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[9px] uppercase tracking-[0.15em]" style={{ color: "#666" }}>{title}</div>
      {children}
    </div>
  );
}

/** S-entropy coordinate as cards + mini-bars. */
function SentropyView({ value }) {
  const { sk = 0, st = 0, se = 0 } = value;
  return (
    <Section title="S-entropy coordinates (Sₖ, Sₜ, Sₑ)">
      <div className="grid grid-cols-3 gap-2">
        <Card label="Sₖ knowledge" value={sk.toFixed(4)} color="#22d3ee" />
        <Card label="Sₜ temporal"  value={st.toFixed(4)} color="#fbbf24" />
        <Card label="Sₑ evolution" value={se.toFixed(4)} color="#a78bfa" />
      </div>
      <MiniBars max={1} rows={[
        { label: "Sₖ", value: sk, color: "#22d3ee" },
        { label: "Sₜ", value: st, color: "#fbbf24" },
        { label: "Sₑ", value: se, color: "#a78bfa" },
      ]} />
    </Section>
  );
}

/** Dual-path validation: convergence + false-positive probability. */
function ValidationView({ value }) {
  const cp = value.commonPrefixLen ?? 0;
  const conv = value.convergenceScore ?? 0;
  return (
    <Section title="Ion-droplet dual-path validation">
      <div className="grid grid-cols-3 gap-2">
        <Card label="Common prefix" value={`${cp} trits`} color="#9cdcfe" />
        <Card label="Convergence" value={`${(conv * 100).toFixed(1)}%`}
          color={conv > 0.7 ? "#34d399" : conv > 0.4 ? "#dcdcaa" : "#f48771"} />
        <Card label="False-pos ≤" value={(value.falsePosProb ?? 1).toExponential(2)} color="#fb923c" />
      </div>
      {value.ionAddress && (
        <div className="font-mono text-[9px]" style={{ color: "#666" }}>
          ion:&nbsp;&nbsp;&nbsp;<span style={{ color: "#5fa8d3" }}>{value.ionAddress}</span><br />
          drip:&nbsp;&nbsp;<span style={{ color: "#e07a7a" }}>{value.dropletAddress}</span>
        </div>
      )}
    </Section>
  );
}

/** Purpose domain: reduction ratio + prefix count. */
function DomainView({ value }) {
  const rho = value.reductionPct ?? (value.reductionRatio != null ? value.reductionRatio * 100 : null);
  const prefixes = value.prefixes?.length ?? 0;
  return (
    <Section title={`Purpose domain — ${value.label || value.name || "context"}`}>
      <div className="grid grid-cols-2 gap-2">
        <Card label="Prefixes" value={prefixes} color="#9cdcfe" />
        {rho != null && <Card label="Reduction" value={`${rho.toFixed(1)}%`} color="#34d399" />}
      </div>
      {value.bounds && (
        <MiniBars max={1} rows={[
          { label: "Sₖ range", value: value.bounds.sk[1] - value.bounds.sk[0], color: "#22d3ee" },
          { label: "Sₜ range", value: value.bounds.st[1] - value.bounds.st[0], color: "#fbbf24" },
          { label: "Sₑ range", value: value.bounds.se[1] - value.bounds.se[0], color: "#a78bfa" },
        ]} />
      )}
    </Section>
  );
}

/** Subharmonic frequency ratios as a bar chart. */
function SubharmonicsView({ value }) {
  const rows = value.slice(0, 16).map(s => ({
    label: `${s.fragmentMz?.toFixed(1)}`,
    value: s.frequencyRatio ?? 0,
    color: s.selfConsistent ? "#34d399" : "#f48771",
  }));
  return (
    <Section title={`Fragment subharmonics — ω_f / ω_prec (${value.length})`}>
      <MiniBars rows={rows} />
      <div className="text-[9px]" style={{ color: "#666" }}>
        Self-consistent: {value.filter(s => s.selfConsistent).length}/{value.length} (&lt;10⁻⁶ ppm)
      </div>
    </Section>
  );
}

/** Virtual tensor report: off-shell fraction + mean recovery. */
function TensorReportView({ value }) {
  const v = value.verified || {};
  return (
    <Section title="Virtual partition tensor V_{ijkl}">
      <div className="grid grid-cols-3 gap-2">
        <Card label="Components" value={(value.tensor?.length ?? 0).toLocaleString()} color="#9cdcfe" />
        <Card label="Off-shell" value={`${((v.offShellFraction ?? 0) * 100).toFixed(1)}%`} color="#fb923c" />
        <Card label="Planck depth" value={value.planckDepth ?? "—"} color="#a78bfa" />
      </div>
      <MiniBars max={1} rows={[
        { label: "mean (recov.)", value: v.mean ?? 0, color: "#34d399" },
        { label: "v_phys", value: v.vPhys ?? 0, color: "#5fa8d3" },
      ]} />
      <div className="text-[9px]" style={{ color: v.meanRecoveryHolds ? "#6a9955" : "#f48771" }}>
        mean-recovery {v.meanRecoveryHolds ? "✓ holds" : "✗ violated"} · d_eff = {(value.dEff ?? 0).toLocaleString()}
      </div>
    </Section>
  );
}

function ImpossibleView({ value }) {
  return (
    <Section title={`Impossible ions — crossing-symmetry probes (${value.length})`}>
      <div className="overflow-x-auto">
        <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
          <thead><tr style={{ color: "#666", borderBottom: `1px solid ${BORDER}` }}>
            {["ion 1", "ion 2", "impossible m/z"].map(h => <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>)}
          </tr></thead>
          <tbody>
            {value.slice(0, 12).map((p, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #2a2a2a", color: FG }}>
                <td className="pr-3">{p.ion1_mz?.toFixed(3)}</td>
                <td className="pr-3">{p.ion2_mz?.toFixed(3)}</td>
                <td className="pr-3" style={{ color: "#fb923c" }}>{p.impossibleMz?.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function TransientView({ value }) {
  return (
    <Section title="Single-transient contents (Theorem 11.1)">
      <div className="grid grid-cols-2 gap-2">
        <Card label="Precursor freq" value={`${value.precursor?.freq_Hz?.toExponential(2)} Hz`} color="#5fa8d3" />
        <Card label="Fragment subharmonics" value={value.fragments?.length ?? 0} color="#7cc77c" />
        <Card label="Charge states" value={value.chargeStates?.length ?? 0} color="#a78bfa" />
        <Card label="Polarity Δφ" value="π" color="#e07a7a" />
      </div>
    </Section>
  );
}

function ComplementView({ value }) {
  return (
    <Section title="Partition complement (SWIFT antistate)">
      <div className="grid grid-cols-2 gap-2">
        <Card label="Original m/z" value={value.originalMz?.toFixed(3)} color="#5fa8d3" />
        <Card label="Complement m/z" value={value.complementMz?.toFixed(3)} color="#e07a7a" />
        <Card label="M_ion" value={value.M_ion} />
        <Card label="C_max" value={value.Cmax} />
      </div>
    </Section>
  );
}

function ScalarView({ name, value }) {
  return (
    <Section title={name}>
      <Card label="value" value={typeof value === "string" ? value : JSON.stringify(value)} />
    </Section>
  );
}

/** Online DB search: unresolved in Buhera v1 — shows a pending placeholder. */
function DbSearchView({ value }) {
  if (value.__async && !value.resolved) {
    return (
      <Section title={`Database search — ${value.__fn}`}>
        <div className="flex items-center gap-2 text-[11px]" style={{ color: "#dcdcaa" }}>
          <span className="inline-block animate-pulse">◌</span>
          pending: {(value.dbs || ["massbank"]).join(", ")} for m/z {value.precMz?.toFixed(4)}
          <span style={{ color: "#666" }}>(online search not wired in this build)</span>
        </div>
      </Section>
    );
  }
  const hits = value.hits || [];
  return (
    <Section title={`Database search — ${hits.length} hit(s)`}>
      {hits.length === 0
        ? <div className="text-[10px]" style={{ color: "#666" }}>No hits.</div>
        : (
          <div className="overflow-x-auto">
            <table className="w-full font-mono text-[10px]" style={{ borderCollapse: "collapse" }}>
              <thead><tr style={{ color: "#666", borderBottom: `1px solid ${BORDER}` }}>
                {["Compound", "m/z", "Formula", "Score", "DB"].map(h =>
                  <th key={h} className="py-1 pr-3 text-left font-normal">{h}</th>)}
              </tr></thead>
              <tbody>
                {hits.slice(0, 20).map((h, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid #2a2a2a", color: FG }}>
                    <td className="py-0.5 pr-3 truncate" style={{ maxWidth: 160, color: "#9cdcfe" }}>{h.name}</td>
                    <td className="pr-3">{Number(h.precursorMz)?.toFixed?.(3) ?? h.precursorMz}</td>
                    <td className="pr-3" style={{ color: "#dcdcaa" }}>{h.formula}</td>
                    <td className="pr-3" style={{ color: "#34d399" }}>{Number(h.score)?.toFixed?.(2) ?? h.score}</td>
                    <td className="pr-3" style={{ color: "#666" }}>{h.database}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
    </Section>
  );
}

/**
 * Render a single workspace entry by its classified kind. Mirrors the Sandbox's
 * WorkspaceValue switch; falls back to a JSON dump for unrecognised kinds.
 */
export default function WorkspaceValue({ entry }) {
  const { name, kind, value } = entry;
  switch (kind) {
    case "records":      return <RecordsPanel records={value} />;
    case "cells":        return <CellsPanel cells={value} />;
    case "addresses":    return <AddressesPanel addresses={value} />;
    case "sentropy":     return <SentropyView value={value} />;
    case "validation":   return <ValidationView value={value} />;
    case "domain":       return <DomainView value={value} />;
    case "subharmonics": return <SubharmonicsView value={value} />;
    case "tensorReport": return <TensorReportView value={value} />;
    case "impossible":   return <ImpossibleView value={value} />;
    case "transient":    return <TransientView value={value} />;
    case "complement":   return <ComplementView value={value} />;
    case "pending":      return <DbSearchView value={value} />;
    case "string":
    case "number":
    case "scalar":       return <ScalarView name={name} value={value} />;
    default:
      return (
        <Section title={`${name} (${kind})`}>
          <pre className="overflow-auto rounded p-2 font-mono text-[10px]"
            style={{ background: "#1a1c1e", color: FG, maxHeight: 220 }}>
            {JSON.stringify(value, null, 2)}
          </pre>
        </Section>
      );
  }
}
