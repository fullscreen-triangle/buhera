/* ============================================================================
 * ValueCard — the honjo per-entity detail card.
 *
 * Reproduced from borgia/honjo-masamune/src/pages/playground.js (private app —
 * reproduced as grounding, not copied wholesale). Renders one entry of the
 * interpreter's `result.named` map: an Atom (Z, config, valence…), a Bond, a
 * Compound (formula, geometry, angle…), a Path, or a Scalar — plus the residue
 * and floor every honjo value carries.
 * ========================================================================== */

import { useMemo } from "react";

const ACCENT = "#58E6D9";

const TYPE_COLOR = {
  Atom: "#3b82f6",
  Bond: "#22c55e",
  Compound: "#f97316",
  Path: "#a855f7",
  Scalar: "#6b7280",
};

export function fmt(n) {
  if (typeof n !== "number" || Number.isNaN(n)) return "—";
  if (Number.isInteger(n)) return String(n);
  return n.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

export default function ValueCard({ name, v }) {
  const color = TYPE_COLOR[v.ty] || "#6b7280";
  const rows = useMemo(() => {
    switch (v.ty) {
      case "Atom":
        return [
          ["Z", v.Z], ["symbol", v.symbol], ["config", v.config],
          ["term", v.term], ["vacancy", v.vacancy], ["valence", v.valence],
        ];
      case "Bond":
        return [
          ["pair", `${v.a}~${v.b}`], ["exists", String(v.exists)],
          ["Δthickness", fmt(v.delta)], ["shared", v.shared],
        ];
      case "Compound": {
        const lig = v.formula[1] > 1 ? v.ligand + v.formula[1] : v.formula[1] === 1 ? v.ligand : "";
        const formula = v.formula[0] === 2 ? v.central + "₂" : v.central + lig;
        return [
          ["formula", formula], ["geometry", v.geometry],
          ["angle", v.angleDeg == null ? "—" : fmt(v.angleDeg) + "°"],
          ["closed", String(v.valenceClosed)],
        ];
      }
      case "Path":
        return [
          ["item", v.item], ["steps", v.steps], ["converged", String(v.converged)],
          ["reps", v.reps.join(", ")], ["amalgamation", v.amalgamation.join(", ") || "—"],
        ];
      default:
        return [["value", fmt(v.value)]];
    }
  }, [v]);

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/60 p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono text-sm text-white">{name}</span>
        <span
          className="rounded-full px-2 py-0.5 text-[10px] uppercase tracking-widest"
          style={{ color, border: `1px solid ${color}55` }}
        >
          {v.ty}
        </span>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
        {rows.map(([k, val]) => (
          <div key={k} className="contents">
            <dt className="text-neutral-500">{k}</dt>
            <dd className="break-all font-mono text-neutral-200">{String(val)}</dd>
          </div>
        ))}
        <dt className="text-neutral-500">residue</dt>
        <dd className="font-mono" style={{ color: ACCENT }}>{fmt(v.residue)} @ floor {fmt(v.floor)}</dd>
      </dl>
    </div>
  );
}
