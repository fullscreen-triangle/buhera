/* CapabilityVenn — interactive Feat ⊇ Capset(S) ⊇ Req diagram.
 *
 * Drag the Req circle's radius (via the slider) to see satisfiability of
 * Theorem "static decidability of containment" flip live: Req ⊆ Capset(S)
 * or not. Pure SVG, no d3 scales needed — this is a static-geometry diagram
 * driven by React state, kept in the datamodel/ family for consistency with
 * the other d3-based charts on the same page.
 */
import React, { useState } from "react";

const FEAT_R = 150;
const CAPSET_R = 100;
const CENTER = 170;

export default function CapabilityVenn() {
  const [reqSize, setReqSize] = useState(55); // radius of Req circle
  const [reqOffset, setReqOffset] = useState(0); // horizontal offset from center

  // A crude but sufficient "is Req inside Capset" test: distance between
  // centers + reqSize must not exceed CAPSET_R (Req fully contained).
  const contained = reqOffset + reqSize <= CAPSET_R;

  return (
    <div className="flex flex-col items-center">
      <svg width={340} height={340} viewBox="0 0 340 340">
        <circle cx={CENTER} cy={CENTER} r={FEAT_R} fill="#0d1520" stroke="#3a4a5c" strokeWidth={1.5} />
        <text x={CENTER} y={CENTER - FEAT_R + 18} textAnchor="middle" fill="#6b7f94" fontSize="11" fontFamily="monospace">Feat</text>

        <circle cx={CENTER} cy={CENTER} r={CAPSET_R} fill="#0f2a1f" stroke="#2e7d5b" strokeWidth={1.5} />
        <text x={CENTER} y={CENTER - CAPSET_R + 16} textAnchor="middle" fill="#4fae86" fontSize="11" fontFamily="monospace">Capset(S)</text>

        <circle
          cx={CENTER + reqOffset}
          cy={CENTER}
          r={reqSize}
          fill={contained ? "#1a3a5c" : "#5c1a1a"}
          stroke={contained ? "#4a9eff" : "#ff5555"}
          strokeWidth={2}
          opacity={0.85}
        />
        <text
          x={CENTER + reqOffset}
          y={CENTER + 4}
          textAnchor="middle"
          fill={contained ? "#8fc4ff" : "#ff9d9d"}
          fontSize="12"
          fontFamily="monospace"
          fontWeight="bold"
        >
          Req
        </text>
      </svg>

      <div className="w-full max-w-xs space-y-3 mt-2">
        <label className="block text-xs text-gray-400">
          |Req| (requested feature set size) — radius {reqSize}
          <input
            type="range"
            min={20}
            max={140}
            value={reqSize}
            onChange={(e) => setReqSize(Number(e.target.value))}
            className="w-full accent-emerald-500"
          />
        </label>
        <label className="block text-xs text-gray-400">
          Divergence from Capset(S) (a feature Req needs that S doesn&apos;t declare) — offset {reqOffset}
          <input
            type="range"
            min={0}
            max={130}
            value={reqOffset}
            onChange={(e) => setReqOffset(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
        </label>
      </div>

      <div className={`mt-3 text-sm font-mono px-3 py-1.5 rounded ${contained ? "bg-blue-950 text-blue-300 border border-blue-800" : "bg-red-950 text-red-300 border border-red-800"}`}>
        {contained ? "Req ⊆ Capset(S) — request satisfiable, decided in O(|Feat|)" : "Req ⊄ Capset(S) — refused before any record is read"}
      </div>
    </div>
  );
}
