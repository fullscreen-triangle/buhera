/* ============================================================================
 * CkgSandboxes — the CKG experiment apparatus.
 *
 * Four REAL sandbox panels, each running the source app's real interpreter in
 * the browser on that app's real example script, and rendering the interpreter's
 * own output with the source app's own renderer components. This is the live
 * sandbox of each source site, reproduced in-page — not a text echo of it.
 *
 *   SBS          — the sbs-sandbox renderer: 3D force graph of the KEGG hsa00190
 *                  oxidative-phosphorylation circuit (with the rotenone Complex-I
 *                  perturbation), ~15 D3 charts, console, and the compiled
 *                  glsl / ast / circuit / js.
 *   honjo        — the honjo playground: the causal table (a Path @ floor,
 *                  converged, with an amalgamation) + per-entity detail cards.
 *   shapeshifter — the shapeshifter sandbox: charts + a records view over the
 *                  live PredictedRecord[], console/terminal, and the IR.
 *   shakespeare  — the shakespeare (.shk) sandbox: the femtosecond
 *                  electron-transfer trace + the ET-chain GLB structure + console.
 *
 * Each panel is dynamic(ssr:false): all four touch WebGL / D3 / the DOM on mount.
 * ========================================================================== */

import { useState } from "react";
import dynamic from "next/dynamic";

const loading = (label) => (
  <div className="flex items-center justify-center rounded border border-neutral-800 text-sm text-neutral-500" style={{ height: 560 }}>
    loading {label} sandbox…
  </div>
);

const SbsSandboxPanel = dynamic(() => import("@/components/sandboxes/sbs/SbsSandboxPanel"), {
  ssr: false, loading: () => loading("SBS"),
});
const HonjoSandboxPanel = dynamic(() => import("@/components/sandboxes/honjo/HonjoSandboxPanel"), {
  ssr: false, loading: () => loading("honjo"),
});
const ShapeshifterSandboxPanel = dynamic(() => import("@/components/sandboxes/shapeshifter/ShapeshifterSandboxPanel"), {
  ssr: false, loading: () => loading("shapeshifter"),
});
const ShakespeareSandboxPanel = dynamic(() => import("@/components/sandboxes/shakespeare/ShakespeareSandboxPanel"), {
  ssr: false, loading: () => loading("shakespeare"),
});

const PANELS = [
  {
    id: "sbs", label: "SBS", accent: "#4ec9b0",
    caption: "the sbs-sandbox renderer, run in-page — KEGG hsa00190 electron transport, rotenone Complex-I perturbation (NADH→CoQ ×0.3)",
    Panel: SbsSandboxPanel,
  },
  {
    id: "honjo", label: "honjo", accent: "#58E6D9",
    caption: "the honjo playground, run in-page — track.hj, the causal table",
    Panel: HonjoSandboxPanel,
  },
  {
    id: "shapeshifter", label: "shapeshifter", accent: "#c586c0",
    caption: "the shapeshifter sandbox, run in-page — proteomics_experiment.ss over the live PredictedRecord[]",
    Panel: ShapeshifterSandboxPanel,
  },
  {
    id: "shakespeare", label: "shakespeare", accent: "#B63E96",
    caption: "the shakespeare (.shk) sandbox, run in-page — 07_electron-chain, the femtosecond electron-transfer trace",
    Panel: ShakespeareSandboxPanel,
  },
];

export default function CkgSandboxes() {
  const [active, setActive] = useState("sbs");
  const current = PANELS.find((p) => p.id === active) ?? PANELS[0];
  const { Panel } = current;

  return (
    <div>
      {/* Tab strip */}
      <div className="flex flex-wrap gap-2 border-b border-neutral-800 pb-2">
        {PANELS.map((p) => {
          const on = p.id === active;
          return (
            <button
              key={p.id}
              onClick={() => setActive(p.id)}
              className="rounded px-3 py-1 font-mono text-[13px] transition-colors"
              style={{
                color: on ? "#0a0a0a" : p.accent,
                background: on ? p.accent : "transparent",
                border: `1px solid ${p.accent}${on ? "" : "55"}`,
              }}
            >
              {p.label}
            </button>
          );
        })}
      </div>

      <p className="mt-2 mb-3 text-xs text-gray-500">{current.caption}</p>

      {/* Keep every mounted panel alive but only show the active one, so a
          panel's interpreter run and WebGL context are not torn down and
          rebuilt each time the reader switches tabs. */}
      {PANELS.map((p) => (
        <div key={p.id} style={{ display: p.id === active ? "block" : "none" }}>
          <p.Panel />
        </div>
      ))}
    </div>
  );
}
