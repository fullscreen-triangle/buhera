import { useMemo, useState } from "react";
import { catalogue, LAYER_ORDER, repoName, type Binding, type ModuleRow } from "../data";
import { useTooltip } from "./tooltip";

// Upstream repositories → modules (grouped by layer) → the two hosts. Every
// edge is drawn from catalogue.json: engine provenance on the left, binding
// kind per host on the right. Hover isolates one module's paths; click opens
// its specification.

const W = 1120;
const ROW = 46;
const LAYER_GAP = 30;
const X_REPO = 24;
const REPO_W = 170;
const X_MOD = 330;
const MOD_W = 290;
const X_RUST = 760;
const X_TS = 940;
const HOST_W = 150;

const bindingColor = (b: Binding) => `var(--${b})`;
const langColor = (l: string) => (l === "rust" ? "var(--rust)" : "var(--ts)");

export function FederationMap({ onOpen }: { onOpen: (id: string) => void }) {
  const [hover, setHover] = useState<string | null>(null);
  const tip = useTooltip();

  const layout = useMemo(() => {
    const rows: Array<{ kind: "layer"; layer: string; y: number } | { kind: "mod"; m: ModuleRow; y: number }> = [];
    let y = 70;
    for (const layer of LAYER_ORDER) {
      const mods = catalogue.modules.filter((m) => m.layer === layer);
      if (!mods.length) continue;
      rows.push({ kind: "layer", layer, y });
      y += 26;
      for (const m of mods) {
        rows.push({ kind: "mod", m, y });
        y += ROW;
      }
      y += LAYER_GAP - 26 + 8;
    }
    const height = y + 10;
    const modY = new Map(rows.filter((r) => r.kind === "mod").map((r) => [(r as { m: ModuleRow }).m.id, r.y]));

    const repos = [...new Set(catalogue.modules.flatMap((m) => (m.upstream.length ? m.upstream.map((u) => u.repo) : ["(this repo)"])))];
    const repoY = new Map<string, number>();
    const step = (height - 110) / Math.max(1, repos.length);
    repos.forEach((r, i) => repoY.set(r, 90 + i * step + step / 2 - 18));
    return { rows, height, modY, repos, repoY };
  }, []);

  const dim = (id: string) => (hover && hover !== id ? 0.16 : 1);

  return (
    <div className="diagram" onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${layout.height}`} role="img" aria-label="Federation map: upstream repositories, modules, and host bindings">
        {/* column headers */}
        {[
          [X_REPO, "Upstream repository"],
          [X_MOD, "Module (by layer)"],
          [X_RUST, "Rust host"],
          [X_TS, "TypeScript host"],
        ].map(([x, t]) => (
          <text key={t as string} x={x as number} y={36} fontSize={11.5} fontWeight={600} style={{ fill: "var(--ink-3)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            {t}
          </text>
        ))}
        <line x1={X_REPO} x2={W - 24} y1={48} y2={48} stroke="var(--rule)" />

        {/* repo → module edges */}
        {catalogue.modules.flatMap((m) =>
          (m.upstream.length ? m.upstream : [{ repo: "(this repo)", language: "rust", path: "", commit: "" }]).map((u, i) => {
            const y1 = (layout.repoY.get(u.repo) ?? 0) + 18;
            const y2 = (layout.modY.get(m.id) ?? 0) + 18;
            const x1 = X_REPO + REPO_W;
            const x2 = X_MOD;
            return (
              <path key={`${m.id}-${i}`} d={`M${x1},${y1} C${x1 + 70},${y1} ${x2 - 70},${y2} ${x2},${y2}`} fill="none" stroke={langColor(u.language)} strokeWidth={1.6} opacity={0.55 * dim(m.id)} />
            );
          }),
        )}

        {/* module → host edges */}
        {catalogue.modules.flatMap((m) =>
          (["rust", "ts"] as const).map((h) => {
            const b = m.bindings[h];
            const y = (layout.modY.get(m.id) ?? 0) + 18;
            const x2 = h === "rust" ? X_RUST : X_TS;
            if (b === "none") return null;
            return (
              <line key={`${m.id}-${h}`} x1={X_MOD + MOD_W} x2={x2} y1={y} y2={y} stroke={bindingColor(b)} strokeWidth={2} strokeDasharray={b === "remote" ? "6 4" : b === "bridge" ? "2 3" : undefined} opacity={0.8 * dim(m.id)} />
            );
          }),
        )}

        {/* repos */}
        {layout.repos.map((r) => {
          const y = layout.repoY.get(r)!;
          const active = hover ? catalogue.modules.some((m) => m.id === hover && (m.upstream.some((u) => u.repo === r) || (!m.upstream.length && r === "(this repo)"))) : true;
          return (
            <g key={r} opacity={active ? 1 : 0.25}>
              <rect x={X_REPO} y={y} width={REPO_W} height={36} rx={8} fill="var(--panel)" stroke="var(--rule)" />
              <text x={X_REPO + 12} y={y + 23} fontSize={13} fontWeight={600}>
                {repoName(r)}
              </text>
            </g>
          );
        })}

        {/* layers + modules */}
        {layout.rows.map((row) =>
          row.kind === "layer" ? (
            <text key={row.layer} x={X_MOD} y={row.y + 14} fontSize={11} fontWeight={600} style={{ fill: "var(--accent)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
              {row.layer}
            </text>
          ) : (
            <g
              key={row.m.id}
              style={{ cursor: "pointer" }}
              opacity={dim(row.m.id)}
              onMouseEnter={(e) => {
                setHover(row.m.id);
                tip.show(e, `${row.m.name} — ${row.m.summary}`);
              }}
              onMouseMove={tip.move}
              onMouseLeave={tip.hide}
              onClick={() => onOpen(row.m.id)}
            >
              <rect x={X_MOD} y={row.y} width={MOD_W} height={36} rx={8} fill="var(--panel)" stroke={hover === row.m.id ? "var(--accent)" : "var(--rule)"} strokeWidth={hover === row.m.id ? 2 : 1} />
              <text x={X_MOD + 12} y={row.y + 16} fontSize={13} fontWeight={600}>
                {row.m.id}
              </text>
              <text x={X_MOD + 12} y={row.y + 30} fontSize={10.5} style={{ fill: "var(--ink-3)" }}>
                {row.m.dsl ? `language: ${row.m.dsl}` : "no language"}
              </text>
              {(["rust", "ts"] as const).map((h) => {
                const b = row.m.bindings[h];
                const x = h === "rust" ? X_RUST : X_TS;
                return (
                  <g key={h}>
                    <rect x={x} y={row.y + 5} width={HOST_W} height={26} rx={13} fill="var(--panel)" stroke={bindingColor(b)} strokeWidth={1.5} strokeDasharray={b === "none" ? "3 3" : undefined} />
                    <circle cx={x + 14} cy={row.y + 18} r={4} fill={bindingColor(b)} />
                    <text x={x + 26} y={row.y + 22} fontSize={11.5} style={{ fill: bindingColor(b), fontFamily: "var(--mono)" }}>
                      {b}
                    </text>
                  </g>
                );
              })}
            </g>
          ),
        )}
      </svg>
      {tip.node}
    </div>
  );
}
