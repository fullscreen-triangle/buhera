import { useMemo, useState } from "react";
import { repoName, shortSha, vendor, type VendorEntry } from "../data";
import { useTooltip } from "./tooltip";

// Upstream repository → vendored copy → engine store, from vendor.json. Each
// middle node is one manifest entry with its recorded commit and mode; hover
// shows the entry's note (why it looks the way it does).

const W = 1120;
const ROW = 40;

const storeOf = (to: string) =>
  to.startsWith("buhera-os/vendor")
    ? "buhera-os/vendor  (Rust engines)"
    : to.startsWith("long-grass/vendor")
      ? "long-grass/vendor  (TS/JS engines)"
      : "long-grass/src/lib  (host infrastructure)";

export function SourcingGraph() {
  const [hover, setHover] = useState<string | null>(null);
  const tip = useTooltip();
  const L = useMemo(() => {
    const entries = [...vendor.entries].sort((a, b) => a.repo.localeCompare(b.repo) || a.id.localeCompare(b.id));
    const repos = [...new Set(entries.map((e) => e.repo))];
    const stores = [...new Set(entries.map((e) => storeOf(e.to)))];
    const height = 70 + entries.length * ROW + 20;
    const eY = new Map(entries.map((e, i) => [e.id, 70 + i * ROW]));
    const span = (ids: string[]) => {
      const ys = ids.map((id) => eY.get(id)!);
      return (Math.min(...ys) + Math.max(...ys)) / 2;
    };
    const rY = new Map(repos.map((r) => [r, span(entries.filter((e) => e.repo === r).map((e) => e.id))]));
    const sY = new Map(stores.map((s, i) => [s, 90 + i * ((height - 150) / Math.max(1, stores.length - 1))]));
    return { entries, repos, stores, height, eY, rY, sY };
  }, []);

  const modeColor = (m: VendorEntry["mode"]) => (m === "build" ? "var(--remote)" : m === "file" ? "var(--bridge)" : "var(--native)");
  const lit = (e: VendorEntry) => !hover || hover === e.id || hover === e.repo || hover === storeOf(e.to);

  return (
    <div className="diagram" onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${L.height}`} role="img" aria-label="Sourcing graph">
        {[
          [24, "Upstream repository"],
          [330, "vendor.json entry · mode · commit"],
          [800, "Engine store"],
        ].map(([x, t]) => (
          <text key={t as string} x={x as number} y={36} fontSize={11.5} fontWeight={600} style={{ fill: "var(--ink-3)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            {t}
          </text>
        ))}
        {L.entries.map((e) => {
          const y = L.eY.get(e.id)! + 15;
          const ry = L.rY.get(e.repo)! + 15;
          const sy = L.sY.get(storeOf(e.to))! + 18;
          return (
            <g key={`edges-${e.id}`} opacity={lit(e) ? 0.75 : 0.12}>
              <path d={`M194,${ry} C260,${ry} 270,${y} 330,${y}`} fill="none" stroke="var(--ink-3)" strokeWidth={1.4} />
              <path d={`M660,${y} C730,${y} 740,${sy} 800,${sy}`} fill="none" stroke={modeColor(e.mode)} strokeWidth={1.6} />
            </g>
          );
        })}
        {L.repos.map((r) => (
          <g key={r} onMouseEnter={() => setHover(r)} style={{ cursor: "default" }}>
            <rect x={24} y={L.rY.get(r)!} width={170} height={30} rx={8} fill="var(--panel)" stroke="var(--rule)" />
            <text x={36} y={L.rY.get(r)! + 20} fontSize={12.5} fontWeight={600}>
              {repoName(r)}
            </text>
          </g>
        ))}
        {L.entries.map((e) => {
          const y = L.eY.get(e.id)!;
          return (
            <g
              key={e.id}
              opacity={lit(e) ? 1 : 0.25}
              onMouseEnter={(ev) => {
                setHover(e.id);
                tip.show(ev, `${e.repo}:${e.from} → ${e.to}${e.note ? " — " + e.note : ""}`);
              }}
              onMouseMove={tip.move}
              onMouseLeave={tip.hide}
            >
              <rect x={330} y={y} width={330} height={30} rx={8} fill="var(--panel)" stroke={hover === e.id ? "var(--accent)" : "var(--rule)"} />
              <text x={342} y={y + 19} fontSize={12} fontWeight={600}>
                {e.id}
              </text>
              <text x={505} y={y + 19} fontSize={11} style={{ fill: modeColor(e.mode), fontFamily: "var(--mono)" }}>
                {e.mode}
                {e.local?.length ? "+local" : ""}
              </text>
              <text x={590} y={y + 19} fontSize={11} style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)" }}>
                @{shortSha(e.commit)}
              </text>
            </g>
          );
        })}
        {L.stores.map((s) => (
          <g key={s} onMouseEnter={() => setHover(s)}>
            <rect x={800} y={L.sY.get(s)!} width={296} height={36} rx={8} fill="var(--panel)" stroke="var(--rule)" />
            <text x={814} y={L.sY.get(s)! + 23} fontSize={12.5} fontWeight={600}>
              {s}
            </text>
          </g>
        ))}
      </svg>
      {tip.node}
    </div>
  );
}
