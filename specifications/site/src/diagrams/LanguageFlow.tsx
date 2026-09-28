import { catalogue } from "../data";

// The DSL registry as a picture: each language → the hosts whose registry
// carries its REAL validator → the module that executes it → the knowledge
// pack that grounds generation (specification 04).

const W = 1120;
const ROW = 44;

export function LanguageFlow({ onOpen }: { onOpen: (id: string) => void }) {
  const H = 60 + catalogue.dsls.length * ROW + 10;
  return (
    <div className="diagram">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Language flow: language, validators, executing module, knowledge pack">
        {[
          [24, "Language"],
          [290, "Real validator on"],
          [560, "Executed by"],
          [850, "Grounding pack"],
        ].map(([x, t]) => (
          <text key={t as string} x={x as number} y={32} fontSize={11.5} fontWeight={600} style={{ fill: "var(--ink-3)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            {t}
          </text>
        ))}
        {catalogue.dsls.map((d, i) => {
          const y = 50 + i * ROW;
          const mid = y + 16;
          return (
            <g key={d.id} style={{ cursor: "pointer" }} onClick={() => onOpen(d.module_id)}>
              <path d={`M214,${mid} L290,${mid}`} stroke="var(--rule)" strokeWidth={1.6} />
              <path d={`M480,${mid} L560,${mid}`} stroke="var(--rule)" strokeWidth={1.6} />
              <path d={`M770,${mid} L850,${mid}`} stroke="var(--rule)" strokeWidth={1.6} strokeDasharray="4 4" />
              <rect x={24} y={y} width={190} height={32} rx={8} fill="var(--panel)" stroke="var(--rule)" />
              <text x={36} y={y + 21} fontSize={13} fontWeight={600}>
                {d.label}
              </text>
              <text x={200} y={y + 21} fontSize={11} textAnchor="end" style={{ fill: "var(--ink-3)", fontFamily: "var(--mono)" }}>
                {d.extension}
              </text>
              {(["rust", "ts"] as const).map((h, k) => {
                const has = d.validators.includes(h);
                const x = 290 + k * 96;
                return (
                  <g key={h}>
                    <rect x={x} y={y + 3} width={88} height={26} rx={13} fill="var(--panel)" stroke={has ? `var(--${h})` : "var(--none)"} strokeDasharray={has ? undefined : "3 3"} />
                    <text x={x + 44} y={y + 20} fontSize={11.5} textAnchor="middle" style={{ fill: has ? `var(--${h})` : "var(--none)", fontFamily: "var(--mono)" }}>
                      {h === "ts" ? "TypeScript" : "Rust"}
                    </text>
                  </g>
                );
              })}
              <rect x={560} y={y} width={210} height={32} rx={8} fill="var(--panel)" stroke="var(--accent)" />
              <text x={572} y={y + 21} fontSize={12.5} fontWeight={600} style={{ fill: "var(--accent)", fontFamily: "var(--mono)" }}>
                {d.module_id}
              </text>
              <rect x={850} y={y} width={246} height={32} rx={8} fill="var(--panel-2)" stroke="var(--rule)" />
              <text x={862} y={y + 21} fontSize={12} style={{ fontFamily: "var(--mono)" }}>
                knowledge-packs/{d.pack_id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
