/* ============================================================================
 * Artifact — renders one module output_delta by its `kind`.
 *
 * Shared by the blank surface, the legacy terminal (/terminal) and tutorial
 * RunnableCells. Every show/hide toggle goes through useDisclosure(), so a
 * host that wraps its tree in <FlatContext.Provider value={true}> gets every
 * section rendered open with no toggles: the surface does this, because a
 * page is an inert snapshot and nothing on it may be hidden.
 * ========================================================================== */

import { tbToString } from "@/lib/turbulance";
import { MetricsDashboard } from "@sachikonye/sbs/react";
import WorkspaceValue from "@/components/shapeshifter/WorkspaceValue";
import { SpraypaintResult, SpraypaintVerify } from "@/components/sandboxes/spraypaint/SpraypaintResult";
import { CodeBlock as InterceptorCodeBlock, ConsoleOutput as InterceptorConsoleOutput, WindTunnelReadout } from "@/components/sandboxes/interceptor/InterceptorConsole";
import { useDisclosure } from "@/components/artifacts/disclosure";
import VisBoard from "@/components/vis/VisBoard";
import SurfaceArtifact from "@/components/surface/SurfaceArtifacts";

// The show/hide control for one disclosure. Toggleable hosts get a button;
// flat hosts (the surface) get a static label, since the section is open.
function Toggle({ d, open, closed, flatLabel, className = "text-xs text-gray-500 hover:text-gray-300" }) {
  if (d.flat) {
    return flatLabel ? <div className="text-xs text-gray-600">{flatLabel}</div> : null;
  }
  return (
    <button type="button" className={className} onClick={d.toggle}>
      {d.open ? open : closed}
    </button>
  );
}

// ────────────────────────────────────────────────────────────
//  Artifact renderers.
// ────────────────────────────────────────────────────────────

function ProteinHeader({ name, p }) {
  return (
    <div className="flex items-baseline gap-6 mb-3 flex-wrap">
      <span className="text-white text-base">{name}</span>
      {p.gene && <span className="text-gray-500 text-xs">{p.gene}</span>}
      {p.uniprot && <span className="text-gray-600 text-xs">{p.uniprot}</span>}
      {p.length && <span className="text-gray-600 text-xs">{p.length} aa</span>}
      {p.role && <span className="text-gray-500 text-xs italic">{p.role}</span>}
    </div>
  );
}

function Field({ label, value }) {
  if (!value || (Array.isArray(value) && value.length === 0)) return null;
  return (
    <div className="flex mb-1">
      <span className="text-gray-500 w-28 shrink-0">{label}</span>
      <span className="text-gray-300 flex-1">
        {Array.isArray(value) ? value.join(", ") : value}
      </span>
    </div>
  );
}

function ArtifactProtein({ name, payload, aspect }) {
  const p = payload;
  if (aspect === "function") {
    return (<div><ProteinHeader name={name} p={p} /><Field label="function" value={p.function} /></div>);
  }
  if (aspect === "diseases") {
    return (<div><ProteinHeader name={name} p={p} /><Field label="diseases" value={p.diseases} /></div>);
  }
  if (aspect === "interacts") {
    return (<div><ProteinHeader name={name} p={p} /><Field label="interacts" value={p.interacts} /></div>);
  }
  if (aspect === "domains") {
    return (<div><ProteinHeader name={name} p={p} /><Field label="domains" value={p.domains} /></div>);
  }
  return (
    <div>
      <ProteinHeader name={name} p={p} />
      <Field label="function" value={p.function} />
      <Field label="domains" value={p.domains} />
      <Field label="diseases" value={p.diseases} />
      <Field label="interacts" value={p.interacts} />
      <Field label="pathway" value={p.pathway} />
      <Field label="location" value={p.localization} />
    </div>
  );
}

function ArtifactCompare({ a, b }) {
  const rows = [
    ["role", a.payload.role, b.payload.role],
    ["length", `${a.payload.length} aa`, `${b.payload.length} aa`],
    ["function", a.payload.function, b.payload.function],
    ["domains", a.payload.domains?.join(", "), b.payload.domains?.join(", ")],
    ["diseases", a.payload.diseases?.join(", "), b.payload.diseases?.join(", ")],
    ["pathway", a.payload.pathway, b.payload.pathway],
  ];
  return (
    <div>
      <div className="flex items-baseline gap-6 mb-4">
        <span className="text-white text-base">{a.name}</span>
        <span className="text-gray-500 text-xs">vs</span>
        <span className="text-white text-base">{b.name}</span>
      </div>
      {rows.map(([label, va, vb]) => (
        <div key={label} className="grid grid-cols-[7rem_1fr_1fr] gap-4 mb-2 text-xs">
          <span className="text-gray-500">{label}</span>
          <span className="text-gray-300">{va || "—"}</span>
          <span className="text-gray-300">{vb || "—"}</span>
        </div>
      ))}
    </div>
  );
}

function ArtifactFind({ query, items }) {
  if (!items.length) {
    return (<div className="text-gray-500 italic">no hits for &quot;{query}&quot;</div>);
  }
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">
        nearest to <span className="text-gray-300">&quot;{query}&quot;</span>
      </div>
      <ul className="text-gray-300">
        {items.map((it, i) => (
          <li key={i} className="py-1.5 grid grid-cols-[1.5rem_8rem_5rem_1fr] gap-3 items-baseline text-xs">
            <span className="text-gray-600">[{i + 1}]</span>
            <span className="text-gray-200 truncate">{it.name}</span>
            <span className="text-gray-600">d={it.distance.toFixed(3)}</span>
            <span className="text-gray-400 truncate">{it.source || ""}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactNote({ name, text, address, tier }) {
  return (
    <div>
      <div className="flex items-baseline gap-6 mb-2 flex-wrap">
        <span className="text-white text-base">{name}</span>
        <span className="text-gray-600 text-xs">addr {address.slice(0, 12)}</span>
        <span className="text-gray-600 text-xs">{tier}</span>
      </div>
      <div className="text-gray-300">{text}</div>
    </div>
  );
}

function ArtifactObjectList({ items }) {
  if (!items.length) {
    return <div className="text-gray-500 italic">memory is empty</div>;
  }
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">memory ({items.length})</div>
      <ul>
        {items.map((it, i) => (
          <li key={i} className="py-1 grid grid-cols-[9rem_8rem_4rem_1fr] gap-3 items-baseline text-xs">
            <span className="text-gray-600">{it.address.slice(0, 12)}</span>
            <span className="text-gray-200 truncate">{it.name}</span>
            <span className="text-gray-600">{it.tier}</span>
            <span className="text-gray-400 truncate">{it.source || ""}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactDump({ name, object }) {
  if (!object) {
    return <div className="text-gray-500 italic">dump {name}: not found</div>;
  }
  const o = object;
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">dump {name}</div>
      <Field label="address" value={o.address} />
      <Field label="coord" value={`S(${o.coord.k.toFixed(3)},${o.coord.t.toFixed(3)},${o.coord.e.toFixed(3)})`} />
      <Field label="tier" value={o.tier} />
      {typeof o.payload === "string" ? (
        <Field label="text" value={o.payload} />
      ) : (
        <div className="flex mb-1">
          <span className="text-gray-500 w-28 shrink-0">payload</span>
          <pre className="text-gray-300 flex-1 whitespace-pre-wrap font-mono text-xs">
            {JSON.stringify(o.payload, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function ArtifactStats({ stats }) {
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">kernel stats</div>
      <Field label="objects" value={String(stats.objects)} />
      <Field label="PVE ok" value={String(stats.pveOk)} />
      <Field label="PVE rejected" value={String(stats.pveRej)} />
      <Field label="TEM samples" value={String(stats.tem)} />
    </div>
  );
}

function ArtifactTrace({ log }) {
  if (!log.length) return <div className="text-gray-500 italic">no activity</div>;
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">activity ({log.length})</div>
      <ul className="text-gray-400 text-xs leading-relaxed font-mono">
        {log.slice(-20).map((l, i) => (<li key={i}>{l}</li>))}
      </ul>
    </div>
  );
}

function ArtifactSorted({ items }) {
  return (
    <div>
      <div className="text-gray-500 text-xs mb-2">sorted by S-distance to origin</div>
      <ul>
        {items.map((it, i) => (
          <li key={i} className="py-1 grid grid-cols-[2rem_9rem_8rem] gap-3 items-baseline text-xs">
            <span className="text-gray-600">{i + 1}</span>
            <span className="text-gray-600">{it.address.slice(0, 12)}</span>
            <span className="text-gray-200">{it.name}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactProcesses({ items }) {
  if (!items.length) return <div className="text-gray-500 italic">no processes</div>;
  return (
    <ul className="text-gray-300 text-xs font-mono">
      {items.map((p, i) => (
        <li key={i}>
          {p.name} <span className="text-gray-500">state={p.state}</span>
        </li>
      ))}
    </ul>
  );
}

function ArtifactVerify({ samples, message }) {
  return (
    <div>
      <div className="text-gray-300">{message}</div>
      <div className="text-gray-600 text-xs">samples observed: {samples}</div>
    </div>
  );
}

// Label → value rows. The generic shape for any status/config readout, so a
// module that reports "a few named facts" needs no renderer of its own.
function ArtifactKV({ title, rows }) {
  const list = Array.isArray(rows) ? rows : [];
  return (
    <div className="text-sm">
      {title && <div className="text-gray-500 text-xs mb-2">{title}</div>}
      {list.length === 0 && <div className="text-gray-500 italic">(nothing to report)</div>}
      {list.map(([label, value], i) => (
        <div key={i} className="grid grid-cols-[12rem_1fr] gap-4 mb-1 text-xs">
          <span className="text-gray-500 whitespace-pre">{label}</span>
          <span className="text-gray-200 break-all">{value == null ? "—" : String(value)}</span>
        </div>
      ))}
    </div>
  );
}

function ArtifactText({ lines }) {
  return (
    <div className="text-gray-300">
      {lines.map((line, i) => (<p key={i}>{line}</p>))}
    </div>
  );
}

function ArtifactTurbulance({ tb }) {
  if (!tb) return null;
  const lines = (tb.output || []).map((v) => {
    try { return tbToString ? tbToString(v) : String(v); }
    catch { return String(v); }
  });
  return (
    <div className="text-gray-300">
      {tb.ok === false && tb.error && (
        <p className="text-red-400 mb-2">
          turbulance error{tb.error.line ? ` (line ${tb.error.line})` : ""}: {tb.error.message}
        </p>
      )}
      {lines.length > 0 && (
        <pre className="font-mono text-sm whitespace-pre-wrap">{lines.join("\n")}</pre>
      )}
      {tb.propositions && tb.propositions.length > 0 && (
        <div className="mt-3 text-xs text-gray-500">
          <span className="text-gray-400">propositions: </span>
          {tb.propositions.map((p, i) => (
            <span key={i}>{p.name}{i < tb.propositions.length - 1 ? ", " : ""}</span>
          ))}
        </div>
      )}
      {tb.points && tb.points.length > 0 && (
        <div className="mt-1 text-xs text-gray-500">
          <span className="text-gray-400">points: </span>{tb.points.length}
        </div>
      )}
      {lines.length === 0 && !tb.error && (
        <p className="text-gray-500">(script ran; no output emitted)</p>
      )}
    </div>
  );
}

function ArtifactScope({ result, log }) {
  const logD = useDisclosure();
  if (!result) return null;

  const fmt = (n, d = 3) => (typeof n === "number" && isFinite(n) ? n.toFixed(d) : "—");
  const se = result.sEntropy || {};
  const vis = result.visualData || {};
  const goals = result.goalStatus || [];

  return (
    <div className="text-gray-300 font-mono text-sm">
      <div className="mb-2">
        <span className="text-gray-400">structure: </span>
        <span className="text-white">{result.structure || "—"}</span>
        {vis.activeVisMode && (
          <span className="ml-3 text-gray-500">
            visualise: <span className="text-white">{vis.activeVisMode}</span>
          </span>
        )}
      </div>

      <div className="text-xs text-gray-400 mb-2">
        <span>
          S = <span className="text-white">{fmt(se.sum)}</span>
          {"  "}(k {fmt(se.sk)} · t {fmt(se.st)} · e {fmt(se.se)})
        </span>
        {result.distance != null && (
          <span className="ml-3">
            d = <span className="text-white">{fmt(result.distance, 2)} µm</span>
            {result.uncertainty != null && (
              <span className="text-gray-500"> ± {fmt(result.uncertainty, 2)}</span>
            )}
          </span>
        )}
      </div>

      {goals.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-1">
          {goals.map((g, i) => (
            <span
              key={i}
              className={`text-xs px-1.5 py-0.5 rounded ${
                g.passed ? "bg-green-900/40 text-green-300" : "bg-red-900/40 text-red-300"
              }`}
            >
              {g.metric} {g.op} {g.threshold}{g.unit} {g.passed ? "✓" : "✗"} ({fmt(g.actual, 2)})
            </span>
          ))}
        </div>
      )}

      <div className="text-xs text-gray-500">
        field {vis.width || "?"}×{vis.height || "?"}
        {typeof result.channelCapacity?.snr === "number" && (
          <span> · SNR {fmt(result.channelCapacity.snr, 1)}</span>
        )}
      </div>

      {log && log.length > 0 && (
        <div className="mt-2">
          <Toggle d={logD} open={`▾ hide log (${log.length})`} closed={`▸ show log (${log.length})`} flatLabel={`log (${log.length})`} />
          {logD.open && (
            <pre className="mt-1 whitespace-pre-wrap text-xs text-gray-500">{log.join("\n")}</pre>
          )}
        </div>
      )}
    </div>
  );
}

function ArtifactLavoisier({ summary, records, config }) {
  const recD = useDisclosure();
  if (!summary) return null;

  const perClass = Object.entries(summary.perClass || {});
  const perAdduct = Object.entries(summary.perAdduct || {});
  const [mzLo, mzHi] = summary.mzRange || [0, 0];
  const [iLo, iHi] = summary.intensityRange || [0, 0];
  const fmt = (n) => (typeof n === "number" ? n.toFixed(4) : String(n));

  return (
    <div className="text-gray-300">
      <div className="mb-2">
        <span className="text-gray-400">
          {config?.experimentType || "run"}, {config?.analyser || "?"},
          polarity {config?.polarity || "?"}, CE{" "}
          {config?.collisionEnergy_eV ?? "?"} eV
        </span>
      </div>

      <div className="text-sm">
        <p>records: <span className="text-white">{summary.count}</span></p>
        <p>
          m/z range: <span className="text-white">{fmt(mzLo)} – {fmt(mzHi)}</span>
        </p>
        <p>
          intensity range:{" "}
          <span className="text-white">{fmt(iLo)} – {fmt(iHi)}</span>
        </p>
        <p>
          mean partition entropy:{" "}
          <span className="text-white">{fmt(summary.avgEntropy)}</span>
        </p>
      </div>

      {perClass.length > 0 && (
        <div className="mt-2 text-xs text-gray-500">
          <span className="text-gray-400">by class: </span>
          {perClass.map(([k, v], i) => (
            <span key={k}>
              {k}={v}
              {i < perClass.length - 1 ? ", " : ""}
            </span>
          ))}
        </div>
      )}
      {perAdduct.length > 0 && (
        <div className="mt-1 text-xs text-gray-500">
          <span className="text-gray-400">by adduct: </span>
          {perAdduct.map(([k, v], i) => (
            <span key={k}>
              {k}={v}
              {i < perAdduct.length - 1 ? ", " : ""}
            </span>
          ))}
        </div>
      )}
      {summary.shellsHistogram && summary.shellsHistogram.length > 0 && (
        <div className="mt-1 text-xs text-gray-500">
          <span className="text-gray-400">principal shells: </span>
          {summary.shellsHistogram.map((b, i) => (
            <span key={b.n}>
              n={b.n}:{b.count}
              {i < summary.shellsHistogram.length - 1 ? ", " : ""}
            </span>
          ))}
        </div>
      )}

      {records && records.length > 0 && (
        <div className="mt-3">
          <Toggle d={recD} className="text-xs text-blue-400 hover:text-blue-300" open="hide records" closed={`show ${records.length} records`} flatLabel={`records (${records.length})`} />
          {recD.open && (
            <pre className="mt-2 text-xs font-mono whitespace-pre-wrap text-gray-400">
              {records.slice(0, 50).map((r) =>
                `${(r.name || r.analyteClass || "?").padEnd(14)} ` +
                `${(r.adduct || "").padEnd(8)} ` +
                `m/z=${fmt(r.precursorMz)}  ` +
                `I=${fmt(r.intensity)}`
              ).join("\n")}
              {records.length > 50 ? `\n… and ${records.length - 50} more` : ""}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function ArtifactZangalewa({ caption, leaves, coord, provider, model }) {
  const primary = Array.isArray(leaves) && leaves.length > 0 ? leaves[0] : null;
  const params = primary?.params;
  return (
    <div className="text-gray-300">
      {caption && <div className="mb-2 text-xs text-gray-500">{caption}</div>}
      {(provider || model) && (
        <div className="mb-1 text-xs text-gray-600">
          <span className="text-gray-400">via:</span> {provider}
          {model && <> · {model}</>}
        </div>
      )}
      {coord && (
        <div className="mb-2 text-xs text-gray-500">
          <span className="text-gray-400">coord:</span> S_k={coord.S_k?.toFixed?.(2) ?? coord.S_k},{" "}
          S_t={coord.S_t?.toFixed?.(2) ?? coord.S_t}, S_e={coord.S_e?.toFixed?.(2) ?? coord.S_e}
        </div>
      )}
      {params && (
        <div>
          <div className="mb-1">
            <span className="text-white text-sm">{params.title}</span>
            {params.kind && (
              <span className="ml-2 text-xs text-gray-500">({params.kind})</span>
            )}
          </div>
          {params.tag && (
            <div className="mb-2 text-xs text-yellow-400">{params.tag}</div>
          )}
          {Array.isArray(params.sections) &&
            params.sections.map((s, i) => (
              <div key={i} className="mb-2">
                <p className="text-xs text-gray-400">{s.heading}</p>
                <p className="text-sm">{s.body}</p>
              </div>
            ))}
          {Array.isArray(params.references) && params.references.length > 0 && (
            <div className="mt-3 text-xs">
              <span className="text-gray-400">references:</span>
              <ul className="ml-4 mt-1">
                {params.references.map((r, i) => (
                  <li key={i}>
                    {r.url ? (
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-blue-400 hover:text-blue-300"
                      >
                        {r.citation}
                      </a>
                    ) : (
                      <span>{r.citation}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      {!primary && (
        <p className="text-gray-500">(no leaves returned)</p>
      )}
    </div>
  );
}

function ArtifactCatalystList({ entries }) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return <p className="text-gray-500 text-sm">(no catalysts registered)</p>;
  }
  return (
    <div className="text-gray-300 text-sm">
      <p className="text-xs text-gray-500 mb-2">{entries.length} catalyst{entries.length === 1 ? "" : "s"}</p>
      <ul>
        {entries.map((e) => (
          <li key={e.name} className="mb-2">
            <span className="text-white font-mono">{e.name}</span>
            <span className="text-gray-500"> — {e.availability || "?"}</span>
            <span className="text-gray-500"> · {e.cost_hint || "?"}</span>
            <div className="ml-4 text-xs">
              <p className="text-blue-400 font-mono">{e.url}</p>
              {Array.isArray(e.capabilities) && e.capabilities.length > 0 && (
                <p className="text-gray-500">caps: {e.capabilities.join(", ")}</p>
              )}
              {e.notes && <p className="text-gray-500">note: {e.notes}</p>}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactCatalystEntry({ entry }) {
  if (!entry) return null;
  return (
    <div className="text-gray-300 text-sm">
      <p><span className="text-gray-400">name:</span> <span className="text-white font-mono">{entry.name}</span></p>
      <p><span className="text-gray-400">url:</span> <span className="text-blue-400 font-mono text-xs">{entry.url}</span></p>
      <p><span className="text-gray-400">auth:</span> {entry.auth?.kind || "none"}</p>
      <p><span className="text-gray-400">cost:</span> {entry.cost_hint || "?"}</p>
      <p><span className="text-gray-400">availability:</span> {entry.availability || "?"}</p>
      {Array.isArray(entry.capabilities) && entry.capabilities.length > 0 && (
        <p><span className="text-gray-400">capabilities:</span> {entry.capabilities.join(", ")}</p>
      )}
      {entry.notes && <p><span className="text-gray-400">notes:</span> {entry.notes}</p>}
      {entry.added_at && <p className="text-xs text-gray-500 mt-1">added {entry.added_at}</p>}
    </div>
  );
}

function ArtifactCatalystPing({ name, url, result, lines }) {
  const ok = result?.ok;
  return (
    <div className="text-gray-300 text-sm">
      <p>
        <span className={ok ? "text-green-400" : "text-red-400"}>{ok ? "●" : "○"}</span>{" "}
        <span className="text-white font-mono">{name}</span>
        <span className="text-gray-500 text-xs"> — {url}</span>
      </p>
      <div className="ml-4 text-xs">
        {Array.isArray(lines) && lines.slice(1).map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
    </div>
  );
}

function ArtifactGatewayMachines({ entries }) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return <p className="text-gray-500 text-sm">(no machines paired)</p>;
  }
  return (
    <div className="text-gray-300 text-sm">
      <p className="text-xs text-gray-500 mb-2">{entries.length} machine{entries.length === 1 ? "" : "s"}</p>
      <ul>
        {entries.map((c) => (
          <li key={c.name} className="mb-1">
            <span className={c.live ? "text-green-400" : "text-gray-600"}>{c.live ? "●" : "○"}</span>{" "}
            <span className="text-white font-mono">{c.name}</span>
            <span className="text-gray-500"> — {c.live ? "live" : "asleep"}</span>
            {Array.isArray(c.capabilities) && c.capabilities.length > 0 && (
              <span className="text-gray-500 text-xs"> · {c.capabilities.join(", ")}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactGatewayPairToken({ name, token, expires_at }) {
  return (
    <div className="text-gray-300 text-sm">
      <p>
        <span className="text-green-400">paired</span>{" "}
        <span className="text-white font-mono">{name}</span>
      </p>
      <p className="text-xs text-gray-500 mt-1">
        this token is shown once — paste it into that machine so it can dial in:
      </p>
      <pre className="mt-2 p-2 bg-black/40 border border-gray-700 rounded text-xs font-mono text-yellow-300 whitespace-pre-wrap break-all">
        {token}
      </pre>
      {expires_at && (
        <p className="text-xs text-gray-500 mt-1">
          expires {new Date(expires_at * 1000).toISOString()}
        </p>
      )}
    </div>
  );
}

function ArtifactGatewayRun({ executed_on, note, results, trace }) {
  return (
    <div className="text-gray-300 text-sm">
      <p>
        <span className="text-gray-400">ran on:</span>{" "}
        <span className="text-white font-mono">{executed_on}</span>
      </p>
      {note && (
        <p className="text-yellow-400 text-xs mt-1">note: {note}</p>
      )}
      {Array.isArray(results) && results.length > 0 && (
        <pre className="mt-2 font-mono text-xs whitespace-pre-wrap">
          {JSON.stringify(results, null, 2)}
        </pre>
      )}
      {Array.isArray(trace) && trace.length > 0 && (
        <div className="mt-2 text-xs text-gray-500">
          <span className="text-gray-400">trace: </span>{trace.join("  ")}
        </div>
      )}
    </div>
  );
}

function ArtifactGatewayExperiments({ entries }) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return <p className="text-gray-500 text-sm">(no experiments yet)</p>;
  }
  return (
    <div className="text-gray-300 text-sm">
      <p className="text-xs text-gray-500 mb-2">
        {entries.length} experiment{entries.length === 1 ? "" : "s"}
      </p>
      <ul>
        {entries.map((e) => (
          <li key={e.id} className="mb-1">
            <span className="text-white font-mono">{e.name}</span>{" "}
            <span className="text-gray-600 text-xs">({e.id})</span>
            {e.standing?.kind === "owner" ? (
              <span className="text-emerald-400 text-xs"> — owner</span>
            ) : (
              <span className="text-gray-500 text-xs">
                {" "}
                — grantee · [{(e.standing?.capabilities || []).join(", ") || "no capabilities"}]
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactGatewayGrants({ entries }) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return <p className="text-gray-500 text-sm">(no grants yet)</p>;
  }
  return (
    <div className="text-gray-300 text-sm">
      <p className="text-xs text-gray-500 mb-2">
        {entries.length} grant{entries.length === 1 ? "" : "s"}
      </p>
      <ul>
        {entries.map((g) => (
          <li key={g.account_id} className="mb-1">
            <span className="text-white font-mono">{g.account_id}</span>
            <span className="text-gray-500 text-xs"> — [{g.capabilities.join(", ") || "no capabilities"}]</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactGatewayDispatch({ module, executed_on, act_id, result }) {
  return (
    <div className="text-gray-300 text-sm">
      <p>
        <span className="text-gray-400">dispatched</span>{" "}
        <span className="text-white font-mono">{module}</span>{" "}
        <span className="text-gray-500">on {executed_on}</span>{" "}
        <span className="text-gray-600 text-xs">(act {act_id})</span>
      </p>
      {result && (
        <pre className="mt-2 font-mono text-xs whitespace-pre-wrap">
          {JSON.stringify(result.output_delta ?? result, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ArtifactSpraypaintIndexResult({ root, documents, passages, scenes, would_index, identity_fingerprint, elapsed_ms }) {
  const isDryRun = typeof would_index === "number";
  return (
    <div className="text-gray-300 text-sm">
      <p>
        <span className="text-green-400">{isDryRun ? "would index" : "indexed"}</span>{" "}
        {isDryRun ? (
          <span className="text-white">{would_index} file(s)</span>
        ) : (
          <span className="text-white">{documents} document(s), {passages} passage(s), {scenes} scene(s)</span>
        )}
      </p>
      <p className="text-xs text-gray-500 mt-1 font-mono truncate">{root}</p>
      {identity_fingerprint && (
        <p className="text-xs text-gray-600 mt-1 font-mono truncate">fp: {identity_fingerprint}</p>
      )}
      {typeof elapsed_ms === "number" && <p className="text-xs text-gray-600 mt-1">{elapsed_ms} ms</p>}
    </div>
  );
}

function ArtifactWebSearchResult({ query, content, webSearchQueries, sources, grounded }) {
  return (
    <div className="text-gray-300 text-sm">
      <div className="text-xs text-gray-500 mb-2">
        <span className="text-gray-400">web search</span>{" "}
        <span className="text-white font-mono">&quot;{query}&quot;</span>
        {!grounded && <span className="text-yellow-400"> · ungrounded (no citations returned)</span>}
      </div>
      <p className="whitespace-pre-wrap leading-relaxed">{content}</p>
      {Array.isArray(sources) && sources.length > 0 && (
        <ol className="mt-3 text-xs text-gray-400 space-y-1 border-t border-gray-800 pt-2">
          {sources.map((s) => (
            <li key={s.index}>
              [{s.index + 1}]{" "}
              {s.uri ? (
                <a href={s.uri} target="_blank" rel="noreferrer" className="text-blue-400 hover:underline">
                  {s.title || s.uri}
                </a>
              ) : (
                <span>{s.title || "(untitled source)"}</span>
              )}
            </li>
          ))}
        </ol>
      )}
      {Array.isArray(webSearchQueries) && webSearchQueries.length > 0 && (
        <p className="mt-2 text-xs text-gray-600">
          queries issued: {webSearchQueries.join(" · ")}
        </p>
      )}
    </div>
  );
}

function ArtifactSearchCombined({ query, local, local_ok, web, web_ok }) {
  return (
    <div className="text-gray-300 text-sm space-y-4">
      <div>
        <p className="text-xs uppercase tracking-wider text-gray-500 mb-2">local (spraypaint)</p>
        {local_ok ? <Artifact result={local} /> : <ArtifactText lines={local?.lines || ["(local search failed)"]} />}
      </div>
      <div className="border-t border-gray-800 pt-3">
        <p className="text-xs uppercase tracking-wider text-gray-500 mb-2">internet (web search)</p>
        {web_ok ? <Artifact result={web} /> : <ArtifactText lines={web?.lines || ["(web search failed)"]} />}
      </div>
    </div>
  );
}

function ArtifactSpraypaintIdentity({ fingerprint, chi, floor, vertices, edges }) {
  return (
    <div className="text-gray-300 text-sm space-y-1">
      <p>
        <span className="text-gray-400">chi (char invariant):</span>{" "}
        <span className="text-white font-mono">{chi}</span>{" "}
        <span className="text-gray-500">
          {typeof chi === "number" && typeof floor === "number" && chi >= floor ? "≥" : "<"} floor {floor}
        </span>
      </p>
      <p className="text-xs text-gray-500 font-mono break-all">fp: {fingerprint}</p>
      <p className="text-xs text-gray-500">{vertices} vertices · {edges} edges</p>
    </div>
  );
}

function ArtifactSpraypaintCount({ committed_count }) {
  return (
    <p className="text-gray-300 text-sm">
      committed acts: <span className="text-white font-mono">{committed_count}</span>{" "}
      <span className="text-gray-500 text-xs">(monotone — only ever goes up)</span>
    </p>
  );
}

function ArtifactSpraypaintScenes({ scenes }) {
  const rows = Array.isArray(scenes) ? scenes : [];
  const maxDocs = Math.max(1, ...rows.map((r) => r.documents));
  return (
    <div className="text-gray-300 text-sm">
      <ul className="space-y-1">
        {rows.map((r) => (
          <li key={r.scene} className="flex items-center gap-2">
            <span className="font-mono text-xs w-40 truncate text-teal-300">{r.scene}</span>
            <span className="flex-1 bg-gray-800 rounded h-2 overflow-hidden">
              <span
                className="block h-full bg-teal-600"
                style={{ width: `${(r.documents / maxDocs) * 100}%` }}
              />
            </span>
            <span className="text-xs text-gray-500 w-32 text-right">
              {r.documents} doc · {r.passages} psg
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ────────────────────────────────────────────────────────────
//  Interceptor: AI code generation + sandboxed execution + vaHera
//  monitoring + wind-tunnel stability testing. Renderers delegate the
//  actual panels to sandboxes/interceptor/InterceptorConsole.js; this
//  file only assembles them per output_delta.kind, same convention as
//  the spraypaint renderers above.
// ────────────────────────────────────────────────────────────

function ArtifactInterceptorGenerated({ language, code, provider, model, run_error }) {
  return (
    <div className="text-gray-300 text-sm">
      <div className="text-xs text-gray-500 mb-2">
        <span className="text-gray-400">interceptor generate</span>{" "}
        {provider && <span className="text-white">{provider}{model ? `/${model}` : ""}</span>}
      </div>
      <InterceptorCodeBlock language={language} code={code} />
      {run_error && <p className="text-red-400 text-xs">run failed: {run_error}</p>}
    </div>
  );
}

function ArtifactInterceptorRun({ language, code, ok, stdout, stderr, exit_code, elapsed_ms, timed_out, truncated, vahera_memory }) {
  return (
    <div className="text-gray-300 text-sm">
      <div className="text-xs text-gray-500 mb-2">
        <span className="text-gray-400">interceptor run</span>{" "}
        <span className="text-white font-mono">{language}</span>
      </div>
      <InterceptorCodeBlock language={language} code={code} />
      <InterceptorConsoleOutput ok={ok} stdout={stdout} stderr={stderr} exit_code={exit_code} elapsed_ms={elapsed_ms} timed_out={timed_out} truncated={truncated} />
      {vahera_memory && (
        <p className="text-xs text-gray-500 px-1">
          stored in vaHera as <span className="text-teal-300 font-mono">&quot;{vahera_memory}&quot;</span> —
          recall it with <span className="font-mono">memory find nearest</span>
        </p>
      )}
    </div>
  );
}

function ArtifactWindTunnelReport({ language, code, elapsed_ms, ...stability }) {
  return (
    <div className="text-gray-300 text-sm">
      <div className="text-xs text-gray-500 mb-2">
        <span className="text-gray-400">wind-tunnel test</span>{" "}
        <span className="text-white font-mono">{language}</span>
        {typeof elapsed_ms === "number" && <span className="text-gray-500"> · {elapsed_ms} ms</span>}
      </div>
      <InterceptorCodeBlock language={language} code={code} />
      <WindTunnelReadout {...stability} />
    </div>
  );
}

function ArtifactInterceptorAssist({ language, task, code, provider, model, run, vahera_memory, windtunnel }) {
  return (
    <div className="text-gray-300 text-sm">
      <div className="text-xs text-gray-500 mb-2">
        <span className="text-gray-400">interceptor assist</span>{" "}
        <span className="text-white">&quot;{task}&quot;</span>{" "}
        {provider && <span className="text-gray-500">via {provider}{model ? `/${model}` : ""}</span>}
      </div>
      <InterceptorCodeBlock language={language} code={code} />
      {run && (
        <InterceptorConsoleOutput ok={run.ok} stdout={run.stdout} stderr={run.stderr} exit_code={run.exit_code} elapsed_ms={run.elapsed_ms} timed_out={run.timed_out} truncated={run.truncated} />
      )}
      {vahera_memory && (
        <p className="text-xs text-gray-500 px-1 mb-2">
          stored in vaHera as <span className="text-teal-300 font-mono">&quot;{vahera_memory}&quot;</span> —
          recall it with <span className="font-mono">memory find nearest</span>
        </p>
      )}
      {windtunnel && <WindTunnelReadout {...windtunnel} />}
    </div>
  );
}

function ArtifactRemoteDispatch({ target, moduleId, url, elapsed_ms, status, remote, error_stage, error_message }) {
  const remoteOk = !!remote?.ok;
  return (
    <div className="text-gray-300 text-sm">
      <div className="text-xs text-gray-500 mb-2">
        <span className="text-gray-400">via:</span> {target}{" "}
        <span className="text-gray-400">→</span>{" "}
        <span className="text-white">{moduleId}</span>
        {typeof elapsed_ms === "number" && <> · {elapsed_ms} ms</>}
        {typeof status === "number" && <> · HTTP {status}</>}
      </div>
      {error_stage && (
        <div className="mb-2 text-red-400 text-xs">
          {error_stage} error: {error_message}
        </div>
      )}
      {remote && (
        <div className="mt-2">
          <p className={remoteOk ? "text-green-400 text-xs mb-1" : "text-red-400 text-xs mb-1"}>
            remote returned ok={String(remoteOk)}
            {remote._catalyst?.name && <> from {remote._catalyst.name}</>}
          </p>
          <pre className="text-xs text-gray-500 whitespace-pre-wrap overflow-x-auto">
            {JSON.stringify(remote.output_delta ?? remote, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

function ArtifactPurposeCarry({
  keep,
  regenerable,
  dropped,
  ambientFloor,
  residue_entries,
  diagnostics,
  goal_terms,
  budget,
  session_step_count,
}) {
  const bdD = useDisclosure();
  const keepCount = Array.isArray(keep) ? keep.length : 0;
  const regCount = Array.isArray(regenerable) ? regenerable.length : 0;
  const dropCount = Array.isArray(dropped) ? dropped.length : 0;
  const residuePairs = Array.isArray(residue_entries) ? residue_entries : [];

  const totalKeptCost = diagnostics?.totalKeptCost ?? 0;
  const budgetRemaining = diagnostics?.budgetRemaining ?? 0;
  const gap = diagnostics?.knapsackRelaxationGap ?? 0;

  return (
    <div className="text-gray-300">
      <div className="mb-2 text-xs text-gray-500">
        <span className="text-gray-400">session:</span> {session_step_count} steps
        {" · "}
        <span className="text-gray-400">floor:</span>{" "}
        {typeof ambientFloor === "number" ? ambientFloor.toFixed(3) : "?"}
        {" · "}
        <span className="text-gray-400">budget:</span> {budget}
      </div>

      {goal_terms && goal_terms.length > 0 && (
        <div className="mb-2 text-xs text-gray-500">
          <span className="text-gray-400">goal:</span> {goal_terms.join(", ")}
        </div>
      )}

      <div className="text-sm mb-2">
        <p>
          <span className="text-green-400">keep:</span> {keepCount}
          {"  ·  "}
          <span className="text-blue-400">regenerable:</span> {regCount}
          {"  ·  "}
          <span className="text-gray-500">dropped:</span> {dropCount}
        </p>
        <p className="text-xs text-gray-500 mt-1">
          cost {totalKeptCost}/{budget} (remaining {budgetRemaining})
          {gap > 0 && `, relaxation gap ${gap.toFixed(3)}`}
        </p>
      </div>

      {keepCount + regCount + dropCount > 0 && (
        <Toggle d={bdD} className="text-xs text-blue-400 hover:text-blue-300" open="hide breakdown" closed="show breakdown" />
      )}

      {bdD.open && (
        <div className="mt-2 text-xs">
          {keepCount > 0 && (
            <div>
              <p className="text-gray-400">keep ({keepCount}):</p>
              <ul className="ml-4">
                {keep.map((id) => {
                  const rentry = residuePairs.find(([k]) => k === id);
                  const r = rentry ? rentry[1] : null;
                  return (
                    <li key={id}>
                      <span className="text-green-400">{id}</span>
                      {r != null && (
                        <span className="text-gray-500"> — ρ={r.toFixed(3)}</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
          {regCount > 0 && (
            <div className="mt-2">
              <p className="text-gray-400">regenerable ({regCount}):</p>
              <ul className="ml-4">
                {regenerable.map((id) => (
                  <li key={id} className="text-blue-400">{id}</li>
                ))}
              </ul>
            </div>
          )}
          {dropCount > 0 && (
            <div className="mt-2">
              <p className="text-gray-400">dropped ({dropCount}):</p>
              <ul className="ml-4">
                {dropped.map((id) => (
                  <li key={id} className="text-gray-500">{id}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ArtifactGraffiti({ projects, diagnostics, ambient_floor }) {
  const projectNames = Object.keys(projects || {});
  return (
    <div className="text-gray-300">
      <div className="mb-2 text-xs text-gray-500">
        <span className="text-gray-400">floor:</span>{" "}
        {typeof ambient_floor === "number" ? ambient_floor.toFixed(3) : "?"}
        {" · "}
        <span className="text-gray-400">projects:</span> {projectNames.length}
      </div>
      {projectNames.length === 0 && (
        <p className="text-gray-500">(no projects yielded)</p>
      )}
      {projectNames.map((name) => {
        const yields = projects[name] || {};
        const yieldNames = Object.keys(yields);
        return (
          <div key={name} className="mb-3">
            <p className="text-white text-sm">{name}</p>
            {yieldNames.length === 0 ? (
              <p className="ml-2 text-xs text-gray-500">(no yields)</p>
            ) : (
              <ul className="ml-4 text-xs">
                {yieldNames.map((y) => {
                  const v = yields[y];
                  const preview =
                    typeof v === "string" ? v : JSON.stringify(v);
                  return (
                    <li key={y}>
                      <span className="text-gray-400">{y}:</span>{" "}
                      <span className="text-white">{preview}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        );
      })}
      {diagnostics && diagnostics.length > 0 && (
        <div className="mt-2 text-xs">
          <span className="text-gray-400">diagnostics:</span>
          <ul className="ml-4">
            {diagnostics.map((d, i) => (
              <li
                key={i}
                className={
                  d.severity === "error" ? "text-red-400" : "text-yellow-400"
                }
              >
                {d.severity}: {d.message}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ArtifactSmith({ ok, agents, diagnostics, steps, finalCounts }) {
  const traceD = useDisclosure();
  const list = agents || [];
  const totalFloor = list
    .map((a) => a.floor)
    .filter((f) => typeof f === "number")
    .reduce((s, f) => s + f, 0);

  const fmtPartition = (blocks) =>
    (blocks || []).map((b) => `{${b.join(",")}}`).join(" | ");

  return (
    <div className="text-gray-300">
      <div className="mb-2 text-xs text-gray-500">
        <span className={ok ? "text-green-400" : "text-red-400"}>
          {ok ? "checked" : "rejected"}
        </span>
        {" · "}
        <span className="text-gray-400">agents:</span> {list.length}
        {" · "}
        <span className="text-gray-400">Σfloor:</span> {totalFloor.toFixed(3)}
      </div>

      {list.length === 0 && <p className="text-gray-500">(no agents generated)</p>}

      {list.map((a) => (
        <div key={a.name} className="mb-3">
          <p className="text-white text-sm">
            {a.name}{" "}
            <span className="text-xs text-gray-500">
              [{a.regime}
              {a.nonLocal ? " · non-local" : ""}]
            </span>
          </p>
          <ul className="ml-4 text-xs">
            <li>
              <span className="text-gray-400">χ (character):</span>{" "}
              <span className="text-white">
                {typeof a.chi === "number" ? a.chi.toFixed(3) : "∞"}
              </span>
            </li>
            <li>
              <span className="text-gray-400">realised floor:</span>{" "}
              <span className="text-white">
                {typeof a.floor === "number" ? a.floor.toFixed(3) : "∞"}
              </span>
            </li>
            <li>
              <span className="text-gray-400">χ-partition:</span>{" "}
              <span className="text-white">{fmtPartition(a.chiPartition)}</span>
            </li>
          </ul>
        </div>
      ))}

      {diagnostics && diagnostics.length > 0 && (
        <div className="mt-2 text-xs">
          <span className="text-gray-400">diagnostics:</span>
          <ul className="ml-4">
            {diagnostics.map((d, i) => (
              <li
                key={i}
                className={
                  d.severity === "error" ? "text-red-400" : "text-yellow-400"
                }
              >
                {d.severity}: {d.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {steps && steps.length > 0 && (
        <div className="mt-3 text-xs">
          <Toggle d={traceD} className="text-gray-500 hover:text-gray-300" open="▾ hide run trace" closed={`▸ run trace (${steps.length} steps)`} flatLabel={`run trace (${steps.length} steps)`} />
          {finalCounts && (
            <span className="ml-2 text-gray-600">
              commits:{" "}
              {Object.entries(finalCounts)
                .map(([n, c]) => `${n}=${c}`)
                .join(" ")}
            </span>
          )}
          {traceD.open && (
            <div className={`mt-1 font-mono${traceD.flat ? "" : " max-h-48 overflow-y-auto"}`}>
              {steps.map((s, i) => (
                <div key={i} className="text-gray-500">
                  <span className="text-gray-600">t{s.tick}</span>{" "}
                  <span className="text-gray-400">{s.agent}</span>{" "}
                  <span
                    className={
                      s.outcome === "commit"
                        ? "text-green-400"
                        : s.outcome === "quiescent"
                        ? "text-blue-400"
                        : s.outcome === "decline"
                        ? "text-red-400"
                        : "text-gray-500"
                    }
                  >
                    {s.outcome}
                  </span>
                  {s.scene ? <> · {s.scene}</> : null}
                  {typeof s.residual === "number" ? (
                    <> · r={s.residual.toFixed(3)}</>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ArtifactSrnResult({ glyph, provider, model, node, elapsed_ms, value, chart }) {
  const rawD = useDisclosure();
  const scalar =
    value != null && typeof value === "object" && !Array.isArray(value)
      ? value.result ?? value.value ?? null
      : value;
  return (
    <div className="text-gray-300 font-mono text-sm">
      {glyph && (
        <div className="mb-2 text-xs text-gray-500">
          <span className="text-gray-400">glyph:</span>{" "}
          <span className="text-purple-300">{glyph}</span>
        </div>
      )}
      {(provider || node) && (
        <div className="mb-1 text-xs text-gray-600">
          {provider && <><span className="text-gray-400">via:</span> {provider}{model ? ` · ${model}` : ""} · </>}
          {node && <><span className="text-gray-400">node:</span> {node.replace(/^https?:\/\//, "")}</>}
          {elapsed_ms != null && <> · {elapsed_ms}ms</>}
        </div>
      )}
      {chart ? (
        <div className="mt-2">
          <div className="text-xs text-gray-500 mb-1">series ({chart.values.length})</div>
          <div className="flex items-end gap-px h-12 overflow-hidden">
            {(() => {
              const max = Math.max(...chart.values, 1);
              return chart.values.slice(0, 80).map((v, i) => (
                <div
                  key={i}
                  className="flex-1 min-w-px bg-purple-500/60 rounded-sm"
                  style={{ height: `${Math.round((v / max) * 100)}%` }}
                />
              ));
            })()}
          </div>
        </div>
      ) : scalar != null ? (
        <div className="text-white">{typeof scalar === "number" ? scalar.toFixed(6) : String(scalar)}</div>
      ) : null}
      {value != null && (
        <div className="mt-2">
          <Toggle d={rawD} open="▾ hide raw" closed="▸ raw response" flatLabel="raw response" />
        </div>
      )}
      {value != null && rawD.open && (
        <pre className="mt-1 text-xs text-gray-500 whitespace-pre-wrap overflow-x-auto">
          {JSON.stringify(value, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ArtifactSrnPeers({ peers, node, elapsed_ms }) {
  return (
    <div className="text-gray-300 text-sm">
      <div className="text-xs text-gray-500 mb-2">
        <span className="text-gray-400">forest peers</span>
        {node && <> · from {node.replace(/^https?:\/\//, "")}</>}
        {elapsed_ms != null && <> · {elapsed_ms}ms</>}
      </div>
      {(!peers || peers.length === 0) ? (
        <p className="text-gray-500">(no peers known yet)</p>
      ) : (
        <ul>
          {peers.map((p, i) => (
            <li key={i} className="py-0.5 text-xs font-mono">
              <span className="text-purple-300">{typeof p === "string" ? p : p.addr || JSON.stringify(p)}</span>
              {p.coord && <span className="text-gray-500"> coord=({p.coord.n},{p.coord.l},{p.coord.m},{p.coord.s})</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ArtifactSrnProbe({ target, ok, elapsed_ms, ...rest }) {
  return (
    <div className="text-gray-300 text-sm">
      <p>
        <span className={ok ? "text-green-400" : "text-red-400"}>{ok ? "●" : "○"}</span>{" "}
        <span className="font-mono text-purple-300">{(target || "").replace(/^https?:\/\//, "")}</span>
        {elapsed_ms != null && <span className="text-gray-500 text-xs"> · {elapsed_ms}ms</span>}
      </p>
      {Object.keys(rest).length > 0 && (
        <pre className="mt-1 text-xs text-gray-500 whitespace-pre-wrap">
          {JSON.stringify(rest, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ArtifactSrnError({ message }) {
  return <p className="text-red-400 text-sm font-mono">srn: {message}</p>;
}

function ArtifactPurpose({ synthesis, model, provider, federation, floor }) {
  return (
    <div className="text-gray-300">
      <div className="mb-2 text-xs text-gray-500">
        {provider && (
          <>
            <span className="text-gray-400">via:</span> {provider}
            {" · "}
          </>
        )}
        <span className="text-gray-400">model:</span> {model || "?"}
        {typeof floor === "number" && (
          <>
            {" · "}
            <span className="text-gray-400">floor:</span> {floor.toFixed(2)}
          </>
        )}
        {federation?.active_drafts?.length > 0 && (
          <>
            {" · "}
            <span className="text-gray-400">federation:</span>{" "}
            {federation.active_drafts.length} drafts
          </>
        )}
      </div>
      <pre className="whitespace-pre-wrap text-sm font-mono">{synthesis}</pre>
    </div>
  );
}

function ArtifactTriangleSources({ entries }) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return <p className="text-gray-500 text-sm">(no sources configured)</p>;
  }
  return (
    <div className="text-gray-300 text-sm">
      <p className="text-xs text-gray-500 mb-2">{entries.length} source{entries.length === 1 ? "" : "s"}</p>
      <ul>
        {entries.map((s) => (
          <li key={s.id} className="mb-1">
            <span className="text-white font-mono">{s.id}</span>
            <span className="text-gray-500"> — {s.kind}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ArtifactPurposeCli({ utterance, mode, payload, elapsed_ms }) {
  return (
    <div className="text-gray-300">
      <div className="mb-2 text-xs text-gray-500">
        <span className="text-gray-400">query:</span> {utterance}
        {" · "}
        <span className="text-gray-400">mode:</span> {mode}
        {" · "}
        <span className="text-gray-400">{elapsed_ms}ms</span>
      </div>
      <pre className="whitespace-pre-wrap text-sm font-mono">{JSON.stringify(payload, null, 2)}</pre>
    </div>
  );
}

function ArtifactShapeshifter({ term, workspace }) {
  const streamColor = (s) =>
    s === "stderr" ? "text-rose-400" : s === "stage" ? "text-sky-400" : "text-gray-300";
  return (
    <div className="text-gray-300 font-mono text-sm">
      {/* terminal stream: compile/run stages, logs, summary */}
      <div className="mb-3">
        {(term || []).map((line, i) => (
          <div key={i} className={`whitespace-pre-wrap ${streamColor(line.stream)}`}>
            {line.stream === "stage" ? `— ${line.text} —` : line.text}
          </div>
        ))}
      </div>
      {/* one inline panel/chart per produced workspace value (notebook cell) */}
      {workspace && workspace.length > 0 && (
        <div className="space-y-4 border-t border-gray-800 pt-3">
          {workspace.map((w, i) => (
            <div key={i}>
              <div className="mb-1 text-[10px] uppercase tracking-wider text-gray-500">
                {w.name} · {w.kind}
              </div>
              <WorkspaceValue entry={w} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ArtifactSBS({ summary, circuit, metrics, navigation, warnings }) {
  return (
    <div className="text-gray-300 space-y-3">
      <div className="font-mono text-sm text-white">{summary}</div>
      {warnings && warnings.length > 0 && (
        <div className="text-xs text-yellow-500/80 font-mono">
          {warnings.map((w, i) => (
            <div key={i}>warning: {w.message}</div>
          ))}
        </div>
      )}
      {metrics && circuit ? (
        <MetricsDashboard
          metrics={metrics}
          circuit={circuit}
          navigation={navigation}
        />
      ) : (
        <div className="text-xs text-gray-500">
          compiled; no circuit declared (nothing to observe)
        </div>
      )}
    </div>
  );
}

// The trajectory-as-knowledge-graph. Nodes = touched subtasks, edges = the
// carrier reads that landed this run (the run-induced causal relation), facts =
// the value-deltas the federation emitted onto each node. This is the SPARQL
// replacement: you walk the graph the run produced, not one you authored.
// One fact on a node in the graph view. A module fact ({module, ok, delta,
// findings}) shows its headline and expands to the module's real artifact; a
// bare value (e.g. an error string) just prints.
function CkgFact({ predicate, object }) {
  const d = useDisclosure();
  const open = d.open;
  const isError = predicate === "error";
  const moduleFact = object && typeof object === "object" && (object.module != null || object.ok != null);
  const delta = moduleFact ? object.delta : null;
  const renderable = ckgRenderable(delta);
  const headline = moduleFact
    ? (object.findings && object.findings.headline) || (delta && delta.kind) || "∅"
    : object == null
    ? "∅"
    : typeof object === "object"
    ? JSON.stringify(object)
    : String(object);
  return (
    <li>
      <button
        onClick={() => renderable && !d.flat && d.toggle()}
        className={`text-left ${renderable && !d.flat ? "hover:text-white" : "cursor-default"}`}
      >
        <span className={isError ? "text-red-400" : "text-gray-400"}>{predicate}:</span>{" "}
        {moduleFact && (
          <span className={object.ok ? "text-green-400" : "text-yellow-400"}>
            {object.ok ? "✓" : "×"}{" "}
          </span>
        )}
        <span className="text-white">{headline}</span>
        {renderable && !d.flat && (
          <span className="ml-2 text-[10px] text-gray-600">{open ? "▾" : "▸"}</span>
        )}
      </button>
      {open && renderable && (
        <div className="mt-1 mb-1 rounded border border-gray-800 bg-black/40 p-2">
          <Artifact result={delta} />
        </div>
      )}
    </li>
  );
}

function ArtifactCkgGraph({ nodes, edges, node_count, edge_count, fact_count }) {
  return (
    <div className="text-gray-300">
      <div className="mb-2 text-xs text-gray-500">
        <span className="text-gray-400">nodes:</span> {node_count}
        {" · "}
        <span className="text-gray-400">edges:</span> {edge_count}
        {" · "}
        <span className="text-gray-400">facts:</span> {fact_count}
        <span className="ml-2 text-gray-600">— this graph is the runtime trajectory</span>
      </div>

      <div className="mb-3 text-xs">
        <span className="text-gray-400">trajectory (this run):</span>{" "}
        {edges && edges.length ? (
          <span className="text-white">
            {edges
              .map((e) => `${e.from}→${e.to}(${e.magnitude})`)
              .join("  ")}
          </span>
        ) : (
          <span className="text-gray-500">(no edges induced)</span>
        )}
      </div>

      {(nodes || []).map((n) => (
        <div key={n.tau} className="mb-2">
          <p className="text-white text-sm">
            {n.tau}
            <span className="ml-2 text-xs text-gray-600">
              {(n.address || []).join("/")}
              {typeof n.signal === "number" ? ` · signal ${n.signal}` : ""}
            </span>
          </p>
          {n.facts && n.facts.length ? (
            <ul className="ml-4 text-xs">
              {n.facts.map((f, i) => (
                <CkgFact key={i} predicate={f.predicate} object={f.object} />
              ))}
            </ul>
          ) : (
            <p className="ml-4 text-xs text-gray-500">(no facts emitted)</p>
          )}
        </div>
      ))}
    </div>
  );
}

// The report the original pipeline never produced. It reads the audit (what
// ran) and the emitted facts (what the federation asserted), grouped by
// contributing module. It judges nothing: an error is reported as a fact.
// Module output kinds that <Artifact> draws a real view for. A fact whose delta
// is one of these gets an inline "show chart" expander; anything else (e.g. the
// echo smoke-test's bare {kind:"echo",value}) just shows its headline, so we
// never offer to expand into an empty box.
const CKG_RENDERABLE_KINDS = new Set([
  "sbs_result",
  "shapeshifter_run",
  "scope_run",
  "lavoisier_run",
  "graffiti_result",
  "purpose_carry",
  "turbulance_result",
  "text",
]);

function ckgRenderable(delta) {
  return !!(delta && typeof delta === "object" && CKG_RENDERABLE_KINDS.has(delta.kind));
}

// One contribution in the report: the module's findings headline, then the
// module's OWN artifact — the identical chart/panel you'd get dispatching that
// module directly — rendered by handing its full delta back to <Artifact>.
// Collapsed by default so the report reads as a dossier you expand section by
// section, not a wall of every chart at once.
function CkgContribution({ tau, ok, findings, delta }) {
  const d = useDisclosure();
  const open = d.open;
  const headline = findings && findings.headline;
  const props = (findings && findings.props) || [];
  const renderable = ckgRenderable(delta);
  return (
    <div className="mb-2 border-l border-gray-800 pl-3">
      <button
        onClick={() => renderable && !d.flat && d.toggle()}
        className={`text-left w-full ${renderable && !d.flat ? "hover:text-white" : "cursor-default"}`}
      >
        <span className="text-gray-400 text-xs">{tau}</span>{" "}
        <span className={ok ? "text-green-400" : "text-yellow-400"}>{ok ? "✓" : "·"}</span>{" "}
        <span className="text-white text-sm">{headline || (delta && delta.kind) || "∅"}</span>
        {renderable && !d.flat && (
          <span className="ml-2 text-[10px] text-gray-600">{open ? "▾ hide chart" : "▸ show chart"}</span>
        )}
      </button>
      {props.length > 0 && (
        <div className="ml-1 mt-0.5 text-[11px] text-gray-500 flex flex-wrap gap-x-3">
          {props.map((p, i) => (
            <span key={i}>
              <span className="text-gray-600">{p.label}:</span>{" "}
              <span className="text-gray-300">{String(p.value)}</span>
            </span>
          ))}
        </div>
      )}
      {open && renderable && (
        <div className="mt-2 rounded border border-gray-800 bg-black/40 p-2">
          {/* the module's real view — same component as a direct dispatch */}
          <Artifact result={delta} />
        </div>
      )}
    </div>
  );
}

function ArtifactCkgReport({
  tau_count,
  acts,
  edges,
  error_facts,
  contributors,
  contributions,
  audit,
}) {
  const auditD = useDisclosure();
  return (
    <div className="text-gray-300">
      <div className="mb-1 text-sm text-white">CKG report — assembled findings</div>
      <div className="mb-3 text-xs text-gray-500">
        <span className="text-gray-400">subtasks:</span> {tau_count}
        {" · "}
        <span className="text-gray-400">acts:</span> {acts}
        {" · "}
        <span className="text-gray-400">edges:</span> {edges}
        {" · "}
        <span className={error_facts ? "text-red-400" : "text-gray-400"}>error-facts:</span>{" "}
        {error_facts}
        {" · "}
        <span className="text-gray-400">contributors:</span>{" "}
        {contributors && contributors.length ? contributors.join(", ") : "(none)"}
      </div>

      {(contributors || []).length === 0 && (
        <p className="text-xs text-gray-500">
          no module asserted a fact yet — attach a module and dispatch (or carry).
        </p>
      )}

      {(contributors || []).map((mod) => (
        <div key={mod} className="mb-4">
          <p className="text-white text-sm mb-1 uppercase tracking-wider text-[11px] text-gray-400">
            {mod} — {(contributions[mod] || []).length} contribution
            {(contributions[mod] || []).length === 1 ? "" : "s"}
          </p>
          {(contributions[mod] || []).map((c, i) => (
            <CkgContribution
              key={i}
              tau={c.tau}
              ok={c.ok}
              findings={c.findings}
              delta={c.delta}
            />
          ))}
        </div>
      ))}

      <Toggle
        d={auditD}
        className="mt-1 text-xs text-gray-500 hover:text-gray-300 underline"
        open={`hide audit (${(audit || []).length} acts — every act ran, none was gated)`}
        closed={`show audit (${(audit || []).length} acts — every act ran, none was gated)`}
        flatLabel={`audit (${(audit || []).length} acts — every act ran, none was gated)`}
      />
      {auditD.open && (
        <ul className="ml-4 mt-1 text-xs text-gray-500">
          {(audit || []).map((a) => (
            <li key={a.act_id}>
              #{a.act_id} {a.tau}/{a.chunk} → {a.emitted_kind}
              {a.raised ? " (raised)" : ""}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Artifact({ result }) {
  if (!result) return null;
  switch (result.kind) {
    case "protein":         return <ArtifactProtein name={result.name} payload={result.payload} aspect={result.aspect} />;
    case "protein_compare": return <ArtifactCompare a={result.a} b={result.b} />;
    case "find":            return <ArtifactFind query={result.query} items={result.items} />;
    case "note":            return <ArtifactNote name={result.name} text={result.text} address={result.address} tier={result.tier} />;
    case "list_objects":    return <ArtifactObjectList items={result.items} />;
    case "dump":            return <ArtifactDump name={result.name} object={result.object} />;
    case "sorted_objects":  return <ArtifactSorted items={result.items} />;
    case "stats":           return <ArtifactStats stats={result.stats} />;
    case "trace":           return <ArtifactTrace log={result.log} />;
    case "processes":       return <ArtifactProcesses items={result.items} />;
    case "verify":          return <ArtifactVerify samples={result.samples} message={result.message} />;
    case "turbulance_result": return <ArtifactTurbulance tb={result.tb} />;
    case "lavoisier_run":   return <ArtifactLavoisier summary={result.summary} records={result.records} config={result.config} />;
    case "scope_run":       return <ArtifactScope result={result.result} log={result.log} />;
    case "shapeshifter_run": return <ArtifactShapeshifter term={result.term} workspace={result.workspace} />;
    case "sbs_result":      return <ArtifactSBS summary={result.summary} circuit={result.circuit} metrics={result.metrics} navigation={result.navigation} warnings={result.warnings} />;
    case "purpose_synthesis": return <ArtifactPurpose synthesis={result.synthesis} model={result.model} provider={result.provider} federation={result.federation} floor={result.floor} />;
    case "purpose_cli_result": return <ArtifactPurposeCli utterance={result.utterance} mode={result.mode} payload={result.payload} elapsed_ms={result.elapsed_ms} />;
    case "triangle_sources": return <ArtifactTriangleSources entries={result.entries} />;
    case "graffiti_result": return <ArtifactGraffiti projects={result.projects} diagnostics={result.diagnostics} ambient_floor={result.ambient_floor} />;
    case "purpose_carry":   return <ArtifactPurposeCarry keep={result.keep} regenerable={result.regenerable} dropped={result.dropped} ambientFloor={result.ambientFloor} residue_entries={result.residue_entries} diagnostics={result.diagnostics} goal_terms={result.goal_terms} budget={result.budget} session_step_count={result.session_step_count} />;
    case "catalyst_list":   return <ArtifactCatalystList entries={result.entries} />;
    case "catalyst_entry":  return <ArtifactCatalystEntry entry={result.entry} />;
    case "catalyst_ping":   return <ArtifactCatalystPing name={result.name} url={result.url} result={result.result} lines={result.lines} />;
    case "remote_dispatch": return <ArtifactRemoteDispatch target={result.target} moduleId={result.moduleId} url={result.url} elapsed_ms={result.elapsed_ms} status={result.status} remote={result.remote} error_stage={result.error_stage} error_message={result.error_message} />;
    case "purpose_carry_stats": return <ArtifactText lines={[`purpose-carry: ${result.stepCount} steps in session`, `ambient floor β = ${typeof result.ambientFloor === "number" ? result.ambientFloor.toFixed(3) : "?"}`]} />;
    case "zangalewa_render": return <ArtifactZangalewa caption={result.caption} leaves={result.leaves} coord={result.coord} provider={result.provider} model={result.model} />;
    case "desk_tag":        return <ArtifactText lines={[`desk: intent tagged.`, `reason: ${result.reason}`, `goal g = { ${result.terms.join(", ")} }`, `every act from now is scored for contribution toward this reason.`]} />;
    case "desk_stats":      return <ArtifactText lines={result.tagged ? [`desk: intent = "${result.reason}"`, `goal terms: ${result.term_count}`, `acts seen: ${result.acts_seen}  (necessary ${result.necessary} · purposeless ${result.purposeless})`] : ["desk: no intent tagged — the surface is blank."]} />;
    case "desk_surface":    return <ArtifactText lines={deskSurfaceLines(result)} />;
    case "srn_result":      return <ArtifactSrnResult glyph={result.glyph} provider={result.provider} model={result.model} node={result.node} elapsed_ms={result.elapsed_ms} value={result.value} chart={result.chart} />;
    case "srn_peers":       return <ArtifactSrnPeers peers={result.peers} node={result.node} elapsed_ms={result.elapsed_ms} />;
    case "srn_probe":       return <ArtifactSrnProbe target={result.target} ok={result.ok} elapsed_ms={result.elapsed_ms} />;
    case "srn_error":       return <ArtifactSrnError message={result.message} />;
    case "agent_generated": return <ArtifactSmith ok={result.ok} agents={result.agents} diagnostics={result.diagnostics} steps={result.steps} finalCounts={result.finalCounts} />;
    case "ckg_graph":       return <ArtifactCkgGraph nodes={result.nodes} edges={result.edges} node_count={result.node_count} edge_count={result.edge_count} fact_count={result.fact_count} />;
    case "ckg_report":      return <ArtifactCkgReport tau_count={result.tau_count} acts={result.acts} edges={result.edges} error_facts={result.error_facts} contributors={result.contributors} contributions={result.contributions} audit={result.audit} />;
    case "ckg_fingerprint": return <ArtifactText lines={[`fingerprint: ${result.fingerprint}`, `nodes: ${result.nodes}${result.edits && result.edits.length ? `  edits: ${result.edits.join(", ")}` : ""}`, result.note]} />;
    case "ckg_ack":         return <ArtifactText lines={[result.message, result.trajectory && result.trajectory.length ? `trajectory: ${result.trajectory.join("  ")}` : null, result.chunks ? `chunks: ${result.chunks.join(", ")}` : null, result.emitted ? `emitted: ${result.emitted.join(", ")}` : null].filter(Boolean)} />;
    case "text":            return <ArtifactText lines={result.lines} />;
    case "kv":              return <ArtifactKV title={result.title} rows={result.rows} />;
    case "vis":             return <VisBoard title={result.title} dataset={result.dataset} charts={result.charts} notes={result.notes} />;
    case "list":            return <ArtifactFind query={result.title || ""} items={result.items} />;
    case "gateway_session": return <ArtifactText lines={result.lines} />;
    case "gateway_machines": return <ArtifactGatewayMachines entries={result.entries} />;
    case "gateway_pair_token": return <ArtifactGatewayPairToken name={result.name} token={result.token} expires_at={result.expires_at} />;
    case "gateway_run":     return <ArtifactGatewayRun executed_on={result.executed_on} note={result.note} results={result.results} trace={result.trace} />;
    case "gateway_experiments": return <ArtifactGatewayExperiments entries={result.entries} />;
    case "gateway_grants":  return <ArtifactGatewayGrants entries={result.entries} />;
    case "gateway_dispatch": return <ArtifactGatewayDispatch module={result.module} executed_on={result.executed_on} act_id={result.act_id} result={result.result} />;
    case "spraypaint_result": return <SpraypaintResult {...result} />;
    case "spraypaint_index_result": return <ArtifactSpraypaintIndexResult root={result.root} documents={result.documents} passages={result.passages} scenes={result.scenes} would_index={result.would_index} identity_fingerprint={result.identity_fingerprint} elapsed_ms={result.elapsed_ms} />;
    case "web_search_result": return <ArtifactWebSearchResult query={result.query} content={result.content} webSearchQueries={result.webSearchQueries} sources={result.sources} grounded={result.grounded} />;
    case "search_combined": return <ArtifactSearchCombined query={result.query} local={result.local} local_ok={result.local_ok} web={result.web} web_ok={result.web_ok} />;
    case "spraypaint_identity_result": return <ArtifactSpraypaintIdentity fingerprint={result.fingerprint} chi={result.chi} floor={result.floor} vertices={result.vertices} edges={result.edges} />;
    case "spraypaint_count_result": return <ArtifactSpraypaintCount committed_count={result.committed_count} />;
    case "spraypaint_scenes_result": return <ArtifactSpraypaintScenes scenes={result.scenes} />;
    case "spraypaint_verify_result": return <SpraypaintVerify {...result} />;
    case "interceptor_generated": return <ArtifactInterceptorGenerated language={result.language} code={result.code} provider={result.provider} model={result.model} run_error={result.run_error} />;
    case "interceptor_run_result": return <ArtifactInterceptorRun language={result.language} code={result.code} ok={result.ok} stdout={result.stdout} stderr={result.stderr} exit_code={result.exit_code} elapsed_ms={result.elapsed_ms} timed_out={result.timed_out} truncated={result.truncated} vahera_memory={result.vahera_memory} />;
    case "windtunnel_report": return <ArtifactWindTunnelReport language={result.language} code={result.code} elapsed_ms={result.elapsed_ms} runs={result.runs} order_parameter={result.order_parameter} regime={result.regime} reference_index={result.reference_index} per_run={result.per_run} crash_count={result.crash_count} />;
    case "interceptor_assist_result": return <ArtifactInterceptorAssist language={result.language} task={result.task} code={result.code} provider={result.provider} model={result.model} run={result.run} vahera_memory={result.vahera_memory} windtunnel={result.windtunnel} />;
    // What the surface itself produces (settings, stacks, the runtime map,
    // player runs) renders in its own file; it receives this renderer back.
    default:                return <SurfaceArtifact result={result} Artifact={Artifact} />;
  }
}

// ────────────────────────────────────────────────────────────
//  Desk surface renderer. The blank surface, filled: the tagged reason,
//  its term-coverage, and the acts split into necessary (contributed toward
//  the intent) vs. purposeless (correct but did not advance g).
// ────────────────────────────────────────────────────────────

function deskSurfaceLines(result) {
  const lines = [];
  lines.push(`desk — the reason: ${result.reason}`);
  const cov = Math.round((result.coverage || 0) * 100);
  lines.push(
    `goal g = { ${result.goal_terms.join(", ")} }   covered ${cov}% (${result.covered_terms.length}/${result.goal_terms.length})`
  );
  lines.push("");

  if (result.necessary.length === 0 && result.purposeless.length === 0) {
    lines.push("no acts yet — dispatch some work, then surface again.");
    return lines;
  }

  lines.push(`necessary — advanced the reason (${result.necessary.length}):`);
  if (result.necessary.length === 0) {
    lines.push("  (none yet)");
  } else {
    for (const a of result.necessary) {
      const c = a.contribution.toFixed(2);
      lines.push(`  act ${a.act_id}  [${a.module_id}]  δS=${c}`);
    }
  }
  lines.push("");
  lines.push(`purposeless — correct but off the reason (${result.purposeless.length}):`);
  if (result.purposeless.length === 0) {
    lines.push("  (none)");
  } else {
    for (const a of result.purposeless) {
      lines.push(`  act ${a.act_id}  [${a.module_id}]  δS=0`);
    }
  }
  return lines;
}
