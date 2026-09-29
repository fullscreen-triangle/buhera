/* ============================================================================
 * VisBoard — draws the charts the screen decided on (lib/vis/decide.js).
 *
 * One crossfilter over the dataset; every chart that can filter owns one
 * dimension of it. A selection in any chart — a brushed range, a clicked
 * category, a brushed region — rescopes every other chart and the table:
 * each chart draws the whole dataset as context (grey) and the rows that
 * pass the other charts' selections in the accent colour.
 *
 * The page stays inert. The dataset and the chart choice are the snapshot;
 * a selection is transient view state, gone when the page is flipped away.
 * To keep a selection, "keep as a page" commits it as a new step.
 *
 * Design system (validated with the dataviz skill's palette checker on the
 * black surface, --mode dark --pairs all): accent #3987e5 for single-series
 * marks, categorical slots #3987e5 / #d95926 / #199e70 capped at three for
 * scatter colour, grey context, hairline axes, bars ≤ 24px with a rounded
 * data end, 2px lines, ≥ 8px dots with a 2px surface ring, a hover layer on
 * every mark, one filter row above, and a table view as every chart's twin.
 * ========================================================================== */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import crossfilter from "crossfilter2";
import { dispatch as dispatchModule } from "@/lib/modules/registry";
import { useSurfaceActions } from "@/components/surface/actions";

export const INK = {
  surface: "#000000",
  accent: "#3987e5",
  slots: ["#3987e5", "#d95926", "#199e70"],
  context: "#3d3c39",
  grid: "#1f1f1d",
  axis: "#383835",
  muted: "#898781",
  secondary: "#c3c2b7",
  primary: "#ffffff",
};

const TABLE_ROWS = 200;
const BINS = 20;
const HEIGHT = 170;
const M = { top: 8, right: 12, bottom: 26, left: 44 };
const EASE = "220ms ease";

const fmtInt = d3.format(",");
const fmtNum = (v) =>
  v == null || Number.isNaN(v) ? "—" : Number.isInteger(v) ? fmtInt(v) : d3.format(",.4~g")(v);
const fmtTime = d3.timeFormat("%Y-%m-%d %H:%M");

// ── geometry helpers ─────────────────────────────────────────────────────

// A column whose data end (top) is rounded and whose baseline is square.
// Always the same command structure, so CSS can transition `d` between states.
function columnPath(x, y, w, y0) {
  const h = Math.max(0, y0 - y);
  const r = Math.min(4, w / 2, h);
  return `M${x},${y0}V${y0 - h + r}Q${x},${y0 - h} ${x + r},${y0 - h}H${x + w - r}Q${x + w},${y0 - h} ${x + w},${y0 - h + r}V${y0}Z`;
}

// A bar growing right from x0, rounded at its data end.
function barPath(x0, x, y, h) {
  const w = Math.max(0, x - x0);
  const r = Math.min(4, h / 2, w);
  return `M${x0},${y}H${x0 + w - r}Q${x0 + w},${y} ${x0 + w},${y + r}V${y + h - r}Q${x0 + w},${y + h} ${x0 + w - r},${y + h}H${x0}Z`;
}

const pathStyle = (d, fill) => ({ d: `path("${d}")`, fill, transition: `d ${EASE}, fill ${EASE}` });

function useWidth() {
  const ref = useRef(null);
  const [w, setW] = useState(320);
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, Math.floor(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const num = (v) => (typeof v === "number" && isFinite(v) ? v : null);
const timeOf = (v) => (typeof v === "number" ? v : typeof v === "string" ? Date.parse(v) : null);

// ── crossfilter views: one per chart ─────────────────────────────────────
//
// A view owns the chart's dimension and group(s). Groups ignore their own
// dimension's filter, so each chart's counts reflect every OTHER selection.

function makeView(cf, chart, records) {
  const f = chart.fields;
  switch (chart.type) {
    case "histogram":
    case "timeline": {
      const isTime = chart.type === "timeline";
      const key = isTime ? f.time : f.field;
      const acc = isTime ? (r) => timeOf(r[key]) : (r) => num(r[key]);
      const vals = records.map(acc).filter((v) => v != null);
      const [lo, hi] = d3.extent(vals);
      if (isTime && f.value) {
        // value over time: a line; the dimension filters the time range.
        const dim = cf.dimension((r) => acc(r) ?? -Infinity);
        return { dim, isTime, acc, extent: [lo, hi], line: true };
      }
      const scale = (isTime ? d3.scaleTime() : d3.scaleLinear()).domain([lo, hi === lo ? lo + 1 : hi]).nice(BINS);
      const thresholds = scale.ticks(BINS).map(Number);
      const edges = [+scale.domain()[0], ...thresholds.filter((t) => t > +scale.domain()[0] && t < +scale.domain()[1]), +scale.domain()[1]];
      const binOf = (v) => Math.max(0, Math.min(edges.length - 2, d3.bisectRight(edges, v) - 1));
      const dim = cf.dimension((r) => acc(r) ?? -Infinity);
      const group = dim.group((v) => (v === -Infinity ? -1 : binOf(v)));
      const totals = new Array(edges.length - 1).fill(0);
      for (const v of vals) totals[binOf(v)]++;
      return { dim, group, edges, totals, isTime, acc };
    }
    case "series": {
      const dim = cf.dimension((r) => num(r[f.x]) ?? -Infinity);
      return { dim, line: true, acc: (r) => num(r[f.x]), extent: d3.extent(records, (r) => num(r[f.x])) };
    }
    case "bars":
    case "ranked": {
      const key = chart.type === "ranked" ? f.label : f.field;
      const raw = (r) => (r[key] == null || r[key] === "" ? "(none)" : String(r[key]));
      let order;
      let keyOf = raw;
      if (chart.type === "ranked") {
        order = [...records].sort((a, b) => (num(b[f.value]) ?? -Infinity) - (num(a[f.value]) ?? -Infinity)).map(raw);
      } else {
        const counts = d3.rollup(records, (v) => v.length, raw);
        order = f.ordinal
          ? [...counts.keys()].sort((a, b) => Number(a) - Number(b))
          : [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || a.localeCompare(b));
        if (order.length > f.top) {
          const keep = new Set(order.slice(0, f.top - 1));
          keyOf = (r) => (keep.has(raw(r)) ? raw(r) : "other");
          order = [...order.slice(0, f.top - 1), "other"];
        }
      }
      const dim = cf.dimension(keyOf);
      const group = dim.group();
      const totals = new Map();
      if (chart.type === "ranked") for (const r of records) totals.set(raw(r), num(r[f.value]) ?? 0);
      else for (const r of records) totals.set(keyOf(r), (totals.get(keyOf(r)) || 0) + 1);
      return { dim, group, order, totals, keyOf };
    }
    case "scatter": {
      const dim = cf.dimension((r) => r.__i);
      return { dim };
    }
    default:
      return { dim: null };
  }
}

// ── pieces ───────────────────────────────────────────────────────────────

function AxisX({ scale, y, width, time, ticks = 5 }) {
  const values = scale.ticks ? scale.ticks(ticks) : scale.domain();
  // Time ticks use d3's multi-scale format, so the label fits the span:
  // seconds within a minute, hours within a day, dates across days.
  const format = time ? scale.tickFormat(ticks) : scale.tickFormat ? scale.tickFormat(ticks, "~s") : String;
  return (
    <g transform={`translate(0,${y})`} fontSize="10" fill={INK.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
      <line x1={M.left} x2={width - M.right} stroke={INK.axis} />
      {values.map((v, i) => (
        <text key={i} x={scale(v)} y={16} textAnchor="middle">{format(v)}</text>
      ))}
    </g>
  );
}

function AxisY({ scale, width, ticks = 4 }) {
  const format = scale.tickFormat(ticks, "~s");
  return (
    <g fontSize="10" fill={INK.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
      {scale.ticks(ticks).map((v, i) => (
        <g key={i} transform={`translate(0,${scale(v)})`}>
          <line x1={M.left} x2={width - M.right} stroke={INK.grid} />
          <text x={M.left - 6} dy="0.32em" textAnchor="end">{format(v)}</text>
        </g>
      ))}
    </g>
  );
}

// Attach a d3 brush to a <g>; report the selection in data space. `reset`
// changing clears it (from the filter row's ×) without re-reporting.
function useBrush(gRef, { make, extent, invert, onChange, reset }) {
  const brushRef = useRef(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  useEffect(() => {
    if (!gRef.current) return;
    const brush = make().extent(extent).on("end", (ev) => {
      if (!ev.sourceEvent) return; // programmatic moves are not selections
      onChangeRef.current(ev.selection ? invert(ev.selection) : null);
    });
    brushRef.current = brush;
    const g = d3.select(gRef.current).call(brush);
    g.selectAll(".selection").attr("fill", INK.accent).attr("fill-opacity", 0.12).attr("stroke", INK.accent).attr("stroke-opacity", 0.5);
    g.selectAll(".overlay").attr("cursor", "crosshair");
    return () => g.on(".brush", null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(extent)]);
  useEffect(() => {
    if (reset && gRef.current && brushRef.current) d3.select(gRef.current).call(brushRef.current.move, null);
  }, [reset, gRef]);
}

function Histogram({ chart, view, width, active, tip, setFilter, reset }) {
  const { edges, totals, group, isTime } = view;
  const counts = new Map(group.all().map((g) => [g.key, g.value]));
  const x = (isTime ? d3.scaleTime() : d3.scaleLinear()).domain([edges[0], edges[edges.length - 1]]).range([M.left, width - M.right]);
  const y = d3.scaleLinear().domain([0, d3.max(totals) || 1]).nice().range([HEIGHT - M.bottom, M.top]);
  const [range, setRange] = useState(null);
  const gRef = useRef(null);
  useEffect(() => { if (reset) setRange(null); }, [reset]);
  const binLines = (i) => {
    const lo = isTime ? fmtTime(edges[i]) : fmtNum(edges[i]);
    const hi = isTime ? fmtTime(edges[i + 1]) : fmtNum(edges[i + 1]);
    const c = counts.get(i) || 0;
    return [{ value: fmtInt(c), label: active ? `of ${fmtInt(totals[i])} selected` : "rows" }, { value: `${lo} – ${hi}`, label: "" }];
  };
  const binAt = (ev) => {
    const [mx] = d3.pointer(ev, ev.currentTarget);
    const v = +x.invert(mx);
    const i = d3.bisectRight(edges, v) - 1;
    return i >= 0 && i < totals.length ? i : null;
  };
  useBrush(gRef, {
    make: d3.brushX,
    extent: [[M.left, M.top], [width - M.right, HEIGHT - M.bottom]],
    invert: ([a, b]) => [x.invert(a), x.invert(b)].map(Number),
    reset,
    onChange: (sel) => {
      setRange(sel);
      if (sel) view.dim.filterRange([sel[0], sel[1]]);
      else view.dim.filterAll();
      const label = sel ? `${chart.fields.field ?? chart.fields.time} ${isTime ? `${fmtTime(sel[0])} – ${fmtTime(sel[1])}` : `${fmtNum(sel[0])} – ${fmtNum(sel[1])}`}` : null;
      setFilter(label);
    },
  });
  return (
    <svg width={width} height={HEIGHT} role="img" aria-label={chart.title}>
      <AxisY scale={y} width={width} />
      {totals.map((t, i) => {
        const x0 = x(edges[i]);
        const band = x(edges[i + 1]) - x0;
        const w = Math.max(1, Math.min(24, band - 2));
        const bx = x0 + (band - w) / 2;
        const c = counts.get(i) || 0;
        const inRange = !range || (edges[i + 1] > range[0] && edges[i] < range[1]);
        // Focusable per bin for the keyboard; the pointer is read off the
        // brush layer above (binAt), which would otherwise swallow it.
        return (
          <g key={i}>
            <path d={columnPath(bx, y(t), w, y(0))} style={pathStyle(columnPath(bx, y(t), w, y(0)), INK.context)} />
            <path d={columnPath(bx, y(c), w, y(0))} style={pathStyle(columnPath(bx, y(c), w, y(0)), inRange ? INK.accent : INK.context)} />
            <rect x={x0} y={M.top} width={band} height={HEIGHT - M.top - M.bottom} fill="transparent" tabIndex={0} className="vis-hit"
              onFocus={(ev) => tip(ev, binLines(i))} onBlur={() => tip(null)} />
          </g>
        );
      })}
      <AxisX scale={x} y={HEIGHT - M.bottom} width={width} time={isTime} />
      <g ref={gRef} onPointerMove={(ev) => { const i = binAt(ev); i == null ? tip(null) : tip(ev, binLines(i)); }} onPointerLeave={() => tip(null)} />
    </svg>
  );
}

function Bars({ chart, view, width, active, tip, setFilter, reset }) {
  const { order, totals, group } = view;
  const ranked = chart.type === "ranked";
  const counts = new Map(group.all().map((g) => [g.key, g.value]));
  const [picked, setPicked] = useState(() => new Set());
  useEffect(() => { if (reset) setPicked(new Set()); }, [reset]);
  const rowH = 22;
  const labelW = Math.min(150, Math.max(60, width * 0.3));
  const height = order.length * rowH + 22;
  const max = d3.max([...totals.values()]) || 1;
  const x = d3.scaleLinear().domain([0, max]).nice().range([labelW + 8, width - M.right - 48]);

  function toggle(k) {
    const next = new Set(picked);
    next.has(k) ? next.delete(k) : next.add(k);
    setPicked(next);
    if (next.size) view.dim.filterFunction((v) => next.has(v));
    else view.dim.filterAll();
    const key = ranked ? chart.fields.label : chart.fields.field;
    setFilter(next.size ? `${key} ∈ {${[...next].join(", ")}}` : null);
  }

  return (
    <svg width={width} height={height} role="img" aria-label={chart.title}>
      {order.map((k, i) => {
        const yy = i * rowH + 4;
        const total = totals.get(k) || 0;
        // ranked: a row's value is shown when it passes the other selections;
        // bars: the count that passes them.
        const cur = ranked ? ((counts.get(k) || 0) > 0 ? total : 0) : counts.get(k) || 0;
        const on = picked.size === 0 || picked.has(k);
        const h = Math.min(14, rowH - 6);
        const show = (ev) => tip(ev, ranked
          ? [{ value: fmtNum(total), label: chart.fields.value }, { value: k, label: "" }]
          : [{ value: fmtInt(cur), label: active ? `of ${fmtInt(total)} selected` : "rows" }, { value: k, label: "" }]);
        return (
          <g key={k}>
            <text x={labelW} y={yy + h / 2} dy="0.32em" textAnchor="end" fontSize="11" fill={on ? INK.secondary : INK.muted}>
              {k.length > 22 ? `${k.slice(0, 21)}…` : k}
            </text>
            <path d={barPath(x(0), x(total), yy, h)} style={pathStyle(barPath(x(0), x(total), yy, h), INK.context)} />
            <path d={barPath(x(0), x(cur), yy, h)} style={pathStyle(barPath(x(0), x(cur), yy, h), on ? INK.accent : INK.context)} />
            <text x={x(total) + 6} y={yy + h / 2} dy="0.32em" fontSize="10" fill={INK.muted} style={{ fontVariantNumeric: "tabular-nums" }}>
              {ranked ? fmtNum(total) : fmtInt(total)}
            </text>
            <rect x={0} y={yy - 2} width={width} height={rowH} fill="transparent" tabIndex={0} className="vis-hit" style={{ cursor: "pointer" }}
              onClick={() => toggle(k)} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(k); } }}
              onPointerMove={show} onPointerLeave={() => tip(null)} onFocus={show} onBlur={() => tip(null)} />
          </g>
        );
      })}
    </svg>
  );
}

function Scatter({ chart, view, records, selected, width, tip, setFilter, reset }) {
  const f = chart.fields;
  const pts = records.filter((r) => num(r[f.x]) != null && num(r[f.y]) != null);
  const x = d3.scaleLinear().domain(d3.extent(pts, (r) => r[f.x])).nice().range([M.left, width - M.right]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (r) => r[f.y])).nice().range([HEIGHT - M.bottom, M.top]);
  const cats = f.categories ? f.categories.map(String) : null;
  const colorOf = (r) => (cats ? INK.slots[Math.max(0, cats.indexOf(String(r[f.color])))] : INK.accent);
  const delaunay = useMemo(() => d3.Delaunay.from(pts, (r) => x(r[f.x]), (r) => y(r[f.y])), [pts, x, y, f.x, f.y]);
  const gRef = useRef(null);
  useBrush(gRef, {
    make: d3.brush,
    extent: [[M.left, M.top], [width - M.right, HEIGHT - M.bottom]],
    invert: ([[a, b], [c, d]]) => [[x.invert(a), x.invert(c)], [y.invert(d), y.invert(b)]],
    reset,
    onChange: (sel) => {
      if (!sel) { view.dim.filterAll(); setFilter(null); return; }
      const [[x0, x1], [y0, y1]] = sel;
      const inside = new Set(pts.filter((r) => r[f.x] >= x0 && r[f.x] <= x1 && r[f.y] >= y0 && r[f.y] <= y1).map((r) => r.__i));
      view.dim.filterFunction((i) => inside.has(i));
      setFilter(`${f.x} ${fmtNum(x0)} – ${fmtNum(x1)}, ${f.y} ${fmtNum(y0)} – ${fmtNum(y1)}`);
    },
  });
  function hover(ev) {
    const [mx, my] = d3.pointer(ev, ev.currentTarget);
    const i = delaunay.find(mx, my);
    const r = pts[i];
    if (!r || Math.hypot(x(r[f.x]) - mx, y(r[f.y]) - my) > 24) return tip(null);
    tip(ev, [{ value: fmtNum(r[f.y]), label: f.y }, { value: fmtNum(r[f.x]), label: f.x },
      ...(f.color ? [{ value: String(r[f.color]), label: f.color }] : [])]);
  }
  return (
    <div>
      {cats && (
        <div className="flex gap-4 mb-1 text-[10px]" style={{ color: INK.secondary }}>
          {cats.map((c, i) => (
            <span key={c} className="flex items-center gap-1">
              <svg width="8" height="8"><circle cx="4" cy="4" r="4" fill={INK.slots[i]} /></svg>{c}
            </span>
          ))}
        </div>
      )}
      <svg width={width} height={HEIGHT} role="img" aria-label={chart.title}>
        <AxisY scale={y} width={width} />
        {pts.map((r) => {
          const on = selected.has(r);
          return (
            <circle key={r.__i} cx={x(r[f.x])} cy={y(r[f.y])} r={4} stroke={INK.surface} strokeWidth={2}
              style={{ fill: on ? colorOf(r) : INK.context, transition: `fill ${EASE}` }} />
          );
        })}
        <AxisX scale={x} y={HEIGHT - M.bottom} width={width} />
        <g ref={gRef} onPointerMove={hover} onPointerLeave={() => tip(null)} />
      </svg>
    </div>
  );
}

function LineChart({ chart, view, records, selected, active, width, tip, setFilter, reset }) {
  const f = chart.fields;
  const isTime = chart.type === "timeline";
  const xKey = isTime ? f.time : f.x;
  const yKey = isTime ? f.value : f.y;
  const xAcc = isTime ? (r) => timeOf(r[xKey]) : (r) => num(r[xKey]);
  const pts = records.filter((r) => xAcc(r) != null && num(r[yKey]) != null).sort((a, b) => xAcc(a) - xAcc(b));
  const x = (isTime ? d3.scaleTime() : d3.scaleLinear()).domain(d3.extent(pts, xAcc)).range([M.left, width - M.right]);
  const y = d3.scaleLinear().domain(d3.extent(pts, (r) => r[yKey])).nice().range([HEIGHT - M.bottom, M.top]);
  const line = d3.line().x((r) => x(xAcc(r))).y((r) => y(r[yKey])).curve(d3.curveMonotoneX);
  const [hoverX, setHoverX] = useState(null);
  const gRef = useRef(null);
  useBrush(gRef, {
    make: d3.brushX,
    extent: [[M.left, M.top], [width - M.right, HEIGHT - M.bottom]],
    invert: ([a, b]) => [x.invert(a), x.invert(b)].map(Number),
    reset,
    onChange: (sel) => {
      if (sel) view.dim.filterRange(sel); else view.dim.filterAll();
      setFilter(sel ? `${xKey} ${isTime ? `${fmtTime(sel[0])} – ${fmtTime(sel[1])}` : `${fmtNum(sel[0])} – ${fmtNum(sel[1])}`}` : null);
    },
  });
  function hover(ev) {
    const [mx] = d3.pointer(ev, ev.currentTarget);
    const xv = +x.invert(mx);
    const i = Math.min(pts.length - 1, Math.max(0, d3.bisector(xAcc).center(pts, xv)));
    const r = pts[i];
    if (!r) return;
    setHoverX(x(xAcc(r)));
    tip(ev, [{ value: fmtNum(r[yKey]), label: yKey }, { value: isTime ? fmtTime(xAcc(r)) : fmtNum(xAcc(r)), label: xKey }]);
  }
  const dots = active && pts.length <= 400 ? pts.filter((r) => selected.has(r)) : [];
  return (
    <svg width={width} height={HEIGHT} role="img" aria-label={chart.title}>
      <AxisY scale={y} width={width} />
      <path d={line(pts)} fill="none" stroke={active ? INK.context : INK.accent} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" style={{ transition: `stroke ${EASE}` }} />
      {dots.map((r) => <circle key={r.__i} cx={x(xAcc(r))} cy={y(r[yKey])} r={4} fill={INK.accent} stroke={INK.surface} strokeWidth={2} />)}
      {hoverX != null && <line x1={hoverX} x2={hoverX} y1={M.top} y2={HEIGHT - M.bottom} stroke={INK.muted} strokeWidth={1} />}
      <AxisX scale={x} y={HEIGHT - M.bottom} width={width} time={isTime} />
      <g ref={gRef} onPointerMove={hover} onPointerLeave={() => { setHoverX(null); tip(null); }} />
    </svg>
  );
}

function Tiles({ chart, records }) {
  const r = records[0] || {};
  return (
    <div className="flex flex-wrap gap-x-10 gap-y-4 py-2">
      {chart.fields.values.map((k) => (
        <div key={k}>
          <div className="text-[11px]" style={{ color: INK.muted }}>{k}</div>
          <div className="text-2xl" style={{ color: INK.primary }}>{fmtNum(r[k])}</div>
        </div>
      ))}
    </div>
  );
}

// ── one chart card ───────────────────────────────────────────────────────

function ChartCard({ chart, view, records, selected, active, tip, setFilter, reset }) {
  const [ref, width] = useWidth();
  const props = { chart, view, records, selected, active, width, tip, setFilter, reset };
  let body = null;
  if (chart.type === "histogram" || (chart.type === "timeline" && !view.line)) body = <Histogram {...props} />;
  else if (chart.type === "timeline" || chart.type === "series") body = <LineChart {...props} />;
  else if (chart.type === "bars" || chart.type === "ranked") body = <Bars {...props} />;
  else if (chart.type === "scatter") body = <Scatter {...props} />;
  else if (chart.type === "tiles") body = <Tiles chart={chart} records={records} />;
  return (
    <figure className="min-w-0 m-0">
      <figcaption className="mb-2">
        <div className="text-sm" style={{ color: INK.primary }}>{chart.title}</div>
        <div className="text-[11px] leading-snug" style={{ color: INK.muted }}>{chart.reason}</div>
      </figcaption>
      <div ref={ref} className="w-full">{body}</div>
    </figure>
  );
}

// ── the board ────────────────────────────────────────────────────────────

export default function VisBoard({ title, dataset, charts, notes }) {
  const actions = useSurfaceActions();
  const records = useMemo(() => (dataset?.rows || []).map((r, i) => ({ ...r, __i: i })), [dataset]);
  const cf = useMemo(() => crossfilter(records), [records]);
  // No dispose on cleanup: dimensions belong to this crossfilter and go with
  // it. (Disposing in an effect cleanup also breaks under StrictMode, whose
  // simulated unmount would kill every group while the memo keeps them.)
  const views = useMemo(() => Object.fromEntries((charts || []).map((c) => [c.id, makeView(cf, c, records)])), [cf, charts, records]);

  const [filters, setFilters] = useState({});   // chart id → label
  const [resets, setResets] = useState({});     // chart id → token
  const [, setTick] = useState(0);
  const selected = new Set(cf.allFiltered());
  const active = Object.keys(filters).length > 0;

  const setFilterFor = useCallback((id) => (label) => {
    setFilters((fs) => {
      const next = { ...fs };
      if (label) next[id] = label; else delete next[id];
      return next;
    });
    setTick((t) => t + 1);
  }, []);

  function clear(id) {
    const ids = id ? [id] : Object.keys(filters);
    for (const k of ids) views[k]?.dim?.filterAll();
    setFilters((fs) => { const next = { ...fs }; for (const k of ids) delete next[k]; return next; });
    setResets((rs) => { const next = { ...rs }; for (const k of ids) next[k] = (next[k] || 0) + 1; return next; });
  }

  // ── tooltip ──
  const boardRef = useRef(null);
  const [tipState, setTipState] = useState(null);
  const tip = useCallback((ev, lines) => {
    if (!ev || !lines || !boardRef.current) return setTipState(null);
    const box = boardRef.current.getBoundingClientRect();
    const px = ev.clientX ?? (ev.target.getBoundingClientRect().left + 10);
    const py = ev.clientY ?? ev.target.getBoundingClientRect().top;
    setTipState({ x: px - box.left + 14, y: py - box.top + 14, lines });
  }, []);

  function keep() {
    const rows = [...selected].map(({ __i, ...r }) => r);
    const label = `keep ${fmtInt(rows.length)} rows of ${title}: ${Object.values(filters).join("; ")}`;
    actions.derive(label, async () => {
      const res = await dispatchModule("vis", { data: rows, title: `${title} — selection` });
      return res?.output_delta ? { kind: "artifact", result: res.output_delta } : { kind: "text", lines: ["(vis: no output)"] };
    });
  }

  const fields = (dataset?.fields || []).filter((f) => f.type !== "empty");
  const tableRows = [...selected].sort((a, b) => a.__i - b.__i).slice(0, TABLE_ROWS);

  return (
    <div ref={boardRef} className="relative" onPointerLeave={() => setTipState(null)}>
      <div className="mb-1 text-white">{title}</div>
      <div className="mb-4 text-xs" style={{ color: INK.muted }}>
        {fmtInt(records.length)} rows from {dataset.path}
        {dataset.total > records.length ? ` (of ${fmtInt(dataset.total)})` : ""} · {charts.length} chart{charts.length === 1 ? "" : "s"} chosen for it
      </div>

      {/* the one filter row: scopes every chart and the table below */}
      <div className="mb-5 flex flex-wrap items-center gap-2 min-h-[1.5rem] text-xs">
        <span style={{ color: INK.secondary, fontVariantNumeric: "tabular-nums" }}>
          {fmtInt(selected.size)} of {fmtInt(records.length)} rows
        </span>
        {Object.entries(filters).map(([id, label]) => (
          <button key={id} type="button" onClick={() => clear(id)}
            className="px-2 py-0.5 rounded border border-gray-800 hover:border-gray-600" style={{ color: INK.secondary }}>
            {label} <span style={{ color: INK.muted }}>×</span>
          </button>
        ))}
        {active && (
          <>
            <button type="button" onClick={() => clear(null)} className="underline" style={{ color: INK.muted }}>clear</button>
            {actions && (
              <button type="button" onClick={keep} className="ml-2 hover:text-teal-300" style={{ color: INK.secondary }}>
                keep as a page →
              </button>
            )}
          </>
        )}
        {!active && charts.some((c) => c.type !== "tiles") && (
          <span style={{ color: INK.muted }}>— brush a range or pick a bar to select; every chart follows</span>
        )}
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-1 gap-x-10 gap-y-8">
        {charts.map((c) => (
          <ChartCard key={c.id} chart={c} view={views[c.id]} records={records} selected={selected} active={active}
            tip={tip} setFilter={setFilterFor(c.id)} reset={resets[c.id] || 0} />
        ))}
      </div>

      {notes?.length > 0 && (
        <ul className="mt-6 text-[11px] space-y-0.5" style={{ color: INK.muted }}>
          {notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}

      {/* the table twin: every value the charts show, reachable without hovering */}
      <div className="mt-8">
        <div className="text-xs mb-2" style={{ color: INK.muted }}>
          table — {fmtInt(selected.size)} row{selected.size === 1 ? "" : "s"}
          {selected.size > TABLE_ROWS ? `, first ${TABLE_ROWS} shown (keep as a page to carry them all)` : ""}
        </div>
        <div className="overflow-x-auto no-scrollbar">
          <table className="text-[11px] border-collapse" style={{ fontVariantNumeric: "tabular-nums" }}>
            <thead>
              <tr>
                {fields.map((f) => (
                  <th key={f.name} className="text-left font-normal pr-6 pb-1 whitespace-nowrap" style={{ color: INK.muted }}>{f.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((r) => (
                <tr key={r.__i} className="border-t" style={{ borderColor: INK.grid }}>
                  {fields.map((f) => {
                    const v = r[f.name];
                    const text = typeof v === "number" ? (f.type === "temporal" ? fmtTime(v) : fmtNum(v)) : v == null ? "—" : String(v);
                    return (
                      <td key={f.name} className={`pr-6 py-0.5 whitespace-nowrap ${typeof v === "number" ? "text-right" : ""}`} style={{ color: INK.secondary }}>
                        {text.length > 60 ? `${text.slice(0, 59)}…` : text}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* keyboard focus on a mark: a hairline ring, not the browser's box;
          a mouse click shows no ring at all */}
      <style jsx global>{`
        .vis-hit:focus { outline: none; }
        .vis-hit:focus-visible { stroke: ${INK.muted}; stroke-width: 1; }
      `}</style>

      {tipState && (
        <div className="absolute z-20 pointer-events-none px-2 py-1.5 rounded border text-xs"
          style={{ left: tipState.x, top: tipState.y, background: "#0d0d0d", borderColor: "rgba(255,255,255,0.10)" }}>
          {tipState.lines.map((l, i) => (
            <div key={i} className="whitespace-nowrap">
              <span style={{ color: i === 0 ? INK.primary : INK.secondary, fontWeight: i === 0 ? 600 : 400 }}>{l.value}</span>
              {l.label && <span style={{ color: INK.muted }}> {l.label}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
