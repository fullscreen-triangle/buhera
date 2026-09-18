// =====================================================================
//  SpraypaintAllocationChart — one D3 renderer for a spraypaint `ask`
//  result's `allocation` array: how the water-filling budget actually
//  split across scenes, against how many passages each scene had
//  available. Two bars per scene (available = track, allocated = fill)
//  plus a hover tooltip with the exact numbers.
// =====================================================================
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const TRACK = "#3f3f46";
const FILL = "#58E6D9";
const TEXT = "#cbd5e1";
const DIM = "#9ca3af";

function styleAxis(g) {
  g.selectAll("path,line").attr("stroke", "#52525b");
  g.selectAll("text").attr("fill", DIM).attr("font-size", 10);
}

export default function SpraypaintAllocationChart({ allocation, width = 480 }) {
  const ref = useRef(null);
  const [tip, setTip] = useState(null);

  // Only scenes that actually had passages are worth a row — an empty
  // scene (available: 0) clutters the chart with a row that can never
  // receive budget, so it's dropped before drawing rather than shown as a
  // permanently-zero bar.
  const rows = (allocation || []).filter((a) => a.available > 0);
  const rowHeight = 26;
  const height = Math.max(60, rows.length * rowHeight + 30);

  useEffect(() => {
    if (!ref.current || rows.length === 0) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("width", "100%").attr("height", "100%");

    const m = { l: 110, r: 44, t: 8, b: 20 };
    const innerW = width - m.l - m.r;

    const y = d3
      .scaleBand()
      .domain(rows.map((r) => r.scene))
      .range([m.t, height - m.b])
      .padding(0.3);
    const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.available) || 1]).range([0, innerW]);

    svg.append("g").attr("transform", `translate(${m.l},${height - m.b})`).call(d3.axisBottom(x).ticks(4)).call(styleAxis);
    svg
      .append("g")
      .attr("transform", `translate(${m.l - 6},0)`)
      .call(d3.axisLeft(y).tickSize(0))
      .call(styleAxis)
      .selectAll("text")
      .attr("text-anchor", "end")
      .attr("fill", TEXT);

    const g = svg.append("g").attr("transform", `translate(${m.l},0)`);

    // Track: total passages available in the scene.
    g.selectAll("rect.track")
      .data(rows)
      .join("rect")
      .attr("class", "track")
      .attr("x", 0)
      .attr("y", (d) => y(d.scene))
      .attr("width", (d) => x(d.available))
      .attr("height", y.bandwidth())
      .attr("fill", TRACK)
      .attr("rx", 3);

    // Fill: passages the water-filling allocator actually committed.
    g.selectAll("rect.fill")
      .data(rows)
      .join("rect")
      .attr("class", "fill")
      .attr("x", 0)
      .attr("y", (d) => y(d.scene))
      .attr("width", (d) => x(d.allocated))
      .attr("height", y.bandwidth())
      .attr("fill", FILL)
      .attr("rx", 3)
      .style("cursor", "pointer")
      .on("mousemove", (event, d) => {
        const [mx, my] = d3.pointer(event, ref.current.parentNode);
        setTip({ x: mx, y: my, scene: d.scene, allocated: d.allocated, available: d.available });
      })
      .on("mouseleave", () => setTip(null));

    // Allocated-count label at the end of each fill bar.
    g.selectAll("text.count")
      .data(rows)
      .join("text")
      .attr("class", "count")
      .attr("x", (d) => x(d.allocated) + 4)
      .attr("y", (d) => y(d.scene) + y.bandwidth() / 2)
      .attr("dy", "0.35em")
      .attr("fill", d3.max(rows, (r) => r.allocated) > 0 ? TEXT : DIM)
      .attr("font-size", 10)
      .text((d) => `${d.allocated}/${d.available}`);
  }, [allocation, width, height, rows]);

  if (rows.length === 0) {
    return <p className="text-gray-500 text-sm">(no scenes with passages to allocate)</p>;
  }

  return (
    <div className="relative rounded-md border border-neutral-700 bg-[#151515] p-2">
      <div className="mb-1 px-1 text-[11px] uppercase tracking-wider text-neutral-400">
        budget allocation by scene (allocated / available)
      </div>
      <svg ref={ref} />
      {tip && (
        <div
          className="pointer-events-none absolute z-10 rounded border border-neutral-600 bg-black/90 px-2 py-1 text-xs text-gray-200"
          style={{ left: tip.x + 12, top: tip.y - 8 }}
        >
          <div className="font-mono text-teal-300">{tip.scene}</div>
          <div>{tip.allocated} allocated of {tip.available} available</div>
        </div>
      )}
    </div>
  );
}
