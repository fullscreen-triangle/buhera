/* LadderComposition — interactive pow(Ladder) = 1 - Π(1 - pow_i) chart.
 * Sliders control up to 4 rung powers; the d3 line chart shows cumulative
 * composite power after each rung, plus the repetition-saturation curve
 * 1-(1-pow)^n for comparison (Corollary "repetition saturates").
 */
import React, { useRef, useEffect, useState } from "react";

function getD3() {
  try { return require("d3"); } catch { return null; }
}

export default function LadderComposition() {
  const [powers, setPowers] = useState([0.3, 0.4, 0.25, 0.5]);
  const ref = useRef(null);

  const cumulative = powers.reduce(
    (acc, p) => {
      const prevComplement = 1 - acc[acc.length - 1];
      acc.push(1 - prevComplement * (1 - p));
      return acc;
    },
    [0]
  );

  useEffect(() => {
    const d3 = getD3();
    if (!d3 || !ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const margin = { top: 20, right: 20, bottom: 36, left: 42 };
    const width = 420 - margin.left - margin.right;
    const height = 240 - margin.top - margin.bottom;
    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, powers.length]).range([0, width]);
    const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

    g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(powers.length).tickFormat((d) => `rung ${d}`))
      .selectAll("text").style("fill", "#888").style("font-size", "9px");
    g.append("g")
      .call(d3.axisLeft(y).ticks(5))
      .selectAll("text").style("fill", "#888").style("font-size", "9px");
    g.selectAll(".domain, .tick line").style("stroke", "#333");

    // Saturation reference: repeated identical mean rung power.
    const meanP = powers.reduce((a, b) => a + b, 0) / powers.length;
    const satData = d3.range(0, powers.length + 0.01, 0.05).map((n) => ({
      n,
      v: 1 - Math.pow(1 - meanP, n),
    }));
    g.append("path")
      .datum(satData)
      .attr("fill", "none")
      .attr("stroke", "#555")
      .attr("stroke-dasharray", "3 3")
      .attr("stroke-width", 1.5)
      .attr("d", d3.line().x((d) => x(d.n)).y((d) => y(d.v)));

    // Actual composite power curve.
    const compData = cumulative.map((v, i) => ({ n: i, v }));
    g.append("path")
      .datum(compData)
      .attr("fill", "none")
      .attr("stroke", "#4fae86")
      .attr("stroke-width", 2.5)
      .attr("d", d3.line().x((d) => x(d.n)).y((d) => y(d.v)));

    g.selectAll("circle")
      .data(compData)
      .join("circle")
      .attr("cx", (d) => x(d.n))
      .attr("cy", (d) => y(d.v))
      .attr("r", 4)
      .attr("fill", "#4fae86")
      .append("title")
      .text((d) => `after rung ${d.n}: pow(Ladder) = ${d.v.toFixed(3)}`);

    g.append("text").attr("x", width - 4).attr("y", 12).attr("text-anchor", "end")
      .style("fill", "#4fae86").style("font-size", "9px").text("composite pow(Ladder)");
    g.append("text").attr("x", width - 4).attr("y", 24).attr("text-anchor", "end")
      .style("fill", "#777").style("font-size", "9px").text("mean-rung saturation curve");
  }, [powers, cumulative]);

  return (
    <div className="flex flex-col md:flex-row gap-6 items-start">
      <svg ref={ref} width={420} height={240} />
      <div className="w-full md:w-48 space-y-3">
        {powers.map((p, i) => (
          <label key={i} className="block text-xs text-gray-400">
            rung {i} power — {p.toFixed(2)}
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={p}
              onChange={(e) => {
                const next = [...powers];
                next[i] = Number(e.target.value);
                setPowers(next);
              }}
              className="w-full accent-emerald-500"
            />
          </label>
        ))}
        <div className="text-xs text-gray-500 font-mono pt-1">
          pow(Ladder) = {cumulative[cumulative.length - 1].toFixed(3)}
        </div>
      </div>
    </div>
  );
}
