/* SeparationPair — Theorem "shape does not determine admissibility",
 * illustrated as two contact graphs, identical up to the last edge weight
 * (identical shape / identical Capset(S)), differing floor, differing
 * admissibility verdict for the same question v0 -> x*.
 */
import React, { useRef, useEffect } from "react";

function getD3() {
  try { return require("d3"); } catch { return null; }
}

function buildGraph(peripheralWeight) {
  const nodes = [
    { id: "m", label: "m", isMedium: true },
    { id: "v0", label: "v0" },
    { id: "p", label: "p" },
    { id: "q", label: "q" },
    { id: "x", label: "x*" },
    { id: "z", label: "z" }, // the peripheral vertex whose weight differs
  ];
  const links = [
    { source: "m", target: "v0", weight: 4 },
    { source: "v0", target: "p", weight: 3 },
    { source: "p", target: "q", weight: 3 },
    { source: "q", target: "x", weight: 4 },
    { source: "m", target: "z", weight: peripheralWeight },
    { source: "z", target: "x", weight: peripheralWeight },
  ];
  return { nodes, links };
}

function Mini({ peripheralWeight, epsilon, label, tint }) {
  const ref = useRef(null);

  // floor = min edge weight in the graph (toy stand-in for min-cut, enough
  // to show the peripheral weight alone moves the global minimum).
  const floor = Math.min(4, 3, 3, 4, peripheralWeight, peripheralWeight);
  const admissible = floor <= epsilon;

  useEffect(() => {
    const d3 = getD3();
    if (!d3 || !ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();
    const width = 260, height = 210;
    const { nodes, links } = buildGraph(peripheralWeight);

    const sim = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance(48))
      .force("charge", d3.forceManyBody().strength(-160))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide(22));

    const g = svg.append("g");
    const link = g
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (l) => (l.weight === floor ? "#ff5555" : "#3a4a5c"))
      .attr("stroke-width", (l) => (l.weight === floor ? 3 : Math.max(1, l.weight)))
      .attr("stroke-dasharray", (l) => (l.weight === floor ? "4 2" : null));
    link.append("title").text((l) => `weight ${l.weight}`);

    const node = g.selectAll("g.node").data(nodes).join("g").attr("class", "node");
    node
      .append("circle")
      .attr("r", (d) => (d.isMedium ? 15 : d.id === "v0" || d.id === "x" ? 13 : 10))
      .attr("fill", (d) => (d.isMedium ? "#2e7d5b" : d.id === "v0" || d.id === "x" ? tint : "#1a2230"))
      .attr("stroke", (d) => (d.isMedium ? "#4fae86" : "#4a9eff"))
      .attr("stroke-width", 1.5);
    node
      .append("text")
      .text((d) => d.label)
      .attr("text-anchor", "middle")
      .attr("dy", 3.5)
      .style("fill", "#e5e7eb")
      .style("font-size", "9.5px")
      .style("font-family", "monospace")
      .style("pointer-events", "none");

    sim.on("tick", () => {
      link.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
    sim.tick(120);
    sim.stop();
    // static render (deterministic layout) — re-simulate on data change only
    sim.alpha(1).restart();
    return () => sim.stop();
  }, [peripheralWeight, floor, tint]);

  return (
    <div className="flex flex-col items-center">
      <div className="text-xs text-gray-400 mb-1 font-mono">{label}</div>
      <svg ref={ref} width={260} height={210} />
      <div className="text-[11px] text-gray-500 mt-1">floor β = {floor.toFixed(1)}</div>
      <div
        className={`mt-1 text-xs font-mono px-2 py-1 rounded ${
          admissible ? "bg-blue-950 text-blue-300 border border-blue-800" : "bg-red-950 text-red-300 border border-red-800"
        }`}
      >
        {admissible ? "admissible: v0 ⇝ x*" : "not admissible"}
      </div>
    </div>
  );
}

export default function SeparationPair() {
  const epsilon = 2.0;
  return (
    <div>
      <div className="flex flex-wrap justify-center gap-6">
        <Mini peripheralWeight={4.5} epsilon={epsilon} label="Source S₁ — same shape, same Capset(S)" tint="#1a3a5c" />
        <Mini peripheralWeight={1.5} epsilon={epsilon} label="Source S₂ — same shape, same Capset(S)" tint="#3a1a1a" />
      </div>
      <div className="mt-3 text-xs text-gray-400 text-center max-w-lg mx-auto">
        Every field on every record is identical between S₁ and S₂ — same
        classes, same slots, same declared capability set. Only the weight on
        the peripheral contact m↔z↔x* differs, and that alone flips whether
        the question &quot;is x* reachable from v0&quot; is admissible
        (ε = {epsilon.toFixed(1)}). No shape schema, however detailed, reads a
        single record and sees this — the floor is a property of the whole
        graph.
      </div>
    </div>
  );
}
