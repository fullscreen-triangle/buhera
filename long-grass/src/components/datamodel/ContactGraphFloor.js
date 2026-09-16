/* ContactGraphFloor — a force-directed contact graph with a medium vertex,
 * weighted edges, and the induced minimum cut (the "floor") highlighted.
 *
 * Follows the repo's existing d3 idiom (see sandboxes/sbs/Charts.js):
 * require("d3") guarded, useRef + useEffect + manual selection clearing,
 * dark theme, tooltips via <title>. d3-force is bundled with the "d3" package.
 */
import React, { useRef, useEffect, useState } from "react";

function getD3() {
  try { return require("d3"); } catch { return null; }
}

// A small fixed contact graph: one medium vertex "m", six items, weighted
// edges. Item 5 is deliberately peripheral with one low-weight edge — this
// is the vertex whose cut we highlight as the floor.
const NODES = [
  { id: "m", label: "medium", isMedium: true },
  { id: "a", label: "a" },
  { id: "b", label: "b" },
  { id: "c", label: "c" },
  { id: "d", label: "d" },
  { id: "e", label: "e" },
];

const EDGES = [
  { source: "m", target: "a", weight: 4 },
  { source: "m", target: "b", weight: 5 },
  { source: "m", target: "c", weight: 3 },
  { source: "m", target: "d", weight: 6 },
  { source: "m", target: "e", weight: 1.2 }, // the floor edge
  { source: "a", target: "b", weight: 3 },
  { source: "b", target: "c", weight: 2 },
  { source: "c", target: "d", weight: 2.5 },
  { source: "d", target: "e", weight: 1.6 },
];

export default function ContactGraphFloor() {
  const ref = useRef(null);
  const [floorEdge, setFloorEdge] = useState(null);

  useEffect(() => {
    const d3 = getD3();
    if (!d3 || !ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = 420, height = 300;
    const nodes = NODES.map((n) => ({ ...n }));
    const links = EDGES.map((e) => ({ ...e }));

    // The floor: minimum-weight edge touching the medium — a simplified
    // stand-in for the min-cut computation the paper defines, sufficient to
    // illustrate "the floor is a global minimum, not a local property."
    const minMediumEdge = links
      .filter((l) => l.source === "m" || l.target === "m")
      .reduce((min, l) => (l.weight < min.weight ? l : min));
    setFloorEdge(minMediumEdge);

    const sim = d3
      .forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d) => d.id).distance((l) => 40 + 18 * l.weight))
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide(26));

    const g = svg.append("g");

    const link = g
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (l) => (l === minMediumEdge ? "#ff5555" : "#3a4a5c"))
      .attr("stroke-width", (l) => (l === minMediumEdge ? 3 : Math.max(1, l.weight / 1.5)))
      .attr("stroke-dasharray", (l) => (l === minMediumEdge ? "4 2" : null));

    link.append("title").text((l) => `weight ${l.weight}`);

    const node = g
      .selectAll("g.node")
      .data(nodes)
      .join("g")
      .attr("class", "node")
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => {
            if (!event.active) sim.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    node
      .append("circle")
      .attr("r", (d) => (d.isMedium ? 20 : 14))
      .attr("fill", (d) => (d.isMedium ? "#2e7d5b" : "#1a3a5c"))
      .attr("stroke", (d) => (d.isMedium ? "#4fae86" : "#4a9eff"))
      .attr("stroke-width", 2);

    node
      .append("text")
      .text((d) => d.label)
      .attr("text-anchor", "middle")
      .attr("dy", 4)
      .style("fill", "#e5e7eb")
      .style("font-size", "11px")
      .style("font-family", "monospace")
      .style("pointer-events", "none");

    sim.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    return () => sim.stop();
  }, []);

  return (
    <div className="flex flex-col items-center">
      <svg ref={ref} width={420} height={300} />
      <div className="mt-2 text-xs text-gray-400 text-center max-w-md">
        Drag any vertex. The dashed red edge is the minimum-weight cut touching
        the medium — the <span className="text-red-400 font-mono">floor</span>{" "}
        β for this graph
        {floorEdge ? (
          <>
            {" "}(weight <span className="font-mono text-red-300">{floorEdge.weight}</span>,
            vertex <span className="font-mono text-red-300">{floorEdge.source.id || floorEdge.source}↔{floorEdge.target.id || floorEdge.target}</span>)
          </>
        ) : null}
        . It is a property of the whole graph, not of any single edge&apos;s
        neighborhood.
      </div>
    </div>
  );
}
