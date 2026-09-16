/* CompositionTree — d3.hierarchy tree layout of the composed data layer:
 * Data Layer = (Shape source, Contact graph). Shape branches into whichever
 * shape language is adopted (LinkML / JSON Schema / SHACL / OWL — all
 * instances of Capset(S) ⊆ Feat by Prop. "LinkML schemas are capability
 * declarations" and its Corollary "family"); the floor branch produces
 * admissibility verdicts the shape branch structurally cannot.
 */
import React, { useRef, useEffect } from "react";

function getD3() {
  try { return require("d3"); } catch { return null; }
}

const DATA = {
  name: "Composed data layer\n(S, Γ_S)",
  kind: "root",
  children: [
    {
      name: "Shape source S\nCapset(S) ⊆ Feat",
      kind: "shape",
      children: [
        { name: "LinkML\nclasses / slots / enums", kind: "leaf" },
        { name: "JSON Schema\nproperties", kind: "leaf" },
        { name: "SHACL\nshapes", kind: "leaf" },
        { name: "OWL\nclass restrictions", kind: "leaf" },
      ],
    },
    {
      name: "Contact graph Γ_S\nfloor β, residue",
      kind: "floor",
      children: [
        { name: "role / direction\n(computed)", kind: "leaf2" },
        { name: "admissibility\n(certified)", kind: "leaf2" },
        { name: "verdict algebra\n(6-valued)", kind: "leaf2" },
      ],
    },
  ],
};

const COLOR = {
  root: "#4fae86",
  shape: "#4a9eff",
  floor: "#ff8a4a",
  leaf: "#2a4a6c",
  leaf2: "#6c3a1a",
};

export default function CompositionTree() {
  const ref = useRef(null);

  useEffect(() => {
    const d3 = getD3();
    if (!d3 || !ref.current) return;
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    const width = 640, height = 380;
    const root = d3.hierarchy(DATA);
    const treeLayout = d3.tree().size([width - 120, height - 100]);
    treeLayout(root);

    const g = svg.append("g").attr("transform", "translate(60,50)");

    g.selectAll("path.link")
      .data(root.links())
      .join("path")
      .attr("class", "link")
      .attr("fill", "none")
      .attr("stroke", "#3a4a5c")
      .attr("stroke-width", 1.5)
      .attr(
        "d",
        d3
          .linkVertical()
          .x((d) => d.x)
          .y((d) => d.y)
      );

    const node = g
      .selectAll("g.node")
      .data(root.descendants())
      .join("g")
      .attr("class", "node")
      .attr("transform", (d) => `translate(${d.x},${d.y})`);

    node
      .append("rect")
      .attr("x", -62)
      .attr("y", -16)
      .attr("width", 124)
      .attr("height", 34)
      .attr("rx", 6)
      .attr("fill", "#0d1520")
      .attr("stroke", (d) => COLOR[d.data.kind] || "#3a4a5c")
      .attr("stroke-width", 1.5);

    node
      .selectAll("text")
      .data((d) => d.data.name.split("\n").map((line, i) => ({ line, i })))
      .join("text")
      .attr("text-anchor", "middle")
      .attr("y", (t) => -2 + t.i * 12)
      .style("fill", (t, i, nodesArr) => "#e5e7eb")
      .style("font-size", "9.5px")
      .style("font-family", "monospace")
      .text((t) => t.line);
  }, []);

  return (
    <div className="overflow-x-auto">
      <svg ref={ref} width={640} height={380} />
    </div>
  );
}
