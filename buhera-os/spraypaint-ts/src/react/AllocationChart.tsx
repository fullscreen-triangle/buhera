// ─────────────────────────────────────────────────────────────────────────────
// AllocationChart — the primary crossfilter surface.
//
// Draws the REAL water-filling outcome of one ask:
//   • top:    per-scene allocated-slot bars (blue = won slots above p*, grey = priced out)
//   • bottom: per-scene BM25 score box plots with the clearing-price p* as a dashed line
//
// It is EDITABLE, not read-only:
//   • click a scene bar  → invertSceneToggle → add/remove that scene from --scenes
//   • drag the p* line   → invertPriceDrag   → move -k to admit more / fewer passages
//
// Every gesture becomes a QueryDiff handed up via onGesture; the parent applies it
// to the AskQuery and re-runs. That is the chart→code propagation, grounded on the
// real binary's flags rather than a DSL string-replace.
// ─────────────────────────────────────────────────────────────────────────────

import React, { useRef, useEffect, useCallback } from "react";
import * as d3 from "d3";
import type { AskResult, SceneAllocation } from "../types.js";
import {
  invertSceneToggle,
  invertPriceDrag,
  type QueryDiff,
} from "../crossfilter.js";
import type { AskQuery } from "../types.js";

export interface AllocationChartProps {
  query: AskQuery;
  result: AskResult | null;
  stale: boolean;
  onGesture: (diff: QueryDiff) => void;
  height?: number;
}

interface SceneStat {
  scene: string;
  allocated: number;
  available: number;
  scores: number[];
  best: number;
  median: number;
}

/** Collapse an AskResult into per-scene stats the chart needs. */
function toSceneStats(result: AskResult): SceneStat[] {
  const byScene = new Map<string, number[]>();
  for (const hit of result.results) {
    const arr = byScene.get(hit.scene) ?? [];
    arr.push(hit.score);
    byScene.set(hit.scene, arr);
  }
  return result.allocation.map((a: SceneAllocation) => {
    const scores = (byScene.get(a.scene) ?? []).slice().sort((x, y) => y - x);
    const best = scores.length > 0 ? scores[0]! : 0;
    const median = scores.length > 0 ? scores[Math.floor(scores.length / 2)]! : 0;
    return {
      scene: a.scene,
      allocated: a.allocated,
      available: a.available,
      scores,
      best,
      median,
    };
  });
}

export function AllocationChart({
  query,
  result,
  stale,
  onGesture,
  height = 420,
}: AllocationChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const draw = useCallback(() => {
    const svgEl = svgRef.current;
    const container = containerRef.current;
    if (!svgEl || !container || !result) return;

    const width = container.clientWidth;
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("width", width).attr("height", height);

    // Only show scenes that could contribute (available > 0) — the rest are noise.
    const stats = toSceneStats(result).filter((s) => s.available > 0 || s.allocated > 0);
    if (stats.length === 0) return;

    const margin = { top: 20, right: 30, bottom: 60, left: 55 };
    const gap = 46;
    const halfH = (height - margin.top - margin.bottom - gap) / 2;
    const w = width - margin.left - margin.right;
    const price = result.price;

    const x = d3
      .scaleBand<string>()
      .domain(stats.map((s) => s.scene))
      .range([0, w])
      .padding(0.25);

    // ── Top: allocation bars ──────────────────────────────────────────────
    const g1 = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const maxAlloc = d3.max(stats, (s) => s.allocated) ?? 1;
    const y1 = d3.scaleLinear().domain([0, maxAlloc + 1]).range([halfH, 0]);

    g1.append("g")
      .attr("transform", `translate(0,${halfH})`)
      .call(d3.axisBottom(x))
      .call((g) => g.selectAll("text").attr("fill", "#cccccc").attr("font-size", "11px").attr("transform", "rotate(-15)").attr("text-anchor", "end"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444"));

    g1.append("g")
      .call(d3.axisLeft(y1).ticks(maxAlloc + 1).tickFormat(d3.format("d")))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444"));

    g1.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -halfH / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("allocated slots");

    const restricted = new Set(query.scenes ?? stats.map((s) => s.scene));

    for (const s of stats) {
      const barX = x(s.scene)!;
      const abovePrice = s.best >= price;
      const included = restricted.has(s.scene);
      const barH = Math.max(0, halfH - y1(s.allocated));

      g1.append("rect")
        .attr("x", barX)
        .attr("y", y1(s.allocated))
        .attr("width", x.bandwidth())
        .attr("height", barH)
        .attr("fill", abovePrice ? "#4fc1ff" : "#3c3c3c")
        .attr("opacity", stale ? 0.4 : included ? 0.9 : 0.3)
        .attr("stroke", included ? "none" : "#888")
        .attr("stroke-dasharray", included ? "0" : "3,2")
        .attr("rx", 2)
        .attr("cursor", "pointer")
        .on("click", () => {
          const diff = invertSceneToggle(query, result.allocation, s.scene);
          if (diff) onGesture(diff);
        })
        .append("title")
        .text(
          `${s.scene}\nallocated: ${s.allocated}\navailable: ${s.available}\nbest BM25: ${s.best.toFixed(3)}\nclearing price p*: ${price.toFixed(3)}\n${included ? "click to EXCLUDE" : "click to INCLUDE"}`,
        );

      if (s.allocated > 0) {
        g1.append("text")
          .attr("x", barX + x.bandwidth() / 2)
          .attr("y", y1(s.allocated) - 4)
          .attr("text-anchor", "middle")
          .attr("fill", "#cccccc")
          .attr("font-size", "11px")
          .attr("font-weight", "600")
          .text(s.allocated);
      }
    }

    g1.append("text")
      .attr("x", w - 4)
      .attr("y", 12)
      .attr("text-anchor", "end")
      .attr("fill", "#cca700")
      .attr("font-size", "10px")
      .attr("font-family", "monospace")
      .text(`p* = ${price.toFixed(3)}  ·  -k ${result.budget}`);

    // ── Bottom: BM25 box plots + draggable p* line ────────────────────────
    const g2 = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top + halfH + gap})`);

    const maxScore = d3.max(stats, (s) => s.best) ?? 2;
    const y2 = d3.scaleLinear().domain([0, maxScore * 1.1]).range([halfH, 0]);

    g2.append("g")
      .attr("transform", `translate(0,${halfH})`)
      .call(d3.axisBottom(x))
      .call((g) => g.selectAll("text").attr("fill", "#cccccc").attr("font-size", "10px").attr("transform", "rotate(-15)").attr("text-anchor", "end"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444"));

    g2.append("g")
      .call(d3.axisLeft(y2).ticks(4))
      .call((g) => g.selectAll("text").attr("fill", "#858585").attr("font-size", "10px"))
      .call((g) => g.selectAll("line,path").attr("stroke", "#444"));

    g2.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -halfH / 2)
      .attr("y", -40)
      .attr("text-anchor", "middle")
      .attr("fill", "#858585")
      .attr("font-size", "11px")
      .text("BM25 score");

    for (const s of stats) {
      const barX = x(s.scene)!;
      const bw = x.bandwidth();
      if (s.scores.length === 0) continue;
      g2.append("rect")
        .attr("x", barX + bw * 0.15)
        .attr("y", y2(s.best))
        .attr("width", bw * 0.7)
        .attr("height", Math.max(1, y2(s.median) - y2(s.best)))
        .attr("fill", s.best >= price ? "#4fc1ff33" : "#3c3c3c33")
        .attr("stroke", s.best >= price ? "#4fc1ff" : "#555")
        .attr("rx", 2);
      g2.append("line")
        .attr("x1", barX + bw * 0.15)
        .attr("x2", barX + bw * 0.85)
        .attr("y1", y2(s.median))
        .attr("y2", y2(s.median))
        .attr("stroke", "#cccccc")
        .attr("stroke-width", 2);
    }

    // draggable clearing-price line → invertPriceDrag → budget change
    const priceLine = g2
      .append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", y2(price))
      .attr("y2", y2(price))
      .attr("stroke", "#cca700")
      .attr("stroke-dasharray", "6,3")
      .attr("stroke-width", 1.5)
      .attr("cursor", "ns-resize");

    const hit = g2
      .append("line")
      .attr("x1", 0)
      .attr("x2", w)
      .attr("y1", y2(price))
      .attr("y2", y2(price))
      .attr("stroke", "transparent")
      .attr("stroke-width", 14)
      .attr("cursor", "ns-resize");

    const drag = d3
      .drag<SVGLineElement, unknown>()
      .on("drag", (event) => {
        const yPix = Math.max(0, Math.min(halfH, event.y));
        priceLine.attr("y1", yPix).attr("y2", yPix);
        hit.attr("y1", yPix).attr("y2", yPix);
      })
      .on("end", (event) => {
        const yPix = Math.max(0, Math.min(halfH, event.y));
        const draggedPrice = y2.invert(yPix);
        const diff = invertPriceDrag(query, result, draggedPrice);
        if (diff) onGesture(diff);
      });
    hit.call(drag);

    if (stale) {
      svg
        .append("rect")
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "#1e1e1e")
        .attr("opacity", 0.25)
        .attr("pointer-events", "none");
    }
  }, [query, result, stale, onGesture, height]);

  useEffect(() => {
    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  if (!result) {
    return (
      <div
        ref={containerRef}
        style={{ height, display: "flex", alignItems: "center", justifyContent: "center", color: "#858585", fontSize: 13 }}
      >
        Run a query to see the water-filling allocation
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <div style={{ padding: "4px 12px", fontSize: 11, color: "#858585", borderBottom: "1px solid #1e1e1e" }}>
        Water-filling allocation — click bars to toggle scenes · drag p* line to change budget
      </div>
      <svg ref={svgRef} />
    </div>
  );
}
