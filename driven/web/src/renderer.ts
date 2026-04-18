// renderer.ts — inline rendering of artifacts.

import type { ExecResult } from "./vahera";

const PROP_LABELS: Record<string, string> = {
  formula: "formula",
  molecular_weight: "mw",
  boiling_point_c: "bp (°C)",
  melting_point_c: "mp (°C)",
  density_g_cm3: "density (g/cm³)",
  n_atoms: "atoms",
};

export function renderResult(container: HTMLElement, res: ExecResult) {
  const wrap = document.createElement("div");
  wrap.className = "ans";

  switch (res.kind) {
    case "text": {
      if (res.lines) {
        for (const l of res.lines) {
          const p = document.createElement("p");
          p.textContent = l;
          wrap.appendChild(p);
        }
      }
      break;
    }
    case "scalar": {
      const span = document.createElement("span");
      span.className = "scalar";
      span.textContent = String(res.value);
      wrap.appendChild(span);
      break;
    }
    case "list": {
      if (res.title) {
        const h = document.createElement("p");
        h.textContent = res.title;
        wrap.appendChild(h);
      }
      if (res.items) {
        const ul = document.createElement("ul");
        for (const it of res.items) {
          const li = document.createElement("li");
          const d = it.distance.toFixed(3);
          li.textContent = `${it.name}    d=${d}`;
          ul.appendChild(li);
        }
        wrap.appendChild(ul);
      }
      break;
    }
    case "molecule": {
      if (res.compound) {
        const c = res.compound;
        const h = document.createElement("p");
        const strong = document.createElement("strong");
        strong.textContent = c.name;
        h.appendChild(strong);
        if (c.formula) h.appendChild(document.createTextNode("    " + c.formula));
        wrap.appendChild(h);

        const rows = document.createElement("div");
        for (const key of Object.keys(PROP_LABELS)) {
          if (key === "formula") continue;
          if (c.payload[key] === undefined) continue;
          const row = document.createElement("p");
          row.textContent = `${PROP_LABELS[key]}: ${c.payload[key]}`;
          rows.appendChild(row);
        }
        wrap.appendChild(rows);
      }
      break;
    }
  }

  container.appendChild(wrap);
}

// Minimal inline chart renderer (kept for future use).
export function renderChart(container: HTMLElement, values: number[], title = "") {
  const canvas = document.createElement("canvas");
  canvas.className = "inline";
  const w = Math.min(700, window.innerWidth - 200);
  const h = 200;
  canvas.width = w * 2;
  canvas.height = h * 2;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d")!;
  ctx.scale(2, 2);
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, w, h);

  const yMin = Math.min(...values);
  const yMax = Math.max(...values);
  const yRange = yMax - yMin || 1;
  const pad = { t: 20, r: 20, b: 25, l: 40 };
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;

  ctx.strokeStyle = "#2a9d8f";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = pad.l + (pw * i) / (values.length - 1);
    const y = pad.t + ph - ((v - yMin) / yRange) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  if (title) {
    ctx.fillStyle = "#777";
    ctx.font = "11px monospace";
    ctx.fillText(title, pad.l, 14);
  }
  container.appendChild(canvas);
}
