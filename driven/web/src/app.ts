// app.ts — the blank screen.

import "./style.css";
import { Kernel } from "./kernel";
import { embedMolecule } from "./substrate";
import { translate } from "./translator";
import { executeVahera } from "./vahera";
import { renderResult } from "./renderer";
import { COMPOUNDS } from "./compounds";

// ── boot ──────────────────────────────────────────────

const kernel = new Kernel(12);

for (const name of Object.keys(COMPOUNDS)) {
  const coord = embedMolecule(name, COMPOUNDS[name]);
  kernel.allocate(coord, COMPOUNDS[name], { name, formula: COMPOUNDS[name].formula });
}

// ── ui ────────────────────────────────────────────────

const screen = document.getElementById("screen")!;

const history = document.createElement("div");
history.className = "history";
screen.appendChild(history);

const inputLine = document.createElement("div");
inputLine.className = "input-line";

const prefix = document.createElement("span");
prefix.className = "cursor-prefix";
prefix.textContent = "";

const input = document.createElement("textarea");
input.id = "input";
input.rows = 1;
input.autofocus = true;
input.spellcheck = false;

inputLine.appendChild(prefix);
inputLine.appendChild(input);
screen.appendChild(inputLine);

input.focus();
document.addEventListener("click", () => input.focus());

let busy = false;

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
});

input.addEventListener("keydown", async (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (busy) return;
    const text = input.value.trim();
    if (!text) return;
    await handleObservation(text);
  }
});

async function handleObservation(text: string) {
  busy = true;

  const entry = document.createElement("div");
  entry.className = "entry";

  const obs = document.createElement("div");
  obs.className = "obs";
  obs.textContent = text;
  entry.appendChild(obs);

  history.appendChild(entry);
  input.value = "";
  input.style.height = "auto";
  scrollBottom();

  const thinking = document.createElement("span");
  thinking.className = "thinking";
  thinking.textContent = "...";
  entry.appendChild(thinking);
  scrollBottom();

  await delay(120 + Math.random() * 180);

  try {
    const vahera = translate(text);
    const result = executeVahera(vahera, kernel, COMPOUNDS);
    thinking.remove();
    if (result) {
      renderResult(entry, result);
    } else {
      const p = document.createElement("p");
      p.className = "ans";
      p.textContent = "no artifact.";
      entry.appendChild(p);
    }
  } catch (err) {
    thinking.remove();
    const p = document.createElement("p");
    p.className = "ans";
    p.textContent = `[${(err as Error).message}]`;
    entry.appendChild(p);
  }

  scrollBottom();
  busy = false;
  input.focus();
}

function scrollBottom() {
  history.scrollTop = history.scrollHeight;
  window.scrollTo(0, document.body.scrollHeight);
}

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
