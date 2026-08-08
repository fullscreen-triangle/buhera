/* ============================================================================
 * Agent-DSL compiler (parse + check + run) — browser JS port.
 *
 * Web-side twin of smith-ide's `src/compiler/compiler.ts`. Parses the agent
 * DSL into an AST, runs the split-attention mathematics (χ, realised floor,
 * water-fill), and produces a deterministic tick trace. Pure and
 * self-contained: no fetch, no fs, no editor coupling — this is the no-install
 * webtool path, so the same server-side Rust tool can be offered as a download
 * without either being in the loop of the other.
 *
 * Faithfulness note: the .ts source seeds each scene's gain parameter with
 * `Math.random()`. That is replaced here with a deterministic per-scene value
 * (hashed from the scene name) so a given script always produces the same
 * check + run — matching the webtool's reproducibility contract.
 *
 * DSL shape:
 *   agent A {
 *     purpose minimise|reach TARGET;
 *     scene S serves TARGET with HOOK;
 *     self { parts { p, q, r }; separations (p,q: 3) (q,r: 2) }
 *     budget 4; floor 2;
 *     coherence keeps { p, q }
 *   }
 *   society Name { agent ... ; tie(a,b: 3); couple 1 }
 * ========================================================================== */

import {
  characterInvariant,
  realisedFloor,
  waterFill,
  logGainProfile,
} from "./math";

// --- deterministic gain parameter, replacing the .ts Math.random() seed -----
// A small FNV-1a hash of the scene name, squashed into [1, 3): stable across
// runs, still varied across scenes.
function gainForScene(name) {
  let h = 0x811c9dc5;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return 1 + (h % 2000) / 1000; // [1.000, 3.000)
}

// --- Parser (brace-balanced) ------------------------------------------------
//
// The .ts source matched declarations with a lazy `([\s\S]*?)\n\}` regex, which
// (a) required a newline before the closing brace — so single-line input never
// parsed — and (b) mis-terminated on the first inner `}` of a nested `self { …
// }` block. The webtool routes single-line `agent …` cells, so we scan by
// balanced braces instead: correct for both one-line and multi-line input, and
// for bodies with nested braces.

// Find `keyword Name { … }` blocks at top level, honouring brace nesting.
// Returns [{ keyword, name, body, start, end }].
function findBlocks(src) {
  const blocks = [];
  const header = /\b(society|agent)\s+(\w+)\s*\{/g;
  let m;
  while ((m = header.exec(src)) !== null) {
    const keyword = m[1];
    const name = m[2];
    const bodyStart = header.lastIndex; // char after the opening brace
    let depth = 1;
    let i = bodyStart;
    for (; i < src.length && depth > 0; i++) {
      const ch = src[i];
      if (ch === "{") depth++;
      else if (ch === "}") depth--;
    }
    if (depth !== 0) break; // unbalanced — stop; caller reports "nothing found"
    const body = src.slice(bodyStart, i - 1);
    blocks.push({ keyword, name, body, start: m.index, end: i });
    header.lastIndex = i; // continue scanning after this block
  }
  return blocks;
}

export function parse(source) {
  const errors = [];
  const items = [];

  const lines = source.split("\n");
  const cleaned = lines.map((l) => l.replace(/\/\/.*$/, "")).join("\n");

  for (const block of findBlocks(cleaned)) {
    if (block.keyword === "society") {
      const society = parseSociety(block.name, block.body, lines, errors);
      if (society) items.push(society);
    } else {
      const agent = parseAgent(block.name, block.body, lines, errors);
      if (agent) items.push(agent);
    }
  }

  if (items.length === 0 && errors.length === 0) {
    errors.push({ message: "No agent or society declarations found", severity: "error" });
  }

  return { file: { kind: "file", items } , errors };
}

function parseAgent(name, body, _lines, errors) {
  const purposeMatch = body.match(/purpose\s+(minimise|reach)\s+(\w+)/);
  if (!purposeMatch) {
    errors.push({ message: `Agent "${name}": missing purpose declaration`, severity: "error" });
    return null;
  }
  const purpose = { kind: "purpose", mode: purposeMatch[1], target: purposeMatch[2] };

  const scenes = [];
  const sceneRegex = /scene\s+(\w+)\s+serves\s+(\w+)\s+with\s+(\w+)/g;
  let sm;
  while ((sm = sceneRegex.exec(body)) !== null) {
    scenes.push({
      kind: "scene",
      name: sm[1],
      serves: sm[2],
      hook: sm[3],
      gainK: gainForScene(sm[1]),
    });
  }

  const partsMatch = body.match(/parts\s*\{([^}]+)\}/);
  const parts = partsMatch
    ? partsMatch[1].split(",").map((p) => p.trim()).filter(Boolean)
    : [];

  const separations = [];
  const sepRegex = /\((\w+),\s*(\w+):\s*([\d.]+)\)/g;
  let sepM;
  while ((sepM = sepRegex.exec(body)) !== null) {
    separations.push({ from: sepM[1], to: sepM[2], cost: parseFloat(sepM[3]) });
  }

  const budgetMatch = body.match(/budget\s+([\d.]+)/);
  const floorMatch = body.match(/floor\s+([\d.]+)/);
  const budget = budgetMatch ? parseFloat(budgetMatch[1]) : 1.0;
  const floor = floorMatch ? parseFloat(floorMatch[1]) : 2.0;

  const cohMatch = body.match(/coherence\s+keeps\s*\{([^}]+)\}/);
  const coherence = cohMatch
    ? { keeps: cohMatch[1].split(",").map((s) => s.trim()).filter(Boolean) }
    : undefined;

  return {
    kind: "agent",
    name,
    purpose,
    scenes,
    self: { kind: "self", parts, separations },
    budget,
    floor,
    coherence,
  };
}

function parseSociety(name, body, lines, errors) {
  const agents = [];
  for (const block of findBlocks(body)) {
    if (block.keyword !== "agent") continue;
    const agent = parseAgent(block.name, block.body, lines, errors);
    if (agent) agents.push(agent);
  }

  const ties = [];
  const tieRegex = /tie\s*\((\w+),\s*(\w+):\s*([\d.]+)\)/g;
  let tm;
  while ((tm = tieRegex.exec(body)) !== null) {
    ties.push({ from: tm[1], to: tm[2], cost: parseFloat(tm[3]) });
  }

  const coupleMatch = body.match(/couple\s+([\d.]+)/);
  const couple = coupleMatch ? parseFloat(coupleMatch[1]) : 1.0;

  return { kind: "society", name, agents, ties, couple };
}

// --- Checker ----------------------------------------------------------------

export function check(file) {
  const errors = [];
  const agents = [];

  for (const item of file.items) {
    if (item.kind === "agent") {
      const ac = checkAgent(item, errors);
      if (ac) agents.push(ac);
    } else if (item.kind === "society") {
      for (const agent of item.agents) {
        const ac = checkAgent(agent, errors);
        if (ac) agents.push(ac);
      }
      for (const tie of item.ties) {
        if (tie.cost < (item.agents[0]?.floor ?? 2)) {
          errors.push({
            message: `Tie (${tie.from}, ${tie.to}) cost ${tie.cost} is below the floor`,
            severity: "error",
          });
        }
      }
    }
  }

  return { ok: errors.filter((e) => e.severity === "error").length === 0, errors, agents };
}

function checkAgent(agent, errors) {
  const graph = agentToGraph(agent);

  for (const sep of agent.self.separations) {
    if (sep.cost < agent.floor) {
      errors.push({
        message: `Separation (${sep.from}, ${sep.to}) cost ${sep.cost} is below the floor ${agent.floor}`,
        severity: "error",
        line: sep.line,
      });
    }
  }

  for (const scene of agent.scenes) {
    if (scene.serves !== agent.purpose.target) {
      errors.push({
        message: `Scene "${scene.name}" serves "${scene.serves}" but the agent's purpose is "${agent.purpose.target}"`,
        severity: "error",
        line: scene.line,
      });
    }
  }

  const { chi, partition } = characterInvariant(graph);
  const floor = realisedFloor(graph);
  const nonLocal = partition.blocks.every((b) => b.length > 1);

  return {
    name: agent.name,
    regime: agent.purpose.mode === "minimise" ? "character" : "task",
    chi,
    floor,
    nonLocal,
    chiPartition: partition.blocks,
  };
}

// --- Runtime (deterministic tick loop) --------------------------------------

export function run(file, maxTicks = 30) {
  const steps = [];
  const counts = {};
  const residuals = {};

  const allAgents = [];
  for (const item of file.items) {
    if (item.kind === "agent") allAgents.push(item);
    else if (item.kind === "society") allAgents.push(...item.agents);
  }

  for (const agent of allAgents) {
    counts[agent.name] = 0;
    residuals[agent.name] = 10;
  }

  for (let tick = 1; tick <= maxTicks; tick++) {
    for (const agent of allAgents) {
      const phase = tick % 3 === 0 ? "construction" : "commitment";

      if (phase === "construction") {
        steps.push({
          tick,
          agent: agent.name,
          outcome: "observe",
          price: 0,
          residual: residuals[agent.name],
          delta: 0,
          count: counts[agent.name],
          phase,
        });
        continue;
      }

      const profiles = agent.scenes.map((s) => logGainProfile(s.name, s.gainK ?? 1.5));
      const wf = waterFill(profiles, agent.budget);

      const bestAlloc = wf.allocations.reduce(
        (best, a) => (a.allocation > best.allocation ? a : best),
        wf.allocations[0]
      );

      if (bestAlloc && bestAlloc.allocation > 0) {
        const delta = bestAlloc.allocation * 0.3 * Math.exp(-tick * 0.05);
        residuals[agent.name] = Math.max(0.01, residuals[agent.name] - delta);
        counts[agent.name]++;

        steps.push({
          tick,
          agent: agent.name,
          outcome: residuals[agent.name] <= 0.02 ? "quiescent" : "commit",
          scene: bestAlloc.scene,
          price: wf.price,
          residual: residuals[agent.name],
          delta,
          count: counts[agent.name],
          phase,
          disposition: [residuals[agent.name], wf.price],
        });
      } else {
        steps.push({
          tick,
          agent: agent.name,
          outcome: "decline",
          price: wf.price,
          residual: residuals[agent.name],
          delta: 0,
          count: counts[agent.name],
          phase,
        });
      }
    }
  }

  return { steps, finalCounts: counts };
}

// --- Helpers ----------------------------------------------------------------

export function agentToGraph(agent) {
  return {
    parts: agent.self.parts,
    separations: agent.self.separations.map((s) => ({
      from: s.from,
      to: s.to,
      cost: s.cost,
    })),
  };
}

export function compile(source) {
  const { file, errors: parseErrors } = parse(source);
  const checkResult = check(file);
  checkResult.errors = [...parseErrors, ...checkResult.errors];
  checkResult.ok = checkResult.errors.filter((e) => e.severity === "error").length === 0;
  return { file, check: checkResult };
}
