/* ============================================================================
 * Split-Attention Synchronised Agents — core mathematics.
 *
 * Browser JS port of smith-ide's `src/engine/math.ts`. This is the web-side
 * twin of the Rust agent-generation tool: the same definitions and theorems,
 * same numbers, running in-browser with no install. Only the functions the
 * agent compiler actually uses are ported here (floor, χ, water-fill, the log
 * gain profile); the Kuramoto / crowd-sharpening / economics helpers in the
 * IDE's math.ts are UI-chart-only and are intentionally omitted.
 *
 * Every function corresponds to a definition or theorem in the paper; the
 * reference tags below match the .ts source.
 * ========================================================================== */

// --- Boundary cost of a subset U: sum of weights crossing U and V\U. --------

export function boundaryCost(g, subset) {
  let cost = 0;
  for (const sep of g.separations) {
    const fromIn = subset.has(sep.from);
    const toIn = subset.has(sep.to);
    if (fromIn !== toIn) cost += sep.cost;
  }
  return cost;
}

// --- Realised floor (Theorem 2.1): min boundary cost over nonempty proper ---
//     subsets. Infinity when there is nothing to separate (n ≤ 1).

export function realisedFloor(g) {
  const n = g.parts.length;
  if (n <= 1) return Infinity;

  let minCost = Infinity;
  const total = 1 << n;
  for (let mask = 1; mask < total - 1; mask++) {
    const subset = new Set();
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) subset.add(g.parts[i]);
    }
    const cost = boundaryCost(g, subset);
    if (cost < minCost) minCost = cost;
  }
  return minCost;
}

// --- Partition residual ρ(Q): total cut weight between all pairs of blocks. --

export function partitionResidual(g, blocks) {
  const partMap = new Map();
  blocks.forEach((block, idx) => block.forEach((p) => partMap.set(p, idx)));

  let cost = 0;
  for (const sep of g.separations) {
    const bi = partMap.get(sep.from);
    const bj = partMap.get(sep.to);
    if (bi !== undefined && bj !== undefined && bi !== bj) {
      cost += sep.cost;
    }
  }
  return cost;
}

// --- Character invariant χ (Definition 4.2, Theorem 4.1): min ρ(Q) over all --
//     partitions with r ≥ 2 blocks. Returns { chi, partition }.

export function characterInvariant(g) {
  const n = g.parts.length;
  if (n <= 1) return { chi: Infinity, partition: { blocks: [g.parts], cost: Infinity } };

  let bestChi = Infinity;
  let bestPartition = { blocks: [], cost: Infinity };

  const rgs = new Array(n).fill(0);

  function enumerate(pos, maxSoFar) {
    if (pos === n) {
      const numBlocks = maxSoFar + 1;
      if (numBlocks < 2) return;

      const blocks = Array.from({ length: numBlocks }, () => []);
      for (let i = 0; i < n; i++) blocks[rgs[i]].push(g.parts[i]);
      if (blocks.some((b) => b.length === 0)) return;

      const cost = partitionResidual(g, blocks);
      if (cost < bestChi) {
        bestChi = cost;
        bestPartition = { blocks: blocks.map((b) => [...b]), cost };
      }
      return;
    }
    for (let val = 0; val <= maxSoFar + 1; val++) {
      rgs[pos] = val;
      enumerate(pos + 1, Math.max(maxSoFar, val));
    }
  }

  enumerate(0, -1);
  return { chi: bestChi, partition: bestPartition };
}

// --- Log gain profile: γ(a) = ln(1 + k·a), γ'(a) = k/(1+ka). ----------------

export function logGainProfile(name, k) {
  return {
    name,
    gamma: (a) => Math.log(1 + k * a),
    gammaPrime: (a) => k / (1 + k * a),
    gammaPrimeInverse: (p) => (p > 0 ? (k / p - 1) / k : Infinity),
    entryMargin: k,
  };
}

// --- Water-filling attention scheduler (Theorem 5.1, Algorithm 1). ----------
//     Bisection on the Lagrange multiplier p*.

export function waterFill(scenes, budget, tolerance = 1e-10) {
  if (scenes.length === 0) {
    return { allocations: [], price: 0, totalGain: 0, budgetUsed: 0 };
  }

  let pLo = 0;
  let pHi = Math.max(...scenes.map((s) => s.entryMargin));

  // Budget abundant: every scene can be served to the tolerance floor.
  const totalAtZeroPrice = scenes.reduce(
    (sum, s) => sum + s.gammaPrimeInverse(tolerance),
    0
  );
  if (totalAtZeroPrice <= budget) {
    const allocations = scenes.map((s) => ({
      scene: s.name,
      allocation: s.gammaPrimeInverse(tolerance),
      marginalGain: tolerance,
    }));
    const totalGain = scenes.reduce(
      (sum, s, i) => sum + s.gamma(allocations[i].allocation),
      0
    );
    const budgetUsed = allocations.reduce((sum, a) => sum + a.allocation, 0);
    return { allocations, price: 0, totalGain, budgetUsed };
  }

  let price = 0;
  for (let iter = 0; iter < 200; iter++) {
    price = (pLo + pHi) / 2;
    let totalAlloc = 0;
    for (const scene of scenes) {
      if (scene.entryMargin > price) {
        totalAlloc += scene.gammaPrimeInverse(price);
      }
    }
    if (totalAlloc > budget) pLo = price;
    else pHi = price;
    if (pHi - pLo < tolerance) break;
  }

  price = (pLo + pHi) / 2;
  const allocations = scenes.map((s) => {
    const alloc = s.entryMargin > price ? s.gammaPrimeInverse(price) : 0;
    return {
      scene: s.name,
      allocation: alloc,
      marginalGain: alloc > 0 ? s.gammaPrime(alloc) : s.entryMargin,
    };
  });

  const totalGain = scenes.reduce(
    (sum, s, i) => sum + s.gamma(allocations[i].allocation),
    0
  );
  const budgetUsed = allocations.reduce((sum, a) => sum + a.allocation, 0);
  return { allocations, price, totalGain, budgetUsed };
}
