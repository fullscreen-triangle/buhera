// substrate.js — S-entropy coordinates, Fisher metric, backward navigation.

export class SCoord {
  constructor(k, t, e) {
    for (const [name, v] of [["k", k], ["t", t], ["e", e]]) {
      if (v < -1e-9 || v > 1 + 1e-9) {
        throw new Error(`SCoord.${name}=${v} outside [0,1]`);
      }
    }
    this.k = k;
    this.t = t;
    this.e = e;
  }
  asTuple() { return [this.k, this.t, this.e]; }
}

const TEMPORAL = new Set([
  "when","before","after","during","now","then","yesterday","today",
  "recently","previously","last","next","current","old","new","past","future",
]);

const ACTIONS = new Set([
  "what","how","why","find","show","compute","predict","compare",
  "analyze","identify","measure","synthesize","determine","calculate","derive",
]);

export function embedText(content) {
  if (!content) return new SCoord(0, 0, 0);
  const text = content.trim().toLowerCase();
  const chars = [...text].filter((c) => !/\s/.test(c));
  const n = Math.max(chars.length, 1);

  const freq = {};
  for (const c of chars) freq[c] = (freq[c] || 0) + 1;
  let H = 0;
  for (const k in freq) {
    const p = freq[k] / n;
    if (p > 0) H -= p * Math.log2(p);
  }
  const sk = Math.min(H / Math.log2(26), 1);

  const words = text.replace(/[?!]/g, " ").split(/\s+/).filter(Boolean);
  const tHits = words.filter((w) => TEMPORAL.has(w)).length;
  const st = Math.min(tHits / Math.max(words.length * 0.3, 1), 1);

  const actionHits = words.filter((w) => ACTIONS.has(w)).length;
  const qMarks = (content.match(/\?/g) || []).length;
  const se = Math.min((actionHits + qMarks) / Math.max(words.length * 0.4, 1), 1);

  return new SCoord(sk, st, se);
}

function hashSeed(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export function embedProtein(name, props = {}) {
  const seed = hashSeed(name);
  let k = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
  let t = (Math.imul(seed, 1140671485) + 12820163) & 0x7fffffff;
  let e = (Math.imul(seed, 214013) + 2531011) & 0x7fffffff;

  let rk = k / 0x7fffffff;
  let rt = t / 0x7fffffff;
  let re = e / 0x7fffffff;

  // Mix in measurable properties so semantically close proteins live
  // closer in S-entropy space. This is a simplification of the full
  // residue-level embedding the kernel would compute.
  if (typeof props.length === "number") {
    rk = (rk + Math.log(props.length + 1) / 12) % 1;
  }
  if (props.role) {
    // pathway/role hash contribution
    const roleSeed = hashSeed(String(props.role));
    rt = (rt + (roleSeed / 0xffffffff)) % 1;
  }
  if (Array.isArray(props.domains)) {
    re = (re + Math.log(props.domains.length + 1) / 8) % 1;
  }

  return new SCoord(
    Math.abs(rk) % 1,
    Math.abs(rt) % 1,
    Math.abs(re) % 1,
  );
}

export function embedMolecule(name, props = {}) {
  const seed = hashSeed(name);
  let k = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff;
  let t = (Math.imul(seed, 1140671485) + 12820163) & 0x7fffffff;
  let e = (Math.imul(seed, 214013) + 2531011) & 0x7fffffff;

  let rk = k / 0x7fffffff;
  let rt = t / 0x7fffffff;
  let re = e / 0x7fffffff;

  if (typeof props.molecular_weight === "number") {
    rk = (rk + Math.log(props.molecular_weight + 1) / 10) % 1;
  }
  if (typeof props.boiling_point_c === "number") {
    rt = (rt + (props.boiling_point_c + 273) / 1000) % 1;
  }
  if (typeof props.n_atoms === "number") {
    re = (re + Math.log(props.n_atoms + 1) / 10) % 1;
  }

  return new SCoord(
    Math.abs(rk) % 1,
    Math.abs(rt) % 1,
    Math.abs(re) % 1,
  );
}

// ── Fisher metric ─────────────────────────────────────────

export function fisher1d(a, b, eps = 1e-6) {
  const A = Math.min(Math.max(a, eps), 1 - eps);
  const B = Math.min(Math.max(b, eps), 1 - eps);
  return Math.abs(Math.asin(2 * A - 1) - Math.asin(2 * B - 1));
}

export function sDistance(s1, s2) {
  const dk = fisher1d(s1.k, s2.k);
  const dt = fisher1d(s1.t, s2.t);
  const de = fisher1d(s1.e, s2.e);
  return Math.sqrt(dk * dk + dt * dt + de * de);
}

// ── Ternary addressing ───────────────────────────────────

export function ternaryAddress(s, depth) {
  const ranges = [[0, 1], [0, 1], [0, 1]];
  const vals = [s.k, s.t, s.e];
  const out = [];
  for (let d = 0; d < depth; d++) {
    const dim = d % 3;
    const [lo, hi] = ranges[dim];
    const third = (hi - lo) / 3;
    const v = vals[dim];
    let trit;
    if (v < lo + third) trit = 0;
    else if (v < lo + 2 * third) trit = 1;
    else trit = 2;
    out.push(String(trit));
    ranges[dim] = [lo + trit * third, lo + (trit + 1) * third];
  }
  return out.join("");
}

// ── Backward navigation ──────────────────────────────────

export function backwardNavigate(final, depth) {
  const initial = new SCoord(1, 0, 0);
  const path = [final];
  for (let j = depth; j > 0; j--) {
    const alpha = (depth - j + 1) / depth;
    const k = final.k + (initial.k - final.k) * alpha;
    const t = final.t + (initial.t - final.t) * alpha;
    const e = final.e + (initial.e - final.e) * alpha;
    path.push(new SCoord(
      Math.min(Math.max(k, 0), 1),
      Math.min(Math.max(t, 0), 1),
      Math.min(Math.max(e, 0), 1),
    ));
  }
  path.reverse();
  return { final, initial, path, miracleCount: depth, steps: depth };
}

export function completionMorphism(_penultimate, final) {
  return final;
}
