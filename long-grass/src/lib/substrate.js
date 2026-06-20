// substrate.js — S-entropy coordinates, Fisher metric, ternary
// addresses, backward navigation, and a deterministic lexical
// text embedding.
//
// Mirrors the Rust `buhera-substrate` crate so the web demo and the
// desktop OS have identical semantics. No external dependencies.

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
  toString() {
    return `S(${this.k.toFixed(3)},${this.t.toFixed(3)},${this.e.toFixed(3)})`;
  }
}

// ─────────────────────────────────────────────────────────────────────
//  Lexical embedding (token-bag projection).
// ─────────────────────────────────────────────────────────────────────

const TEMPORAL = new Set([
  "when", "before", "after", "during", "now", "then", "today",
  "yesterday", "tomorrow", "recently", "previously", "last",
  "next", "current", "old", "new", "past", "future", "morning",
  "afternoon", "evening", "night", "week", "month", "year",
  "weekend", "schedule", "deadline", "soon", "later", "early", "late",
]);

const ACTIONS = new Set([
  "what", "how", "why", "find", "show", "compute", "predict",
  "compare", "analyze", "identify", "measure", "synthesize",
  "determine", "calculate", "derive", "buy", "do", "make", "go",
  "run", "read", "write", "build", "refactor", "fix", "send",
  "reply", "book", "schedule", "remember", "update", "create",
  "store", "search", "look", "get", "want", "need", "should", "must",
]);

const STOPWORDS = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "been",
  "being", "have", "has", "had", "do", "does", "did", "of", "in",
  "on", "at", "to", "from", "for", "with", "by", "and", "or",
  "but", "if", "then", "this", "that", "these", "those", "it",
  "its", "i", "you", "he", "she", "we", "they", "me", "him",
  "her", "us", "them", "my", "your", "his", "their", "our",
  "as", "so", "not", "no", "yes", "up", "down", "out", "off",
  "over", "under", "again", "just", "very", "much", "some",
  "any", "all", "more", "most", "less", "least", "than",
  "about", "into", "onto", "upon",
]);

function tokenize(text) {
  return text.split(/[^\p{L}\p{N}]+/u).filter(Boolean);
}

function fnv1a64(str) {
  // 64-bit FNV-1a using two 32-bit halves so we don't lose bits on
  // JavaScript's float-backed numbers.
  let hi = 0xcbf2_9ce4 >>> 0;
  let lo = 0x8422_2325 >>> 0;
  const FNV_PRIME_HI = 0x0000_0100;
  const FNV_PRIME_LO = 0x0000_01b3;

  for (let i = 0; i < str.length; i++) {
    lo = (lo ^ str.charCodeAt(i)) >>> 0;
    // 64-bit multiply: (hi:lo) * (FNV_PRIME_HI:FNV_PRIME_LO)
    const a0 = lo & 0xffff;
    const a1 = lo >>> 16;
    const a2 = hi & 0xffff;
    const a3 = hi >>> 16;

    const b0 = FNV_PRIME_LO & 0xffff;
    const b1 = FNV_PRIME_LO >>> 16;
    const b2 = FNV_PRIME_HI & 0xffff;
    const b3 = FNV_PRIME_HI >>> 16;

    const c0 = a0 * b0;
    const c1 = (a1 * b0) + (a0 * b1);
    const c2 = (a2 * b0) + (a1 * b1) + (a0 * b2);
    const c3 = (a3 * b0) + (a2 * b1) + (a1 * b2) + (a0 * b3);

    const newLo = ((c0 & 0xffff) + ((c1 & 0xffff) << 16)) >>> 0;
    const newHi = (
      (c1 >>> 16) +
      (c2 & 0xffffffff) +
      ((c3 & 0xffff) << 16)
    ) >>> 0;
    lo = newLo;
    hi = newHi;
  }
  return { hi, lo };
}

function chunk21(value) {
  // Map a 21-bit field to [0, 1].
  return (value & 0x1fffff) / 0x1fffff;
}

function tokenAxes(token) {
  const { hi, lo } = fnv1a64(token);
  // Split 64 bits into 21+21+22 chunks (top chunk truncated to 21).
  const a = chunk21(lo);
  const b = chunk21(((lo >>> 21) | (hi << 11)) >>> 0);
  const c = chunk21((hi >>> 10) >>> 0);
  return [a, b, c];
}

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function squash(x) {
  // Soft sigmoid centered at 0.5 to spread per-token contributions
  // away from the boundary.
  const c = clamp01(x);
  return 0.5 + 0.5 * Math.tanh((c - 0.5) * 2.5);
}

/**
 * Embed text content into the S-entropy unit cube.
 *
 * Algorithm:
 *   1. Tokenise on non-alphanumerics, lowercase.
 *   2. For each token, hash to three axes (FNV-1a → 21-bit chunks).
 *   3. Weight stopwords down (0.05), content words full (1.0).
 *   4. Add a light bias to `S_t` for temporal markers and `S_e` for
 *      action verbs / question words.
 *   5. Average; squash through tanh to land in [0, 1].
 *
 * Deterministic — same input always yields the same output. No
 * dependencies. Matches `buhera_substrate::embed_text` in Rust.
 */
export function embedText(content) {
  if (!content) return new SCoord(0, 0, 0);
  const trimmed = content.trim().toLowerCase();
  if (!trimmed) return new SCoord(0, 0, 0);

  const tokens = tokenize(trimmed);
  if (!tokens.length) return new SCoord(0, 0, 0);

  let sumK = 0;
  let sumT = 0;
  let sumE = 0;
  let weight = 0;

  for (const tok of tokens) {
    const isStop = STOPWORDS.has(tok);
    const w = isStop ? 0.05 : 1.0;
    weight += w;

    const [hk, ht, he] = tokenAxes(tok);
    sumK += w * hk;
    sumT += w * ht;
    sumE += w * he;

    if (TEMPORAL.has(tok)) sumT += 0.6;
    if (ACTIONS.has(tok)) sumE += 0.5;
  }

  if (weight === 0) return new SCoord(0, 0, 0);

  return new SCoord(
    clamp01(squash(sumK / weight)),
    clamp01(squash(sumT / weight)),
    clamp01(squash(sumE / weight)),
  );
}

/**
 * Score how many tokens of `query` literally appear in `target`.
 *
 * Returns a number in [0, 1]: 0 = no shared tokens, 1 = every query
 * token appears in the target. Used by the find-hits re-ranker as a
 * literal-overlap booster on top of the embedding distance.
 */
export function tokenOverlap(query, target) {
  if (!query || !target) return 0;
  const qs = new Set(
    tokenize(query.toLowerCase()).filter((t) => t.length >= 3)
  );
  if (!qs.size) return 0;
  const ts = new Set(
    tokenize(target.toLowerCase()).filter((t) => t.length >= 3)
  );
  if (!ts.size) return 0;
  let hit = 0;
  for (const q of qs) if (ts.has(q)) hit++;
  return hit / qs.size;
}

// ─────────────────────────────────────────────────────────────────────
//  Domain-specific embeddings (kept for the protein demo).
// ─────────────────────────────────────────────────────────────────────

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

  if (typeof props.length === "number") {
    rk = (rk + Math.log(props.length + 1) / 12) % 1;
  }
  if (props.role) {
    const roleSeed = hashSeed(String(props.role));
    rt = (rt + (roleSeed / 0xffffffff)) % 1;
  }
  if (Array.isArray(props.domains)) {
    re = (re + Math.log(props.domains.length + 1) / 8) % 1;
  }

  return new SCoord(Math.abs(rk) % 1, Math.abs(rt) % 1, Math.abs(re) % 1);
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

  return new SCoord(Math.abs(rk) % 1, Math.abs(rt) % 1, Math.abs(re) % 1);
}

// ─────────────────────────────────────────────────────────────────────
//  Fisher metric.
// ─────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────
//  Ternary addressing.
// ─────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────
//  Backward navigation.
// ─────────────────────────────────────────────────────────────────────

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
