/**
 * Partition coordinates (n, l, m, s) — node identity in the SRN address space.
 *
 * Derived from SO(3) representation theory (SRN paper §2.2):
 *   n >= 1        — partition depth (shell)
 *   0 <= l < n    — structural complexity (angular-momentum index)
 *   -l <= m <= l  — orientation index
 *   s in {-1, +1} — residue parity (spin)
 *
 * Shell capacity: C(n) = 2n^2 (each shell n holds exactly 2n^2 valid addresses).
 *
 * Ported from `crates/srn-node/src/coords.rs` (the reference implementation).
 */
/**
 * Construct a validated PartitionCoords, enforcing the angular-momentum
 * constraints. Returns a typed error rather than throwing (core discipline).
 */
export function makeCoords(n, l, m, s) {
    if (!Number.isInteger(n) || n < 1) {
        return { kind: "invalid-coord", message: `n must be an integer >= 1, got ${n}` };
    }
    if (!Number.isInteger(l) || l < 0 || l >= n) {
        return { kind: "invalid-coord", message: `l must be an integer in [0, n), got l=${l} n=${n}` };
    }
    if (!Number.isInteger(m) || Math.abs(m) > l) {
        return { kind: "invalid-coord", message: `|m| must be <= l, got m=${m} l=${l}` };
    }
    if (s !== 1 && s !== -1) {
        return { kind: "invalid-coord", message: `s must be +1 or -1, got ${s}` };
    }
    return { n, l, m, s };
}
/** Type guard: did makeCoords succeed? */
export function isCoordError(x) {
    return x.kind === "invalid-coord";
}
/** Build coords from a Coord tuple, or return the typed error. */
export function coordsFromTuple(c) {
    const [n, l, m, s] = c;
    return makeCoords(n, l, m, s);
}
/** The tuple form of a coordinate (for the public API surface). */
export function coordTuple(c) {
    return [c.n, c.l, c.m, c.s];
}
/** The minimal-depth reference node (1, 0, 0, +1) — the "chromebook reference". */
export function referenceCoords() {
    return { n: 1, l: 0, m: 0, s: 1 };
}
/** Shell capacity at depth n: C(n) = 2n^2. */
export function shellCapacity(n) {
    return 2 * n * n;
}
/** Enumerate all valid coordinates at depth n (the full shell). */
export function shell(n) {
    const out = [];
    for (let l = 0; l < n; l++) {
        for (let m = -l; m <= l; m++) {
            out.push({ n, l, m, s: -1 });
            out.push({ n, l, m, s: 1 });
        }
    }
    return out;
}
/** Compact content-addressing key, e.g. "(2,1,0,+)". */
export function coordKey(c) {
    return `(${c.n},${c.l},${c.m},${c.s > 0 ? "+" : "-"})`;
}
/** Structural equality of two coordinates. */
export function coordsEqual(a, b) {
    return a.n === b.n && a.l === b.l && a.m === b.m && a.s === b.s;
}
//# sourceMappingURL=coords.js.map